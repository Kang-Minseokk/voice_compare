"""Phase C2 — VOT (Voice Onset Time) 분석기.

stop 자음(평음/경음/격음)으로 시작하는 음절에서 VOT를 측정하고
gt와 sample을 비교. 평음/경음/격음 변별 정확도를 평가.

VOT = burst(에너지 갑작스러운 상승) → voicing(ZCR이 낮아져 voiced 진입)
한국어 VOT 분포:
  - 평음 (ㄱ,ㄷ,ㅂ,ㅈ): ~30-50ms
  - 경음 (ㄲ,ㄸ,ㅃ,ㅉ): ~5-15ms
  - 격음 (ㅋ,ㅌ,ㅍ,ㅊ): ~70-100ms+
"""

from typing import List, Optional

import numpy as np
import torch

from alignment import Alignment, SyllableSpan
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import TARGET_SR, frame_to_sample


CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
STOP_INITIALS = {"ㄱ", "ㄲ", "ㅋ", "ㄷ", "ㄸ", "ㅌ",
                 "ㅂ", "ㅃ", "ㅍ", "ㅈ", "ㅉ", "ㅊ"}

ANALYSIS_FRAME_MS = 10.0
ANALYSIS_HOP_MS = 5.0
LOOKBACK_MS = 200.0       # 음절 center 기준 뒤로 얼만큼 자음 영역 가정
LOOKAHEAD_MS = 150.0
ZCR_VOICED_THRESHOLD = 0.15
MIN_VOICED_FRAMES = 3      # sustained voicing 요구 (aspiration 흡수 방지)
SCORE_SCALE = 25.0        # VOT 차이 ms 단위 → 점수 (exp(-d/25))

# 진짜 closure로 인정할 최대 RMS (이보다 크면 연속 발화의 transition으로 간주).
# 연속 발화의 음절-간 transition이 실제 silence가 아닌 경우 측정 신뢰도 없음.
CLOSURE_SILENCE_FLOOR = 0.005


def _get_initial(syllable: str) -> Optional[str]:
    if len(syllable) != 1:
        return None
    code = ord(syllable) - 0xAC00
    if code < 0 or code >= 11172:
        return None
    return CHOSEONG[code // 588]


def _is_stop(syllable: str) -> bool:
    initial = _get_initial(syllable)
    return initial in STOP_INITIALS if initial else False


def _ms_to_samples(ms: float) -> int:
    return int(ms / 1000.0 * TARGET_SR)


def _short_time_rms(audio: np.ndarray) -> np.ndarray:
    frame_len = _ms_to_samples(ANALYSIS_FRAME_MS)
    hop = _ms_to_samples(ANALYSIS_HOP_MS)
    if audio.size < frame_len:
        return np.array([])
    n_frames = (audio.size - frame_len) // hop + 1
    out = np.empty(n_frames)
    for i in range(n_frames):
        seg = audio[i * hop : i * hop + frame_len]
        out[i] = np.sqrt(np.mean(seg ** 2))
    return out


def _short_time_zcr(audio: np.ndarray) -> np.ndarray:
    frame_len = _ms_to_samples(ANALYSIS_FRAME_MS)
    hop = _ms_to_samples(ANALYSIS_HOP_MS)
    if audio.size < frame_len:
        return np.array([])
    n_frames = (audio.size - frame_len) // hop + 1
    out = np.empty(n_frames)
    for i in range(n_frames):
        seg = audio[i * hop : i * hop + frame_len]
        out[i] = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
    return out


def _find_closure(rms: np.ndarray) -> Optional[int]:
    """영역 전반부 2/3에서 RMS 최저 프레임 (자음 closure 또는 utterance-initial silence)."""
    if rms.size < 3:
        return None
    search_end = max(3, int(len(rms) * 2 / 3))
    return int(np.argmin(rms[:search_end]))


def _detect_burst_frame(
    rms: np.ndarray, closure_frame: int, energy_threshold: float
) -> Optional[int]:
    """closure 이후 energy_threshold를 처음 초과하는 프레임."""
    for i in range(closure_frame + 1, len(rms)):
        if rms[i] > energy_threshold:
            return i
    return None


def _detect_voicing_frame(
    zcr: np.ndarray,
    rms: np.ndarray,
    after_frame: int,
    energy_threshold: float,
) -> Optional[int]:
    """sustained voicing 진입 프레임.

    voiced = (ZCR 낮음) AND (에너지 threshold 이상).
    energy 조건이 없으면 silence (저 ZCR + 저 RMS)를 voiced로 오인.
    """
    voiced_count = 0
    for i in range(after_frame, len(zcr)):
        is_voiced = (zcr[i] < ZCR_VOICED_THRESHOLD) and (rms[i] > energy_threshold)
        if is_voiced:
            voiced_count += 1
            if voiced_count >= MIN_VOICED_FRAMES:
                return i - (MIN_VOICED_FRAMES - 1)
        else:
            voiced_count = 0
    return None


def _measure_vot(
    audio: torch.Tensor,
    span: SyllableSpan,
    prev_end_sample: Optional[int],
) -> Optional[float]:
    center = frame_to_sample(span.center_frame)
    region_start = max(0, center - _ms_to_samples(LOOKBACK_MS))
    if prev_end_sample is not None:
        region_start = max(region_start, prev_end_sample)
    region_end = min(audio.size(0), center + _ms_to_samples(LOOKAHEAD_MS))
    region = audio[region_start:region_end].numpy().astype(np.float64)

    if region.size < _ms_to_samples(30):
        return None

    rms = _short_time_rms(region)
    zcr = _short_time_zcr(region)
    if rms.size == 0 or zcr.size == 0:
        return None

    closure_frame = _find_closure(rms)
    if closure_frame is None:
        return None
    baseline = rms[closure_frame]
    # 진짜 closure가 아니면 (연속 발화 transition) 측정 불가
    if baseline > CLOSURE_SILENCE_FLOOR:
        return None
    energy_threshold = max(baseline * 3.0, baseline + 0.002)

    # 1. voicing onset: sustained 저ZCR + 에너지 충족
    voicing_frame = _detect_voicing_frame(
        zcr, rms, after_frame=closure_frame + 1, energy_threshold=energy_threshold
    )
    if voicing_frame is None:
        return None

    # 2. voicing에서 뒤로 걸어가며 "에너지 있는 unvoiced 구간" 끝까지 찾음.
    #    여기까지가 VOT 영역 (burst + aspiration).
    consonant_start = voicing_frame
    for i in range(voicing_frame - 1, closure_frame, -1):
        if rms[i] > energy_threshold:
            consonant_start = i
        else:
            break

    return (voicing_frame - consonant_start) * ANALYSIS_HOP_MS


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
) -> AnalyzerResult:
    per_syllable: List[SyllableScore] = []
    gt_spans = gt_alignment.spans
    sm_spans = sample_alignment.spans

    for i, (gt_span, sm_span) in enumerate(zip(gt_spans, sm_spans)):
        if not _is_stop(gt_span.char):
            continue

        gt_prev = frame_to_sample(gt_spans[i - 1].end_frame) if i > 0 else None
        sm_prev = frame_to_sample(sm_spans[i - 1].end_frame) if i > 0 else None

        gt_vot = _measure_vot(gt_audio, gt_span, gt_prev)
        sm_vot = _measure_vot(sample_audio, sm_span, sm_prev)

        if gt_vot is None or sm_vot is None:
            score, diff = 0.0, None
        else:
            diff = abs(gt_vot - sm_vot)
            score = float(np.exp(-diff / SCORE_SCALE))

        per_syllable.append(
            SyllableScore(
                char=gt_span.char,
                score=score,
                details={
                    "initial": _get_initial(gt_span.char),
                    "gt_vot_ms": gt_vot,
                    "sample_vot_ms": sm_vot,
                    "diff_ms": diff,
                },
            )
        )

    valid_pairs = [s.score for s in per_syllable if s.details["diff_ms"] is not None]

    if not per_syllable:
        # 발화에 stop 자음 없음 → 분석기 미적용. overall에 영향 없도록 1.0.
        overall = 1.0
        note = "no stop consonants in utterance"
    elif not valid_pairs:
        overall = 0.0
        note = "all VOT measurements failed"
    else:
        overall = sum(valid_pairs) / len(valid_pairs)
        note = ""

    return AnalyzerResult(
        name="vot",
        score=overall,
        per_syllable=per_syllable,
        details={
            "metric": f"|VOT_gt - VOT_sample| ms, score=exp(-d/{SCORE_SCALE})",
            "stop_count": len(per_syllable),
            "valid_count": len(valid_pairs),
            "note": note,
        },
    )
