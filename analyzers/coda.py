"""Phase A — 종성(coda) 약화/탈락 경향 분석기.

휴리스틱 지표:
  1) RMS decay ratio: coda RMS / nucleus-tail RMS
  2) duration ratio: coda 길이 / 음절 길이
  3) voicing residue: coda 구간 ZCR (유성 종성에서 특히 중요)

목적:
  완전한 음운 규칙 엔진이 아니라, 종성 약화/탈락 경향을 조기에 탐지한다.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from alignment import Alignment, SyllableSpan
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import TARGET_SR, frame_to_sample


JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
    "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ",
]

VOICED_CODA = {"ㄴ", "ㄹ", "ㅁ", "ㅇ"}
SCORE_SCALE = 1.2
DROP_RATIO = 0.4
WEAK_RATIO = 0.7
# Phase C-3: coda별 튜닝(우선 ㄱ/ㅁ/ㅇ 분리)
DROP_RATIO_BY_CODA = {
    "ㄱ": 0.20,
    "ㅁ": 0.18,
    "ㅇ": 0.18,
}
WEAK_RATIO_BY_CODA = {
    "ㄱ": 0.75,
    "ㅁ": 0.55,
    "ㅇ": 0.70,
}
# fallback 기반 측정에서 duration이 사실상 동일할 때는 decay 중심 판정으로 전환
DURATION_UNINFORMATIVE_DIFF = 0.05
UNCERTAIN_MIN_DURATION = 0.02
UNCERTAIN_MIN_SYLLABLE_FRAMES = 1
UNCERTAIN_MIN_CODA_FRAMES = 0


def _get_coda_jamo(char: str) -> Optional[str]:
    if len(char) != 1:
        return None
    code = ord(char) - 0xAC00
    if code < 0 or code >= 11172:
        return None
    jong = code % 28
    if jong == 0:
        return None
    return JONG[jong]


def _frame_interval_to_samples(interval: Tuple[int, int]) -> Tuple[int, int]:
    s, e = interval
    start = frame_to_sample(s)
    end = frame_to_sample(e + 1)  # inclusive frame -> exclusive sample
    return max(0, start), max(start + 1, end)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def _zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def _fallback_nucleus_coda_frames(span: SyllableSpan) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """자모 분할 정보가 없을 때 음절 말미 기준으로 nucleus/coda를 휴리스틱 추정."""
    if span.end_frame < span.start_frame:
        return None
    total = span.end_frame - span.start_frame + 1
    if total == 1:
        # 최소 fallback: 같은 프레임을 nucleus/coda로 공유
        single = (span.start_frame, span.end_frame)
        return single, single
    if total == 2:
        return (span.start_frame, span.start_frame), (span.end_frame, span.end_frame)

    coda_len = max(1, int(round(total * 0.22)))
    coda_s = span.end_frame - coda_len + 1
    coda_e = span.end_frame

    nucleus_s = span.start_frame + max(1, int(round(total * 0.35)))
    nucleus_e = coda_s - 1
    if nucleus_e <= nucleus_s:
        nucleus_s = span.start_frame
        nucleus_e = max(span.start_frame, coda_s - 1)
    if nucleus_e <= nucleus_s:
        return None
    return (nucleus_s, nucleus_e), (coda_s, coda_e)


def _measure_coda_features(audio: torch.Tensor, span: SyllableSpan) -> Optional[dict]:
    fallback_used = False
    if span.coda is not None and span.nucleus is not None:
        coda_interval = span.coda
        nucleus_interval = span.nucleus
    else:
        fb = _fallback_nucleus_coda_frames(span)
        if fb is None:
            return None
        nucleus_interval, coda_interval = fb
        fallback_used = True

    coda_s, coda_e = _frame_interval_to_samples(coda_interval)
    nuc_s, nuc_e = _frame_interval_to_samples(nucleus_interval)
    syl_s, syl_e = _frame_interval_to_samples((span.start_frame, span.end_frame))

    audio_np = audio.numpy().astype(np.float64)
    coda_seg = audio_np[coda_s:coda_e]
    nuc_seg = audio_np[nuc_s:nuc_e]
    syl_len = max(1, syl_e - syl_s)
    coda_len = max(1, coda_e - coda_s)
    nucleus_len = max(1, nuc_e - nuc_s)

    # nucleus 마지막 40%를 anchor로 사용 (종성 직전 에너지 기준점)
    tail_start = nuc_s + int((nuc_e - nuc_s) * 0.6)
    nuc_tail = audio_np[tail_start:nuc_e]

    coda_rms = _rms(coda_seg)
    nuc_tail_rms = _rms(nuc_tail) + 1e-8
    decay_ratio = coda_rms / nuc_tail_rms
    duration_ratio = coda_len / syl_len
    voicing_residue = 1.0 - _zcr(coda_seg)  # 클수록 voiced 성분 경향

    return {
        "decay_ratio": float(decay_ratio),
        "duration_ratio": float(duration_ratio),
        "voicing_residue": float(voicing_residue),
        "fallback_used": fallback_used,
        "syllable_frames": int(max(1, span.end_frame - span.start_frame + 1)),
        "coda_frames": int(max(1, coda_interval[1] - coda_interval[0] + 1)),
        "nucleus_frames": int(max(1, nucleus_interval[1] - nucleus_interval[0] + 1)),
        "syllable_samples": int(syl_len),
        "coda_samples": int(coda_len),
        "nucleus_samples": int(nucleus_len),
    }


def _score_from_features(gt_f: dict, sm_f: dict) -> float:
    # 세 지표의 차이를 합쳐 점수화
    d_decay = abs(gt_f["decay_ratio"] - sm_f["decay_ratio"])
    d_dur = abs(gt_f["duration_ratio"] - sm_f["duration_ratio"])
    d_voice = abs(gt_f["voicing_residue"] - sm_f["voicing_residue"])
    dist = d_decay * 1.0 + d_dur * 2.0 + d_voice * 0.8
    return float(np.exp(-dist / SCORE_SCALE))


def _ratios_for_coda(coda_char: Optional[str]) -> Tuple[float, float]:
    drop_ratio = DROP_RATIO_BY_CODA.get(coda_char, DROP_RATIO)
    weak_ratio = WEAK_RATIO_BY_CODA.get(coda_char, WEAK_RATIO)
    weak_ratio = max(drop_ratio, weak_ratio)
    return drop_ratio, weak_ratio


def _status(gt_f: dict, sm_f: dict, coda_char: Optional[str]) -> str:
    # 약화/탈락 경향 rule (보수적으로)
    drop_ratio, weak_ratio = _ratios_for_coda(coda_char)
    both_fallback = gt_f.get("fallback_used") and sm_f.get("fallback_used")
    short_span = (
        min(gt_f.get("syllable_frames", 1), sm_f.get("syllable_frames", 1)) <= UNCERTAIN_MIN_SYLLABLE_FRAMES
        or min(gt_f.get("coda_frames", 1), sm_f.get("coda_frames", 1)) <= UNCERTAIN_MIN_CODA_FRAMES
    )
    too_short_duration = (
        gt_f["duration_ratio"] < UNCERTAIN_MIN_DURATION and sm_f["duration_ratio"] < UNCERTAIN_MIN_DURATION
    )
    if (both_fallback and short_span) or too_short_duration:
        return "uncertain"

    duration_uninformative = (
        both_fallback
        and abs(sm_f["duration_ratio"] - gt_f["duration_ratio"]) <= DURATION_UNINFORMATIVE_DIFF
    )
    # ㄱ/ㅁ/ㅇ은 coda별 튜닝값으로 decay 중심 판정
    if duration_uninformative and coda_char in DROP_RATIO_BY_CODA:
        if sm_f["decay_ratio"] < gt_f["decay_ratio"] * drop_ratio:
            return "dropped"
        if sm_f["decay_ratio"] < gt_f["decay_ratio"] * weak_ratio:
            return "weakened"
        return "ok"

    if sm_f["duration_ratio"] < gt_f["duration_ratio"] * drop_ratio and sm_f["decay_ratio"] < gt_f["decay_ratio"] * drop_ratio:
        return "dropped"
    if sm_f["duration_ratio"] < gt_f["duration_ratio"] * weak_ratio or sm_f["decay_ratio"] < gt_f["decay_ratio"] * weak_ratio:
        return "weakened"
    return "ok"


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
) -> AnalyzerResult:
    per_syllable: List[SyllableScore] = []

    for gt_span, sm_span in zip(gt_alignment.spans, sample_alignment.spans):
        coda_char = _get_coda_jamo(gt_span.char)
        if coda_char is None:
            continue

        gt_f = _measure_coda_features(gt_audio, gt_span)
        sm_f = _measure_coda_features(sample_audio, sm_span)
        if gt_f is None or sm_f is None:
            per_syllable.append(
                SyllableScore(
                    char=gt_span.char,
                    score=0.0,
                    details={"coda_char": coda_char, "status": "not_applicable"},
                )
            )
            continue

        score = _score_from_features(gt_f, sm_f)
        drop_ratio, weak_ratio = _ratios_for_coda(coda_char)
        status = _status(gt_f, sm_f, coda_char)
        per_syllable.append(
            SyllableScore(
                char=gt_span.char,
                score=score,
                details={
                    "coda_char": coda_char,
                    "voiced_coda": coda_char in VOICED_CODA,
                    "gt_fallback": gt_f["fallback_used"],
                    "sample_fallback": sm_f["fallback_used"],
                    "gt_decay_ratio": gt_f["decay_ratio"],
                    "sample_decay_ratio": sm_f["decay_ratio"],
                    "gt_duration_ratio": gt_f["duration_ratio"],
                    "sample_duration_ratio": sm_f["duration_ratio"],
                    "gt_voicing_residue": gt_f["voicing_residue"],
                    "sample_voicing_residue": sm_f["voicing_residue"],
                    "gt_syllable_frames": gt_f["syllable_frames"],
                    "sample_syllable_frames": sm_f["syllable_frames"],
                    "gt_coda_frames": gt_f["coda_frames"],
                    "sample_coda_frames": sm_f["coda_frames"],
                    "gt_nucleus_frames": gt_f["nucleus_frames"],
                    "sample_nucleus_frames": sm_f["nucleus_frames"],
                    "drop_ratio_used": drop_ratio,
                    "weak_ratio_used": weak_ratio,
                    "status": status,
                },
            )
        )

    valid = [s.score for s in per_syllable if s.details.get("status") not in {"not_applicable", "uncertain"}]
    if not per_syllable:
        overall = 1.0
        note = "no coda syllables in utterance"
    elif not valid:
        overall = 0.0
        note = "all coda measurements unavailable"
    else:
        overall = sum(valid) / len(valid)
        note = ""

    weakened = sum(1 for s in per_syllable if s.details.get("status") == "weakened")
    dropped = sum(1 for s in per_syllable if s.details.get("status") == "dropped")
    uncertain = sum(1 for s in per_syllable if s.details.get("status") == "uncertain")

    return AnalyzerResult(
        name="coda",
        score=overall,
        per_syllable=per_syllable,
        details={
            "metric": "coda RMS-decay + duration + voicing residue, score=exp(-dist/1.2)",
            "anchor": "coda vs nucleus-tail",
            "coda_count": len(per_syllable),
            "valid_count": len(valid),
            "weakened_count": weakened,
            "dropped_count": dropped,
            "uncertain_count": uncertain,
            "thresholds": {
                "drop_ratio": DROP_RATIO,
                "weak_ratio": WEAK_RATIO,
                "drop_ratio_by_coda": DROP_RATIO_BY_CODA,
                "weak_ratio_by_coda": WEAK_RATIO_BY_CODA,
                "uncertain_min_duration": UNCERTAIN_MIN_DURATION,
                "uncertain_min_syllable_frames": UNCERTAIN_MIN_SYLLABLE_FRAMES,
                "uncertain_min_coda_frames": UNCERTAIN_MIN_CODA_FRAMES,
                "duration_uninformative_diff": DURATION_UNINFORMATIVE_DIFF,
            },
            "note": note,
        },
    )
