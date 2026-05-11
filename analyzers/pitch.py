"""Phase A1 — F0 contour 분석기 (피치/억양).

전체 발화의 F0 contour를 추출하고, 화자별 정규화(semitone) 후 DTW로 비교.
음소는 같지만 억양이 어색한 경우를 잡아낸다.

부가 측정:
  - 음절별 평균 F0 (semitone) — 어디서 피치가 튀었는지
  - 발화 끝 기울기 — 의문문 상승/평서문 하강 검사
"""

from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch

from alignment import Alignment
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import FRAME_STRIDE_SEC, TARGET_SR


F0_MIN = 70.0
F0_MAX = 400.0
PYIN_HOP = 512                       # 32ms at 16kHz
PYIN_HOP_SEC = PYIN_HOP / TARGET_SR
SCORE_SCALE = 3.0                    # semitone 단위 거리 → 점수 변환


def _extract_f0(audio: torch.Tensor) -> np.ndarray:
    f0, _, _ = librosa.pyin(
        audio.numpy(),
        sr=TARGET_SR,
        fmin=F0_MIN,
        fmax=F0_MAX,
        hop_length=PYIN_HOP,
    )
    return f0


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    nans = np.isnan(arr)
    if not nans.any() or nans.all():
        return arr.copy()
    not_nan = ~nans
    indices = np.arange(len(arr))
    result = arr.copy()
    result[nans] = np.interp(indices[nans], indices[not_nan], arr[not_nan])
    return result


def _normalize_semitones(f0: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return np.full_like(f0, np.nan), None
    median = float(np.median(voiced))
    result = np.full_like(f0, np.nan)
    valid = ~np.isnan(f0)
    result[valid] = 12 * np.log2(f0[valid] / median)
    return result, median


def _dtw_semitone_distance(gt: np.ndarray, sm: np.ndarray) -> Optional[float]:
    if len(gt) < 2 or len(sm) < 2:
        return None
    gt_f = _interpolate_nan(gt).reshape(1, -1)
    sm_f = _interpolate_nan(sm).reshape(1, -1)
    if np.isnan(gt_f).any() or np.isnan(sm_f).any():
        return None
    D, wp = librosa.sequence.dtw(X=gt_f, Y=sm_f, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def _frame_to_f0_idx(frame_idx: int) -> int:
    time = frame_idx * FRAME_STRIDE_SEC
    return int(time / PYIN_HOP_SEC)


def _per_syllable_f0(
    f0_st: np.ndarray, alignment: Alignment, half_width: int = 2
) -> List[Optional[float]]:
    out = []
    for span in alignment.spans:
        center = _frame_to_f0_idx(span.center_frame)
        start = max(0, center - half_width)
        end = min(len(f0_st), center + half_width + 1)
        window = f0_st[start:end]
        valid = window[~np.isnan(window)]
        out.append(float(np.mean(valid)) if len(valid) > 0 else None)
    return out


def _tail_slope(f0_st: np.ndarray, n_frames: int = 8) -> Optional[float]:
    """발화 끝 F0 기울기 (semitone/sec). 의문문 상승 검사용."""
    tail = f0_st[-n_frames:] if len(f0_st) >= n_frames else f0_st
    valid_mask = ~np.isnan(tail)
    if valid_mask.sum() < 2:
        return None
    times = np.arange(len(tail))[valid_mask] * PYIN_HOP_SEC
    values = tail[valid_mask]
    slope, _ = np.polyfit(times, values, 1)
    return float(slope)


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
) -> AnalyzerResult:
    gt_f0 = _extract_f0(gt_audio)
    sm_f0 = _extract_f0(sample_audio)

    gt_st, gt_median = _normalize_semitones(gt_f0)
    sm_st, sm_median = _normalize_semitones(sm_f0)

    # Sub-score 1: contour DTW (모양 유사도, 시간 휘게 허용)
    contour_dist = _dtw_semitone_distance(gt_st, sm_st)
    contour_score = (
        float(np.exp(-contour_dist / SCORE_SCALE))
        if contour_dist is not None
        else 0.0
    )

    # Sub-score 2: alignment-locked per-syllable (시점 일치도, 메인 점수)
    gt_per = _per_syllable_f0(gt_st, gt_alignment)
    sm_per = _per_syllable_f0(sm_st, sample_alignment)

    per_syllable = []
    for span, g_v, s_v in zip(gt_alignment.spans, gt_per, sm_per):
        if g_v is None or s_v is None:
            score, diff = 0.0, None
        else:
            diff = abs(g_v - s_v)
            score = float(np.exp(-diff / SCORE_SCALE))
        per_syllable.append(
            SyllableScore(
                char=span.char,
                score=score,
                details={"gt_st": g_v, "sample_st": s_v, "diff_st": diff},
            )
        )

    alignment_locked_score = (
        sum(s.score for s in per_syllable) / len(per_syllable)
        if per_syllable
        else 0.0
    )

    # Primary: alignment-locked (음절별 평균). 이유: 음절 단위 피드백 가능,
    # 잘못된 강세 위치를 흡수하지 않음, forced alignment 인프라 활용.
    overall = alignment_locked_score

    return AnalyzerResult(
        name="pitch",
        score=overall,
        per_syllable=per_syllable,
        details={
            "metric": f"alignment-locked per-syllable mean, score=exp(-d/{SCORE_SCALE})",
            "alignment_locked_score": alignment_locked_score,
            "contour_score": contour_score,
            "gt_median_hz": gt_median,
            "sample_median_hz": sm_median,
            "contour_dtw_distance_st": contour_dist,
            "gt_tail_slope_st_per_sec": _tail_slope(gt_st),
            "sample_tail_slope_st_per_sec": _tail_slope(sm_st),
        },
    )
