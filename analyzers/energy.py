"""Phase A2 — Energy contour 분석기 (강세/dynamics).

전체 발화의 RMS energy를 dB로 변환 후 화자별 평균 기준으로 정규화하고,
음절별 평균 + 전체 contour DTW 두 sub-score를 계산.

목적: 러시아어 lexical stress 전이 등 잘못된 강세 위치, 동적 범위 부족 검출.
구조는 pitch.py와 평행 (dual sub-score, primary=alignment-locked).
"""

from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch

from alignment import Alignment
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import FRAME_STRIDE_SEC, TARGET_SR


ENERGY_FRAME_LEN = 512                            # 32ms at 16kHz
ENERGY_HOP = 256                                  # 16ms
ENERGY_HOP_SEC = ENERGY_HOP / TARGET_SR
SCORE_SCALE = 5.0                                 # dB 단위 거리 → 점수 변환


def _extract_energy_db(
    audio: torch.Tensor,
) -> Tuple[np.ndarray, Optional[float]]:
    """RMS → dB, 그리고 화자 평균 기준 zero-mean 정규화."""
    audio_np = audio.numpy().astype(np.float32)
    rms = librosa.feature.rms(
        y=audio_np, frame_length=ENERGY_FRAME_LEN, hop_length=ENERGY_HOP
    )[0]
    if rms.size == 0:
        return rms, None
    db = 20 * np.log10(rms + 1e-10)
    mean_db = float(np.mean(db))
    return db - mean_db, mean_db


def _dtw_db_distance(gt: np.ndarray, sm: np.ndarray) -> Optional[float]:
    if len(gt) < 2 or len(sm) < 2:
        return None
    D, wp = librosa.sequence.dtw(
        X=gt.reshape(1, -1), Y=sm.reshape(1, -1), metric="euclidean"
    )
    return float(D[-1, -1] / len(wp))


def _frame_to_energy_idx(frame_idx: int) -> int:
    time = frame_idx * FRAME_STRIDE_SEC
    return int(time / ENERGY_HOP_SEC)


def _per_syllable_db(
    db: np.ndarray, alignment: Alignment, half_width: int = 3
) -> List[Optional[float]]:
    out = []
    for span in alignment.spans:
        center = _frame_to_energy_idx(span.center_frame)
        start = max(0, center - half_width)
        end = min(len(db), center + half_width + 1)
        window = db[start:end]
        out.append(float(np.mean(window)) if window.size > 0 else None)
    return out


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
) -> AnalyzerResult:
    gt_db, gt_mean = _extract_energy_db(gt_audio)
    sm_db, sm_mean = _extract_energy_db(sample_audio)

    # Sub-score 1: contour DTW (energy envelope 모양 유사도)
    contour_dist = _dtw_db_distance(gt_db, sm_db)
    contour_score = (
        float(np.exp(-contour_dist / SCORE_SCALE))
        if contour_dist is not None
        else 0.0
    )

    # Sub-score 2: alignment-locked per-syllable (음절별 강세 일치도)
    gt_per = _per_syllable_db(gt_db, gt_alignment)
    sm_per = _per_syllable_db(sm_db, sample_alignment)

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
                details={"gt_db": g_v, "sample_db": s_v, "diff_db": diff},
            )
        )

    alignment_locked_score = (
        sum(s.score for s in per_syllable) / len(per_syllable)
        if per_syllable
        else 0.0
    )

    # Primary: pitch.py와 같은 근거 — 음절 단위 피드백, 강세 위치 어긋남 흡수 방지
    overall = alignment_locked_score

    return AnalyzerResult(
        name="energy",
        score=overall,
        per_syllable=per_syllable,
        details={
            "metric": f"alignment-locked per-syllable mean, score=exp(-d/{SCORE_SCALE})",
            "alignment_locked_score": alignment_locked_score,
            "contour_score": contour_score,
            "gt_mean_db": gt_mean,
            "sample_mean_db": sm_mean,
            "contour_dtw_distance_db": contour_dist,
        },
    )
