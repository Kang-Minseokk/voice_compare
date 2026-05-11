"""Phase B2 — MFCC + DTW 분석기.

각 음절 윈도우의 MFCC 시퀀스를 추출하고, DTW로 정렬한 뒤
평균 path cost를 점수로 변환.

LPC root 선택 같은 numerical 불안정성이 없어서 formants 분석기와 상호 검증 가능.
음향적 spectral envelope 유사도를 부드럽게 측정.
"""

from typing import Optional, Tuple

import librosa
import numpy as np
import torch

from alignment import Alignment
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import (
    TARGET_SR,
    extract_window_at_sample,
    find_energy_peak_sample,
)


N_MFCC = 13
N_FFT = 400          # 25ms
HOP_LENGTH = 160     # 10ms
SCORE_SCALE = 5.0    # exp(-dist/scale): dist ~5 → 0.37, dist ~10 → 0.13


def _mfcc(segment: torch.Tensor) -> np.ndarray:
    audio = segment.numpy().astype(np.float32)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    # CMVN (per-segment): 화자 톤/녹음 게인 영향 완화
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def _dtw_distance(gt_mfcc: np.ndarray, sm_mfcc: np.ndarray) -> Optional[float]:
    if gt_mfcc.shape[1] < 2 or sm_mfcc.shape[1] < 2:
        return None
    D, wp = librosa.sequence.dtw(X=gt_mfcc, Y=sm_mfcc, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def _score_from_distance(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return float(np.exp(-distance / SCORE_SCALE))


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
    half_width_ms: float = 100.0,
) -> AnalyzerResult:
    per_syllable = []

    for gt_span, sm_span in zip(gt_alignment.spans, sample_alignment.spans):
        gt_peak = find_energy_peak_sample(gt_audio, gt_span.center_frame)
        sm_peak = find_energy_peak_sample(sample_audio, sm_span.center_frame)
        gt_seg = extract_window_at_sample(gt_audio, gt_peak, half_width_ms)
        sm_seg = extract_window_at_sample(sample_audio, sm_peak, half_width_ms)

        gt_mfcc = _mfcc(gt_seg)
        sm_mfcc = _mfcc(sm_seg)
        dist = _dtw_distance(gt_mfcc, sm_mfcc)
        score = _score_from_distance(dist)

        per_syllable.append(
            SyllableScore(
                char=gt_span.char,
                score=score,
                details={
                    "dtw_distance": dist,
                    "gt_frames": int(gt_mfcc.shape[1]),
                    "sample_frames": int(sm_mfcc.shape[1]),
                },
            )
        )

    scores = [s.score for s in per_syllable]
    overall = sum(scores) / len(scores) if scores else 0.0

    return AnalyzerResult(
        name="mfcc_dtw",
        score=overall,
        per_syllable=per_syllable,
        details={
            "window_ms": half_width_ms * 2,
            "n_mfcc": N_MFCC,
            "metric": f"DTW path-normalized distance over CMVN-MFCC, "
                      f"score=exp(-d/{SCORE_SCALE})",
        },
    )
