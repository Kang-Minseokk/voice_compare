"""Phase B1 — 모음 formant (F1/F2) 분석기 (robust 버전).

개선점:
  1. alignment center → 윈도우 내 energy peak으로 재중심화 (자음/침묵 회피)
  2. 슬라이딩 LPC (30ms win, 10ms hop) 측정 후 median (numerical outlier 제거)
  3. F1/F2/F3 동시 추출 + 주파수 범위 검증 (F3가 F2 자리에 끼는 현상 방지)

목적: 고려인 화자가 자주 혼동하는 ㅓ/ㅗ, ㅡ/ㅜ 같은 모음 차이 진단.
"""

from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from scipy.signal import lfilter

from alignment import Alignment
from analyzers.base import AnalyzerResult, SyllableScore
from audio_utils import (
    TARGET_SR,
    extract_window_at_sample,
    find_energy_peak_sample,
)


JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
    "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

# 한국어 모음의 일반적 formant 범위 (Hz). 범위 밖이면 분석기가 잘못 잡은 것
F1_RANGE = (200.0, 1100.0)
F2_RANGE = (700.0, 2700.0)
F3_RANGE = (1800.0, 3800.0)
MIN_F2_F1_GAP = 200.0


def _get_vowel(syllable: str) -> Optional[str]:
    if len(syllable) != 1:
        return None
    code = ord(syllable) - 0xAC00
    if code < 0 or code >= 11172:
        return None
    return JUNG[(code % 588) // 28]


def _lpc_formants(audio: np.ndarray, order: int = 18) -> np.ndarray:
    """단일 윈도우에서 valid한 formant 후보들을 정렬해서 반환 (모두 Hz)."""
    if audio.size < order * 2:
        return np.array([])

    audio = lfilter([1.0, -0.97], [1.0], audio)
    audio = audio * np.hamming(len(audio))

    try:
        a = librosa.lpc(audio, order=order)
    except Exception:
        return np.array([])

    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    if len(roots) == 0:
        return np.array([])

    freqs = np.angle(roots) * (TARGET_SR / (2 * np.pi))
    bws = -(TARGET_SR / np.pi) * np.log(np.abs(roots) + 1e-12)
    valid = (freqs > 90) & (freqs < 4000) & (bws < 400)
    return np.sort(freqs[valid])


def _assign_formants(candidates: np.ndarray) -> Tuple[float, float, float]:
    """후보 주파수들을 F1/F2/F3에 범위 검증과 함께 배정."""
    f1 = f2 = f3 = np.nan
    for f in candidates:
        if np.isnan(f1) and F1_RANGE[0] <= f <= F1_RANGE[1]:
            f1 = f
        elif np.isnan(f2) and F2_RANGE[0] <= f <= F2_RANGE[1] and f > f1 + MIN_F2_F1_GAP:
            f2 = f
        elif np.isnan(f3) and F3_RANGE[0] <= f <= F3_RANGE[1] and f > f2:
            f3 = f
    return f1, f2, f3


def _measure_formants_robust(
    segment: torch.Tensor, win_ms: float = 30.0, hop_ms: float = 10.0
) -> Tuple[float, float, float]:
    """슬라이딩 LPC 측정 후 각 formant의 median."""
    audio = segment.numpy().astype(np.float64)
    win = int(win_ms / 1000 * TARGET_SR)
    hop = int(hop_ms / 1000 * TARGET_SR)
    if audio.size < win:
        return np.nan, np.nan, np.nan

    f1s, f2s, f3s = [], [], []
    for i in range(0, audio.size - win + 1, hop):
        candidates = _lpc_formants(audio[i : i + win])
        if candidates.size == 0:
            continue
        f1, f2, f3 = _assign_formants(candidates)
        if not np.isnan(f1):
            f1s.append(f1)
        if not np.isnan(f2):
            f2s.append(f2)
        if not np.isnan(f3):
            f3s.append(f3)

    def med(xs):
        return float(np.median(xs)) if xs else np.nan

    return med(f1s), med(f2s), med(f3s)


def _hz_to_bark(hz: float) -> float:
    return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500) ** 2)


def _formant_score(
    gt: Tuple[float, float], sample: Tuple[float, float], scale: float = 2.0
) -> float:
    if any(np.isnan(v) for v in (*gt, *sample)):
        return 0.0
    gt_bark = np.array([_hz_to_bark(gt[0]), _hz_to_bark(gt[1])])
    sm_bark = np.array([_hz_to_bark(sample[0]), _hz_to_bark(sample[1])])
    dist = float(np.linalg.norm(gt_bark - sm_bark))
    return float(np.exp(-dist / scale))


def _f(value: float) -> Optional[float]:
    return None if np.isnan(value) else float(value)


def analyze(
    gt_audio: torch.Tensor,
    sample_audio: torch.Tensor,
    gt_alignment: Alignment,
    sample_alignment: Alignment,
    text: str,
    half_width_ms: float = 80.0,
) -> AnalyzerResult:
    per_syllable = []

    for gt_span, sm_span in zip(gt_alignment.spans, sample_alignment.spans):
        gt_peak = find_energy_peak_sample(gt_audio, gt_span.center_frame)
        sm_peak = find_energy_peak_sample(sample_audio, sm_span.center_frame)
        gt_seg = extract_window_at_sample(gt_audio, gt_peak, half_width_ms)
        sm_seg = extract_window_at_sample(sample_audio, sm_peak, half_width_ms)

        gt_f1, gt_f2, gt_f3 = _measure_formants_robust(gt_seg)
        sm_f1, sm_f2, sm_f3 = _measure_formants_robust(sm_seg)

        score = _formant_score((gt_f1, gt_f2), (sm_f1, sm_f2))

        per_syllable.append(
            SyllableScore(
                char=gt_span.char,
                score=score,
                details={
                    "vowel": _get_vowel(gt_span.char),
                    "gt_f1": _f(gt_f1),
                    "gt_f2": _f(gt_f2),
                    "gt_f3": _f(gt_f3),
                    "sample_f1": _f(sm_f1),
                    "sample_f2": _f(sm_f2),
                    "sample_f3": _f(sm_f3),
                },
            )
        )

    valid = [s.score for s in per_syllable if s.score > 0]
    overall = sum(valid) / len(valid) if valid else 0.0

    return AnalyzerResult(
        name="formants",
        score=overall,
        per_syllable=per_syllable,
        details={
            "window_ms": half_width_ms * 2,
            "metric": "Bark-space distance of (F1,F2), score=exp(-d/2)",
            "robustness": "energy-peak recenter + sliding-window median + F1/F2/F3 range check",
        },
    )
