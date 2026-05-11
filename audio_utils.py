"""공용 오디오 유틸: 로딩, 샘플레이트, frame↔sample 변환, 윈도우 추출."""

import numpy as np
import torch
import torchaudio


TARGET_SR = 16000
FRAME_STRIDE_SEC = 0.02  # wav2vec2 base stride: 20ms / frame


def load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0)


def frame_to_sample(frame_idx: int) -> int:
    return int(frame_idx * FRAME_STRIDE_SEC * TARGET_SR)


def frame_to_time(frame_idx: int) -> float:
    return frame_idx * FRAME_STRIDE_SEC


def extract_window(
    audio: torch.Tensor, center_frame: int, half_width_ms: float = 60.0
) -> torch.Tensor:
    """alignment center를 기준으로 ±half_width_ms ms 윈도우 추출."""
    half_samples = int(half_width_ms / 1000 * TARGET_SR)
    center_sample = frame_to_sample(center_frame)
    start = max(0, center_sample - half_samples)
    end = min(audio.size(0), center_sample + half_samples)
    return audio[start:end]


def find_energy_peak_sample(
    audio: torch.Tensor,
    center_frame: int,
    search_half_ms: float = 100.0,
    win_ms: float = 20.0,
    hop_ms: float = 5.0,
) -> int:
    """center_frame 부근 ±search_half_ms 안에서 RMS 에너지가 가장 큰 sample 위치.

    alignment center가 자음/침묵 구간에 떨어졌을 때 모음 nucleus 쪽으로 재조정.
    """
    half_samples = int(search_half_ms / 1000 * TARGET_SR)
    center_sample = frame_to_sample(center_frame)
    start = max(0, center_sample - half_samples)
    end = min(audio.size(0), center_sample + half_samples)
    region = audio[start:end].numpy()

    win = int(win_ms / 1000 * TARGET_SR)
    hop = int(hop_ms / 1000 * TARGET_SR)
    if len(region) < win:
        return center_sample

    n_steps = max(1, (len(region) - win) // hop + 1)
    energies = np.empty(n_steps)
    for i in range(n_steps):
        seg = region[i * hop : i * hop + win]
        energies[i] = np.sqrt(np.mean(seg ** 2))

    peak_idx = int(np.argmax(energies))
    return start + peak_idx * hop + win // 2


def extract_window_at_sample(
    audio: torch.Tensor, center_sample: int, half_width_ms: float = 60.0
) -> torch.Tensor:
    """sample 위치 기준 윈도우 추출."""
    half_samples = int(half_width_ms / 1000 * TARGET_SR)
    start = max(0, center_sample - half_samples)
    end = min(audio.size(0), center_sample + half_samples)
    return audio[start:end]
