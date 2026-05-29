"""wav2vec2 기반 forced alignment.

오디오 + 텍스트 → 음절별 frame 구간(start, end)을 반환한다.
hidden state, log_probs도 같이 반환해서 downstream 분석기가 재사용 가능.
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from audio_utils import FRAME_STRIDE_SEC, TARGET_SR


MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SyllableSpan:
    char: str
    start_frame: int
    end_frame: int
    onset: Optional[Tuple[int, int]] = None
    nucleus: Optional[Tuple[int, int]] = None
    coda: Optional[Tuple[int, int]] = None

    @property
    def center_frame(self) -> int:
        return (self.start_frame + self.end_frame) // 2


@dataclass
class Alignment:
    spans: List[SyllableSpan]
    hidden: torch.Tensor   # [T, D]
    log_probs: torch.Tensor  # [T, V]


_model = None
_processor = None


def _has_coda(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char) - 0xAC00
    if code < 0 or code >= 11172:
        return False
    jong = code % 28
    return jong != 0


def _heuristic_split_onset_nucleus_coda(
    char: str, start_frame: int, end_frame: int
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """음절 span을 onset/nucleus/coda로 휴리스틱 분할.

    - 종성이 있으면 onset:nucleus:coda ~= 25:50:25
    - 종성이 없으면 onset:nucleus ~= 35:65
    - 매우 짧은 span은 핵심 구간(nucleus) 위주로 안전하게 축약
    """
    if end_frame < start_frame:
        return None, None, None
    total = end_frame - start_frame + 1
    if total == 1:
        single = (start_frame, end_frame)
        return single, single, None

    has_coda = _has_coda(char)
    if not has_coda:
        onset_len = max(1, int(round(total * 0.35)))
        onset_len = min(onset_len, total - 1)
        nucleus_len = total - onset_len
        o_s = start_frame
        o_e = o_s + onset_len - 1
        n_s = o_e + 1
        n_e = end_frame
        return (o_s, o_e), (n_s, n_e), None

    # 종성이 있는 경우
    if total < 3:
        onset_len = 1
        nucleus_len = total - onset_len
        coda_len = 0
    else:
        onset_len = max(1, int(round(total * 0.25)))
        coda_len = max(1, int(round(total * 0.25)))
        nucleus_len = total - onset_len - coda_len
        if nucleus_len < 1:
            # nucleus 최소 1프레임 보장
            if onset_len >= coda_len and onset_len > 1:
                onset_len -= 1
            elif coda_len > 1:
                coda_len -= 1
            nucleus_len = total - onset_len - coda_len

    o_s = start_frame
    o_e = o_s + onset_len - 1
    n_s = o_e + 1
    n_e = n_s + nucleus_len - 1
    c_s = n_e + 1
    c_e = end_frame
    coda = (c_s, c_e) if c_s <= c_e else None
    return (o_s, o_e), (n_s, n_e), coda


def _make_syllable_span(char: str, start_frame: int, end_frame: int) -> SyllableSpan:
    onset, nucleus, coda = _heuristic_split_onset_nucleus_coda(char, start_frame, end_frame)
    return SyllableSpan(
        char=char,
        start_frame=start_frame,
        end_frame=end_frame,
        onset=onset,
        nucleus=nucleus,
        coda=coda,
    )


def load_model():
    global _model, _processor
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return _model, _processor


@torch.no_grad()
def _extract(audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model, processor = load_model()
    inputs = processor(
        audio.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"
    )
    input_values = inputs.input_values.to(DEVICE)
    out = model(input_values, output_hidden_states=True)
    log_probs = F.log_softmax(out.logits, dim=-1).squeeze(0).cpu()
    hidden = out.hidden_states[-1].squeeze(0).cpu()
    return log_probs, hidden


def _text_to_tokens(text: str, processor) -> Tuple[List[Tuple[str, int]], str]:
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    delim = getattr(tokenizer, "word_delimiter_token", "|")

    tokens = []
    for ch in text:
        key = delim if ch == " " else ch
        if key in vocab:
            tokens.append((key, vocab[key]))
    return tokens, delim


def _forced_align_raw(
    log_probs: torch.Tensor, token_ids: List[int], blank_id: int
) -> List[Tuple[int, int]]:
    targets = torch.tensor([token_ids], dtype=torch.int32)
    log_probs_b = log_probs.unsqueeze(0)
    alignments, scores = torchaudio.functional.forced_align(
        log_probs_b, targets, blank=blank_id
    )
    spans = torchaudio.functional.merge_tokens(
        alignments[0], scores[0], blank=blank_id
    )
    return [(s.start, s.end) for s in spans]


def align(audio: torch.Tensor, text: str) -> Alignment:
    model, processor = load_model()
    blank_id = (
        model.config.pad_token_id if model.config.pad_token_id is not None else 0
    )

    log_probs, hidden = _extract(audio)
    tokens, delim = _text_to_tokens(text, processor)

    if not tokens:
        return Alignment(spans=[], hidden=hidden, log_probs=log_probs)

    token_ids = [tid for _, tid in tokens]
    raw_spans = _forced_align_raw(log_probs, token_ids, blank_id)

    spans = []
    for (ch, _), (s, e) in zip(tokens, raw_spans):
        if ch == delim:
            continue
        spans.append(_make_syllable_span(ch, s, e))

    return Alignment(spans=spans, hidden=hidden, log_probs=log_probs)


def _sec_to_start_frame(sec: float) -> int:
    return max(0, int(round(sec / FRAME_STRIDE_SEC)))


def _sec_to_end_frame(sec: float) -> int:
    # TextGrid xmax는 구간 끝(배타 경계)에 가까우므로 inclusive end로 변환.
    return max(0, int(round(sec / FRAME_STRIDE_SEC)) - 1)


def _extract_word_intervals(textgrid_content: str) -> List[Tuple[float, float, str]]:
    words_tier_match = re.search(
        r'name = "words"(.*?)(?:\n\s*item \[\d+\]:|\Z)',
        textgrid_content,
        flags=re.S,
    )
    if not words_tier_match:
        return []
    words_chunk = words_tier_match.group(1)
    interval_pattern = re.compile(
        r"intervals \[\d+\]:\s*"
        r"xmin = ([0-9.]+)\s*"
        r"xmax = ([0-9.]+)\s*"
        r'text = "(.*?)"',
        flags=re.S,
    )
    intervals = []
    for m in interval_pattern.finditer(words_chunk):
        xmin = float(m.group(1))
        xmax = float(m.group(2))
        text = m.group(3)
        intervals.append((xmin, xmax, text))
    return intervals


def _split_word_to_syllable_spans(
    word: str, start_frame: int, end_frame: int
) -> List[SyllableSpan]:
    chars = [ch for ch in word if ch != " "]
    if not chars or end_frame < start_frame:
        return []
    duration = end_frame - start_frame + 1
    spans: List[SyllableSpan] = []
    for i, ch in enumerate(chars):
        s = start_frame + int(i * duration / len(chars))
        e = start_frame + int((i + 1) * duration / len(chars)) - 1
        e = max(s, e)
        spans.append(_make_syllable_span(ch, s, e))
    return spans


def alignment_from_textgrid(textgrid_path: str, text: Optional[str] = None) -> Alignment:
    """MFA TextGrid(words tier)를 기존 Alignment 포맷으로 변환.

    - words tier를 읽어 단어 구간을 가져온 뒤, 단어 내부를 문자(음절) 개수로 균등 분할한다.
    - text를 전달하면 해당 텍스트의 공백 제외 문자 순서와 길이로 최종 보정한다.
    """
    content = Path(textgrid_path).read_text(encoding="utf-8")
    intervals = _extract_word_intervals(content)
    spans: List[SyllableSpan] = []
    for xmin, xmax, word in intervals:
        if not word.strip():
            continue
        s = _sec_to_start_frame(xmin)
        e = _sec_to_end_frame(xmax)
        spans.extend(_split_word_to_syllable_spans(word, s, e))

    if text is not None:
        target_chars = [ch for ch in text if ch != " "]
        n = min(len(target_chars), len(spans))
        adjusted = []
        for i in range(n):
            sp = spans[i]
            adjusted.append(_make_syllable_span(target_chars[i], sp.start_frame, sp.end_frame))
        spans = adjusted

    # Hidden/logits는 TextGrid 변환 경로에서는 사용하지 않으므로 empty tensor로 채움.
    empty = torch.empty((0, 0), dtype=torch.float32)
    return Alignment(spans=spans, hidden=empty, log_probs=empty)
