"""wav2vec2 기반 forced alignment.

오디오 + 텍스트 → 음절별 frame 구간(start, end)을 반환한다.
hidden state, log_probs도 같이 반환해서 downstream 분석기가 재사용 가능.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from audio_utils import TARGET_SR


MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SyllableSpan:
    char: str
    start_frame: int
    end_frame: int

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
        spans.append(SyllableSpan(char=ch, start_frame=s, end_frame=e))

    return Alignment(spans=spans, hidden=hidden, log_probs=log_probs)
