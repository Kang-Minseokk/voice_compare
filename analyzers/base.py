"""분석기 공통 결과 타입.

모든 분석기는 다음 시그니처를 따른다:
    analyze(gt_audio, sample_audio, gt_alignment, sample_alignment, text)
        -> AnalyzerResult
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SyllableScore:
    char: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    name: str
    score: float                          # 종합 점수 (0~1)
    per_syllable: List[SyllableScore]
    details: Dict[str, Any] = field(default_factory=dict)
