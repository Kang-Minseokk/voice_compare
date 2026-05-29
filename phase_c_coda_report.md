# Phase C Coda Report

- Jobs: 28
- Coda tokens: 182
- Fallback tokens: both=173 one_side=9 none=0
- Status counts:
  - ok: 125 (68.68%)
  - weakened: 34 (18.68%)
  - dropped: 23 (12.64%)
  - uncertain: 0 (0.00%)
  - not_applicable: 0 (0.00%)

## By Coda

| coda | count | avg_score | weakened_rate | dropped_rate | uncertain_rate | avg_decay_diff |
|---|---:|---:|---:|---:|---:|---:|
| ㄱ | 35 | 0.403 | 28.57% | 17.14% | 0.00% | 4.465 |
| ㄴ | 42 | 0.739 | 2.38% | 0.00% | 0.00% | 0.388 |
| ㄶ | 14 | 0.646 | 35.71% | 0.00% | 0.00% | 0.628 |
| ㄷ | 7 | 0.746 | 0.00% | 0.00% | 0.00% | 0.384 |
| ㄹ | 14 | 0.389 | 50.00% | 0.00% | 0.00% | 10.032 |
| ㅁ | 28 | 0.157 | 32.14% | 42.86% | 0.00% | 3.753 |
| ㅂ | 14 | 0.716 | 0.00% | 0.00% | 0.00% | 0.420 |
| ㅅ | 7 | 0.522 | 0.00% | 0.00% | 0.00% | 2.267 |
| ㅇ | 14 | 0.251 | 14.29% | 35.71% | 0.00% | 2.914 |
| ㅌ | 7 | 0.584 | 0.00% | 0.00% | 0.00% | 0.925 |

## Next Tuning Suggestions

- uncertain_rate가 높은 coda는 fallback 규칙 재조정 또는 해당 케이스 제외 기준을 강화
- dropped_rate가 과도하게 높은 coda는 DROP_RATIO 완화(예: 0.4 -> 0.35) 여부 검토
- weakened_rate가 과도하게 높은 coda는 WEAK_RATIO 재조정(예: 0.7 -> 0.65) 검토