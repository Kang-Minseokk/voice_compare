# 고려인 발음 교정 — 음성 비교 엔진

원어민 녹음(`gt`)과 학습자 녹음(`sample`)을 비교해서 **어느 차원에서 어떻게 다른지** 진단형으로 짚어주는 시스템.

## 핵심 철학

> 단일 종합 점수가 아니라, **음향학이 분리해둔 6개 차원을 각각 측정**해서 고려인 학습자가 어려워하는 정확한 오류 유형을 음절 단위로 진단한다.

기존 발음 평가 도구의 한계:
- 종합 점수 70점만 알려줌 → 사용자는 *무엇을* 고쳐야 할지 모름
- "AI 평가"는 블랙박스 → 설명 불가능
- LLM 기반 e2e 평가는 작동하지만 *왜* 그 점수인지 못 말함

우리의 접근:
- 음향학(phonetics)이 이미 정립한 차원들(F1/F2/F0/VOT/RMS)을 분리해서 각각 측정
- 각 측정은 **명확한 phonetic 의미**를 가짐 (F1=혀높이, F0=피치, VOT=자음 변별 …)
- 사용자에게 *"점수 0.7입니다"*가 아니라 *"'쓰려요'에서 피치를 6 semitone 떨어뜨리셨네요"*라고 말할 수 있음

---

## 전체 흐름

```
입력: gt.wav, sample.wav, text("자꾸 배가 아프고 …")
  │
  ▼
[Step 1] wav2vec2 + Viterbi forced alignment
  → gt와 sample 각각에서 음절별 시간 위치 추출
  │
  ▼
[Step 2] 같은 음절끼리 6축 음향 분석
  ├─ (1) Formants (F1, F2)       — 모음 정체성
  ├─ (2) MFCC + DTW              — 스펙트럼 envelope
  ├─ (3) Pitch (F0)              — 피치/억양
  ├─ (4) Energy (RMS, dB)        — 강세/dynamics
  ├─ (5) VOT (Voice Onset Time)  — 자음 변별
  └─ (6) Coda (받침)             — 종성 약화/탈락
  │
  ▼
출력: 분석기별 점수 + 음절별 진단 + alignment 시간 정보
```

---

## Step 1 — wav2vec2 + Viterbi Forced Alignment

### 목적
모든 분석기가 의지하는 토대. **"오디오의 어디에 어떤 음절이 있는가"의 시간 매핑.**

### 왜 wav2vec2를 선택했는가

| 대안 | 문제점 |
|---|---|
| **STT 모델** | 발음을 정답으로 *교정해서* 디코딩. 사용자가 ㅈ을 ㅊ으로 발음해도 "ㅈ"이라고 들음. *오류 정보 자체가 소실*. |
| **순수 음향 분석** (alignment 없이) | 음절 단위 진단 불가능. "어디가 틀렸는지" 짚을 수 없음. |
| **wav2vec2** ✓ | 분류 직전의 연속 표현(hidden state) + CTC 확률 분포. 정답 텍스트와 매칭하면서도 미묘한 발음 차이 보존. |

사용 모델: `kresnik/wav2vec2-large-xlsr-korean` (한국어 사전학습).

### 두 단계로 동작

| 단계 | 주체 | 역할 |
|---|---|---|
| 1 | **wav2vec2 (신경망)** | 매 frame마다 "이 소리가 어떤 한글 음절에 가까운가" 확률 (CTC logits) |
| 2 | **Viterbi (`torchaudio.functional.forced_align`)** | 텍스트가 주어졌을 때 그 순서를 만족하는 최적 frame 배치 |
| 3 | **`merge_tokens`** | 연속된 같은 토큰 묶기, blank 제거 → 음절별 (start, end) frame |

핵심: **"어디에 어떤 소리가 있는가"의 지식**은 wav2vec2가 사전학습으로 갖고 있고, **"텍스트 순서를 강제하는 능력"**은 Viterbi 알고리즘이 담당함.

### 발생한 문제: CTC의 "peaky" 특성

CTC는 학습 과정에서 blank 토큰을 많이 쓰도록 유도되는 결과, 음절이 **40ms 단일 spike**로만 잡힘. 실제 한국어 음절은 150~250ms.

```
이상적: |---- 자 ----|---- 꾸 ----|---- 배 ----|
실제:        *자          *꾸          *배           ← 좁은 spike만
```

### 해결책

1. **Alignment를 boundary가 아닌 pointer로 사용** — "여기 어딘가에 이 음절이 있다"는 위치 정보만 받음
2. **각 분석기가 자체 윈도우 확장** — 포인터 주변 ±60~100ms를 분석 목적에 맞게 추출
3. **Energy peak 재중심화** — 일부 분석기는 포인터 주변에서 RMS 최댓값 위치를 모음 nucleus로 가정해 윈도우를 재조정

→ 결과: alignment 자체를 안 고치고도 분석 품질 확보. 후속 어떤 분석기를 추가해도 이 인프라 재활용.

---

## Step 2 — 6축 음향 분석

각 분석기는 **다른 음향학적 차원을 봄**. 점수 합산이 아니라 *차원별 진단*이 목적.

공통 인터페이스 ([`analyzers/base.py`](analyzers/base.py)):
```python
def analyze(gt_audio, sample_audio, gt_alignment, sample_alignment, text)
    -> AnalyzerResult(
        name: str,
        score: float,                    # 종합 점수 (0~1)
        per_syllable: List[SyllableScore],  # 음절별 세부
        details: Dict[str, Any],         # 분석기별 자유 필드
    )
```

---

### (1) Formants (F1, F2) — 모음 정체성

**위치**: [`analyzers/formants.py`](analyzers/formants.py)

#### 음향학적 근거

- **F1 = 혀 높이**: 입을 크게 벌리면 높음(ㅏ ~750Hz), 작게 벌리면 낮음(ㅣ ~300Hz)
- **F2 = 혀 전후 + 입술 둥글기**: 앞쪽(ㅣ ~2200Hz), 뒤쪽 둥글게(ㅜ ~900Hz)
- (F1, F2) 좌표 = **"모음 공간"** — 모음마다 고유 위치

#### 고려인 학습자에게 왜 중요한가

러시아어 모음 체계에 **ㅓ, ㅡ가 없음**. 결과:
- ㅓ → ㅗ로 대체 ("어머니" → "오모니")
- ㅡ → ㅜ로 대체 ("음식" → "옴식")

이 대체는 **F2에서 정확히 측정**됨 (ㅓ F2≈1100 vs ㅗ F2≈800).

#### 발생한 문제

1. **Alignment center가 자음/침묵에 떨어짐** → 윈도우에 모음이 없어 측정 실패
2. **LPC root selection의 numerical instability** → F3가 F2 자리에 끼는 경우 (sample F2=3737Hz 같은 비현실적 값)
3. **단일 시점 측정 불안정**

#### 해결책 (robustness 강화)

1. **Energy peak 재중심화** — alignment center ±100ms 안에서 RMS 최대 위치로 윈도우 이동
2. **슬라이딩 LPC + median** — 30ms 윈도우를 10ms 간격으로 굴리며 여러 측정 후 *중앙값* (outlier 자동 제거)
3. **F1/F2/F3 동시 추출 + 범위 검증** — F1∈[200,1100], F2∈[700,2700], F3∈[1800,3800]. 후보 주파수를 phonetic 범위에 맞게 배정. F3를 F2로 오인하는 현상 차단.

#### 진단 가치

음절별 (F1, F2) 좌표를 **그대로 노출**. 앱에서:
- 모음 공간 시각화 (사용자의 ㅓ가 ㅗ 영역에 그려짐)
- *"혀를 약간 앞으로 빼서 ㅓ를 발음해보세요"* 같은 구체적 articulatory 가이드 가능

---

### (2) MFCC + DTW — 전반적 스펙트럼 envelope

**위치**: [`analyzers/mfcc_dtw.py`](analyzers/mfcc_dtw.py)

#### 음향학적 근거

- **MFCC**: 인간 청각의 perceptual 곡선을 반영한 13차원 spectral 표현. 음성 인식 표준 feature.
- **DTW**: 시간 신축을 허용한 시퀀스 정렬. 발화 속도 차이를 흡수.

#### 왜 이 분석기를 추가했는가

**Formants의 약점 보완 — cross-validation**:
- Formants는 단일 시점 측정이라 numerical 불안정 가능
- MFCC는 frame별 안정적 계산, 측정 실패 없음
- 두 분석기 둘 다 낮으면 *진짜 차이*, 한쪽만 낮으면 *측정 artifact*

#### 발견한 특성

- 점수 분포가 **좁음 (0.4~0.5)**
- 처음엔 "문제"로 보였으나 — 이게 **사실 정확한 신호**:
  - 사용자가 음소를 옳게 발음하면 MFCC는 *균일하게 비슷*하다고 정확히 말함
  - 좁은 분포 자체가 *"분절은 OK"*라는 진단

#### 해결책

- **CMVN (Cepstral Mean Variance Normalization)** — 음절 윈도우 내 평균/표준편차로 정규화. 화자/마이크 게인 영향 제거.

#### 진단 가치

"음소 윤곽의 거시적 일치도":
- **MFCC 좁고 균일하게 0.4-0.5** → "음소 발음 일관됨" → 문제는 prosody
- **특정 음절만 MFCC 급락** → "그 음절의 발음 자체가 다름"

---

### (3) Pitch (F0) — 음의 높낮이, 억양

**위치**: [`analyzers/pitch.py`](analyzers/pitch.py)

#### 음향학적 근거

- **F0 = 성대 진동 주파수** = 음의 높낮이
- F1/F2와 독립적 (F0는 성대, F1/F2는 성도)
- 의문문 끝 상승, 강조 음절 등 prosody의 핵심

#### 고려인 학습자에게 왜 중요한가

- 러시아어: **어휘 강세 (lexical stress)** — 단어마다 강세 위치 고정
- 한국어: **구문 억양 (phrasal intonation)** — 문장 전체의 곡선이 중요
- L1 운율 패턴 전이로 어색한 억양 발생
- "~요?"의 끝을 안 올리는 등의 패턴

#### 발생한 문제

1. **화자별 절대 F0가 다름** (남 ~120Hz, 여 ~200Hz) → 직접 비교 불가
2. **DTW만 사용 시 시간축 휨으로 *잘못된 강세 위치*가 흡수됨** → 어색한 prosody가 점수에 안 잡힘

#### 해결책

1. **Semitone 정규화**: 절대 Hz → 화자 중앙값 대비 semitone (음악의 반음 단위, perceptual)
2. **Dual sub-score 전략** (핵심 설계 결정):

| sub-score | 측정 | 시간축 | 잡는 것 |
|---|---|---|---|
| **alignment-locked** (primary) | 음절별 평균 F0 비교 | 고정 | "이 시점의 피치가 어긋남" → 강세 위치 오류 |
| **contour DTW** (secondary) | 전체 contour 모양 비교 | 휨 허용 | 전반적 멜로디 유사도 |

**왜 alignment-locked를 primary로 채택**:
- 교육 도구로서의 정렬 — "녕에서 피치가 떨어졌습니다" 음절 단위 피드백 가능
- forced alignment 인프라 활용 — wav2vec2로 깔아둔 시간 매핑 그대로 활용
- 러시아어 L1 오류 패턴과 fit — "잘못된 음절에 강세"는 시간 고정 비교로만 잡힘

#### 진단 가치

- 음절별 semitone 차이 → *"쓰려요의 '쓰'에서 4 semitone 낮춤"*
- **Tail slope** (발화 끝 기울기) → 의문문 검사 (gt 상승 vs sample 하강 등)

---

### (4) Energy (RMS, dB) — 강세 패턴

**위치**: [`analyzers/energy.py`](analyzers/energy.py)

#### 음향학적 근거

- **에너지 = 강세의 주요 acoustic cue**
- 강세 받은 음절은 높이뿐 아니라 *강하게* 발음됨

#### 고려인 학습자에게 왜 중요한가

- 러시아어식 강세 패턴이 한국어에 전이됨
- Pitch와 **함께 prosody의 양대축**
- Monotone speech (강세 없이 평탄) 검출

#### 발생한 문제

거의 없음. RMS는 robust한 측정.

#### 설계 결정

- **dB scale + 화자 평균 정규화** — 마이크 게인 차이 제거
- Pitch와 **동일한 dual sub-score 구조** (alignment-locked primary, contour DTW secondary) — 일관성 + cross-analyzer 검증

#### 진단 가치 & Cross-validation

**Pitch + Energy 동시 진단의 위력**:
- 둘 다 낮음 → **prosody 문제 확정**
- Pitch만 낮음 → 멜로디만 어색
- Energy만 낮음 → 강세 위치만 어색

실험에서 sample의 의도적 prosody 왜곡이 **두 분석기 모두에서 평행하게 잡힘** → 측정 신뢰성 검증.

---

### (5) VOT (Voice Onset Time) — 자음 변별 (평음/경음/격음)

**위치**: [`analyzers/vot.py`](analyzers/vot.py)

#### 음향학적 근거

- VOT = 자음 burst(파열) → 모음 voicing(성대 진동) 시작 사이 시간
- 한국어 stop 3-way 변별의 결정적 cue:

| 카테고리 | 음소 | VOT |
|---|---|---|
| 평음 | ㄱ, ㄷ, ㅂ, ㅈ | ~30-50 ms |
| 경음 | ㄲ, ㄸ, ㅃ, ㅉ | ~5-15 ms |
| 격음 | ㅋ, ㅌ, ㅍ, ㅊ | ~70-100 ms+ |

#### 고려인 학습자에게 왜 중요한가

**가장 핵심적인 자음 오류 원천**. 러시아어 자음은 **유성/무성 2분류**만 존재. 한국어 3분류 자체가 모국어 음운 체계에 없음 → 학습 난이도 최고.

#### 발생한 문제 (가장 흥미로운 디버깅 스토리)

**문제 1: ZCR 기반 voicing 검출 실패**
- 가정: voiced는 ZCR 낮음, unvoiced는 ZCR 높음
- 발견: **silence도 ZCR이 낮음** (진폭이 0에 가까워 부호변동이 거의 없음)
- 결과: closure 직후를 voiced로 오인 → VOT = 0 (가짜)

**문제 2: Relative threshold가 vowel-tail에서 baseline 계산**
- closure search가 RMS 최저점을 찾는데, 연속 발화에서는 그 "최저점"이 이전 음절의 vowel decay 안에 있을 수 있음
- threshold = baseline × 3이 너무 높아져 burst를 vowel onset으로 오인

**문제 3 (본질적 한계): 연속 발화에 closure가 없음**
- Citation form (음절 끊어 읽기): 명확한 silence-burst-voicing 구조
- Running speech (자연 연속 발화): 모음에서 자음으로 바로 흘러감
- 우리 sample(문장 발화)에서 6개 stop 중 5개가 이 상태

#### 해결책 (3단계)

1. **에너지 + ZCR 동시 조건** voiced 검출 → silence를 voiced로 오인하지 않음
2. **Walk-back from voicing** — burst를 따로 찾지 않고 voicing onset에서 거꾸로 walk-back하며 "에너지 있는 unvoiced 구간"의 길이를 측정. *이게 VOT의 정의 그 자체*.
3. **Honest unmeasurable** — Closure가 진짜 silence (RMS < 0.005)가 아니면 측정 거부, `None` 반환. **가짜 0ms를 출력하지 않음**.

#### 핵심 설계 철학: Honest Failure

대부분의 시스템은 측정 실패해도 어떤 값이라도 내놓음. 우리는 **명시적으로 "측정 불가"를 응답**.

- 가짜 값은 차후 분석/피드백에 독을 풀음
- "모름"을 정직하게 말하는 게 진단 시스템의 본질
- **App 설계 함의**: 단어 단위 task를 추가하면 VOT 효용 폭증

#### 진단 가치 (제한적이지만 명확)

측정된 케이스에 한해:
- *"당신의 ㅋ이 VOT 30ms로 ㄱ에 가까웠습니다"* 같은 정확한 자음 카테고리 피드백
- 단어/음절 단위 task에서 위력 최대

---

### (6) Coda (받침) — 종성 약화/탈락 경향

**위치**: [`analyzers/coda.py`](analyzers/coda.py)

#### 음향학적 근거

- 종성 안정성은 **발화 말미 에너지 유지**와 **종성 구간 길이**, **voicing residue(ZCR 기반)**로 근사 가능
- 완전한 음운 규칙 엔진이 아니라, 학습자 음성에서 반복되는 **받침 약화/탈락 경향**을 조기에 탐지

#### 설계 포인트

1. **3개 휴리스틱 결합**: `RMS decay ratio` + `duration ratio` + `voicing residue`
2. **fallback 분할 지원**: 정렬에서 nucleus/coda 분할 정보가 없을 때 프레임 비율로 안전한 대체 분할
3. **coda별 임계치 튜닝**: `ㄱ/ㅁ/ㅇ`의 drop/weak ratio를 분리해 과탐지 완화
4. **uncertain 상태 명시**: 너무 짧거나 신뢰도 낮은 구간은 점수만 억지로 내지 않고 `uncertain`으로 반환

#### 진단 가치

- *"받침이 거의 탈락(dropped)했습니다"* / *"약화(weakened)되었습니다"*를 음절 단위로 제시
- 사용자 피드백(`targeted_tips`)과 연결되어 어느 음절의 어떤 받침을 먼저 고칠지 바로 안내 가능

---

## 6축 조합의 의미

### 차원 매트릭스

| 음향학적 차원 | 분석기 | 잡는 오류 (고려인 패턴 매핑) |
|---|---|---|
| 분절 - 모음 quality | Formants | ㅓ/ㅗ, ㅡ/ㅜ 모음 substitution |
| 분절 - 스펙트럼 envelope | MFCC+DTW | 음소 일반적 정확도, 안정적 backbone |
| 분절 - 자음 변별 | VOT | 평음/경음/격음 (단어 task) |
| 분절 - 종성 안정성 | Coda | 받침 약화/탈락 경향 |
| 초분절 - 피치 | Pitch (F0) | 의문문 억양, 강세 위치 |
| 초분절 - 강세 | Energy (dB) | 강세 강도, dynamic range |

### 진단 패턴 → 사용자 피드백 매핑

| 점수 패턴 | 진단 | 피드백 |
|---|---|---|
| 분절 ↓ / 초분절 OK | 음소 발음 자체 문제 | "ㅓ를 ㅗ에 가깝게 발음하셨네요" |
| 분절 OK / 초분절 ↓ | Prosody 문제 | "발음은 정확한데 억양이 어색합니다" |
| Formants ↓ / MFCC OK | 모음만 문제 | "특정 모음에 집중하세요" |
| VOT ↓ / 다른 분절 OK | 자음 카테고리 오류 | "ㄱ을 ㅋ처럼 강하게 발음하셨네요" |
| Coda ↓ / 나머지 분절 OK | 받침 약화/탈락 | "끝소리를 짧고 또렷하게 남겨보세요" |
| Tail slope: gt 상승 / sample 하강 | 의문문 상승 누락 | "끝을 올려서 질문 형태로 발음해보세요" |

→ **단일 점수 시스템은 절대 못 하는 영역**. 6개 차원이 분리되어 있어서 각각 사용자 행동에 매핑됨.

### 고려인 학습자 오류 카탈로그 매핑

음성학 연구가 보고하는 러시아어 L1 → 한국어 L2 오류:

| # | 오류 | 우리 시스템 분석기 |
|---|---|---|
| 1 | 평음/경음/격음 미구분 | **VOT** (단어 task에서) |
| 2 | ㅓ/ㅗ, ㅡ/ㅜ 모음 혼동 | **Formants** |
| 3 | 의문문 끝 안 올림 | **Pitch tail slope** |
| 4 | 러시아어식 강세 위치 전이 | **Energy + Pitch (alignment-locked)** |
| 5 | 전반적 monotone | **Energy dynamic range + Pitch contour** |
| 6 | 받침 약화/탈락 | **Coda** |
| 7 | 종성 ㄹ trill | (Phase 후속) |

7개 중 6개 직접 커버.

---

## 정직한 한계 (Future Work)

1. **VOT의 connected speech 한계** — 단어/음절 단위 task 추가 시 효용 최대화
2. **음소(자모) 단위 alignment 미구현** — 현재 음절 단위. 초성/중성/종성 분리는 후속 과제
3. **종성 세부 음운 규칙 확장 필요** — 현재는 약화/탈락 탐지 중심, ㄹ trill/교체 규칙은 후속
4. **참고 발화 의존** — 원어민 녹음 대비 비교. 절대 표준 미보유
5. **단일 화자 비교의 본질적 노이즈** — 화자 정규화로 완화했지만 완전 제거는 어려움

각 한계는 후속 개선의 방향성으로 자연스럽게 연결.

---

## 시스템 구조

```
.
├── compare.py             # CLI orchestrator — compare(gt, sample, text) 함수가 핵심 API
├── api.py                 # FastAPI HTTP wrapper
├── batch_compare_wav.py   # 폴더 단위 병렬 비교 + 결과 JSON 생성
├── view_feedback.py       # Streamlit 피드백 대시보드
├── alignment.py           # wav2vec2 로딩 + Viterbi forced alignment
├── audio_utils.py         # 오디오 I/O, 윈도우 추출, energy peak 찾기
├── analyzers/
│   ├── base.py            # AnalyzerResult, SyllableScore 공통 타입
│   ├── formants.py        # (1) F1, F2
│   ├── mfcc_dtw.py        # (2) MFCC + DTW
│   ├── pitch.py           # (3) F0 (피치)
│   ├── energy.py          # (4) RMS (강세)
│   ├── vot.py             # (5) VOT (자음 변별)
│   └── coda.py            # (6) Coda (받침 약화/탈락)
├── hospital_0_ref.wav     # 샘플 — 원어민 녹음
├── hospital_0_real.wav    # 샘플 — 학습자 녹음 (의도적 prosody 왜곡)
└── README.md
```

분석기 추가/제거: `compare.py`의 `ENABLED_ANALYZERS` 리스트만 수정.

---

## 실행

### 환경

```
torch==2.4.1 torchaudio==2.4.1 transformers==4.45.2
librosa soundfile fastapi uvicorn python-multipart
streamlit pandas plotly
```

모델: `kresnik/wav2vec2-large-xlsr-korean` (첫 실행 시 ~1.2GB 자동 다운로드).

### CLI (단일 쌍 비교)

```
python compare.py
```

샘플 파일과 텍스트가 `compare.py`에 하드코딩되어 있음. 결과는 분석기별 점수 + 음절별 진단을 표 형태로 출력.

자주 쓰는 옵션 예시:
```
python compare.py \
  --gt-audio hospital_0_ref.wav \
  --sample-audio hospital_0_real.wav \
  --text "자꾸 배가 아프고 속이 쓰려요" \
  --align-backend wav2vec2
```

- `--align-backend mfa` 선택 시 MFA TextGrid를 우선 사용
- MFA 실패 시 기본값으로 `wav2vec2`로 자동 fallback (`--fail-on-mfa-error`로 중단 모드 가능)

### API

```
python api.py
```

→ `http://localhost:8000/docs` (Swagger UI에서 인터랙티브 테스트 가능)

요청:
```
POST /compare
multipart/form-data:
  gt_audio: <wav file>
  sample_audio: <wav file>
  text: "자꾸 배가 아프고 속이 쓰려요"
```

응답:
```json
{
  "overall_score": 0.49,
  "analyzers": {
    "formants": {
      "score": 0.461,
      "details": {...},
      "per_syllable": [
        {"char": "자", "score": 0.554, "details": {"vowel": "ㅏ", "gt_f1": 689, "gt_f2": 1248, ...}},
        ...
      ]
    },
    "pitch": {...},
    "energy": {...},
    "mfcc_dtw": {...},
    "vot": {...},
    "coda": {...}
  },
  "alignment": {
    "gt":     [{"char": "자", "start_sec": 0.90, "end_sec": 0.96}, ...],
    "sample": [{"char": "자", "start_sec": 0.54, "end_sec": 0.58}, ...]
  }
}
```

### 배치 실행 + 반복 리포트

1) 가상환경 활성화
```
source .venv/bin/activate
```

2) 배치 실행
```
python batch_compare_wav.py \
  --root . \
  --text-map-json ref_texts.json \
  --align-backend wav2vec2 \
  --max-workers 2 \
  --output-json batch_compare_results.json
```

3) 반복 리포트까지 함께 생성하려면
```
python batch_compare_wav.py \
  --root . \
  --text-map-json ref_texts.json \
  --output-json batch_compare_results.json \
  --repeat-report-json repeat_recording_report.json
```

- `batch_compare_results.json`: 샘플별 점수/피드백 전체 결과
- `repeat_recording_report.json`: 화자+문장 그룹 반복 녹음 통계(mean/std)

#### 배치 실행 전 환경 체크 (권장)

`batch_compare_wav.py`는 내부적으로 `compare.py -> alignment.py`를 import하므로
`torch`와 `torchaudio`가 같은 Python 환경에 설치되어 있어야 한다.

```
python -c "import torch, torchaudio; print('torch', torch.__version__)"
```

- 위 명령이 실패하면 해당 conda 환경에서 먼저 의존성을 설치
- 예: `python -m pip install torch torchaudio`

참고:
- 프로젝트 루트에 `batch_compare_results.json`이 이미 있더라도, 그것은 **과거 실행 산출물**일 수 있음
- 따라서 현재 환경 재실행 가능 여부는 위 import 체크로 확인하는 것을 권장

### 음조 분석 피드백 대시보드 (Streamlit)

1) 의존성 설치 (실행 환경에서 1회)
```
python -m pip install streamlit pandas plotly
```

2) 대시보드 실행
```
python -m streamlit run view_feedback.py
```

3) 브라우저에서 열기
- 기본 URL: `http://localhost:8501`

참고:
- `streamlit: command not found`가 나오면 `python -m streamlit ...` 형태로 실행
- `No module named 'plotly'`가 나오면 현재 환경에 `plotly`를 다시 설치

---

## 한 줄 요약

> **음향학이 분리해둔 6개 차원을 각각 측정해서 고려인 학습자가 어려워하는 정확한 오류 유형(모음 substitution, 자음 변별, 받침 약화/탈락, prosody 전이)을 음절 단위 진단으로 짚어주는 시스템. 각 차원은 명확한 phonetic 의미를 갖고, 실측을 통해 발견한 한계는 honest failure로 처리해 가짜 결과를 만들지 않는다.**
