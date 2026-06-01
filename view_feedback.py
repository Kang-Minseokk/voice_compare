"""batch_compare_results.json 시각화 대시보드 (Streamlit).

실행:
  python -m streamlit run view_feedback.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


REAL_RE = re.compile(r"hospital_(\d+)-(\d+)(?:-[A-Za-z0-9_]+)?_real\.wav$")
QUOTED_RE = re.compile(r"(['\"‘’“”])([^'\"‘’“”]+)(['\"‘’“”])")
HANGUL_RE = re.compile(r"[가-힣]")

ANALYZER_COLS = ["formants", "mfcc_dtw", "pitch", "energy", "vot", "coda"]


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score100(v: Any) -> float | None:
    if v is None or pd.isna(v):
        return None

    try:
        return round(float(v) * 100.0, 1)
    except (TypeError, ValueError):
        return None


def _style_score(v: float | None) -> str:
    if v is None:
        return "정보 없음"
    if v >= 70:
        return "좋음"
    if v >= 50:
        return "보통"
    return "개선 필요"


def _hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(text))


def _to_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in results:
        real = item.get("real_file", "")
        m = REAL_RE.match(real)
        person = int(m.group(1)) if m else None
        sentence = int(m.group(2)) if m else None

        analyzers = item.get("analyzers", {})
        feedback = item.get("feedback", {})
        align = item.get("alignment", {})

        row = {
            "real_file": real,
            "ref_file": item.get("ref_file"),
            "text": item.get("text") or feedback.get("text") or "",
            "person": person,
            "sentence": sentence,
            "overall_score": item.get("overall_score"),
            "overall_score_100": _score100(item.get("overall_score")),
            "summary": feedback.get("summary"),
            "diagnosis": feedback.get("diagnosis", []),
            "coaching": feedback.get("coaching", []),
            "targeted_tips": feedback.get("targeted_tips", []),
            "align_backend_used": align.get("used_backend"),
            "align_confidence": align.get("confidence"),
        }

        for name in ANALYZER_COLS:
            row[name] = analyzers.get(name)
            row[f"{name}_100"] = _score100(analyzers.get(name))

        rows.append(row)

    return rows


def _pick_target_from_tip(tip: str) -> str | None:
    """피드백 문장에서 실제 오류 대상 글자를 찾는다.

    예:
      "'어제부터 머리가 너무 아파요.'에서 '요' 발음이 기준과 다릅니다."
      -> "요"

    원칙:
      - 첫 번째 따옴표는 보통 원문 문장이다.
      - 두 번째 이후의 짧은 따옴표가 실제 교정 대상이다.
    """
    matches = list(QUOTED_RE.finditer(tip))

    if len(matches) < 2:
        return None

    for match in matches[1:]:
        inner = match.group(2)
        chars = HANGUL_RE.findall(inner)

        if 1 <= len(chars) <= 3:
            return chars[0]

    return None


def _highlight_target_in_sentence(sentence: str, target: str | None) -> str:
    """원문 문장 안에서 target과 같은 글자를 모두 붉게 표시한다."""
    if not target:
        return html.escape(sentence)

    out: List[str] = []

    for ch in sentence:
        if ch == target:
            out.append(
                f"<span class='bad-syllable-inline'>{html.escape(ch)}</span>"
            )
        else:
            out.append(html.escape(ch))

    return "".join(out)


def _highlight_tip_sentence(tip: str) -> str:
    """targeted_tips 카드 안에서 원문 문장 부분에만 붉은 표시를 넣는다.

    예:
      "'어제부터 머리가 너무 아파요.'에서 '요' 발음이 기준과 다릅니다."

    출력:
      앞쪽 원문 문장 안의 '요'만 붉게 표시
      뒤쪽 설명용 "'요'"는 그대로 표시
    """
    matches = list(QUOTED_RE.finditer(tip))

    if not matches:
        return html.escape(tip)

    target = _pick_target_from_tip(tip)

    out: List[str] = []
    last_end = 0
    highlighted_sentence = False

    for match in matches:
        quote_open = match.group(1)
        inner = match.group(2)
        quote_close = match.group(3)

        out.append(html.escape(tip[last_end:match.start()]))

        if not highlighted_sentence and _hangul_count(inner) >= 4:
            out.append(html.escape(quote_open))
            out.append(_highlight_target_in_sentence(inner, target))
            out.append(html.escape(quote_close))
            highlighted_sentence = True
        else:
            out.append(html.escape(match.group(0)))

        last_end = match.end()

    out.append(html.escape(tip[last_end:]))
    return "".join(out)


def _tip_to_russian_feedback(tip: str) -> str:
    """각 한국어 피드백 문장을 러시아어 피드백 문장으로 변환한다.

    외부 번역 API 없이 규칙 기반으로 생성한다.
    """
    target = _pick_target_from_tip(tip)

    if target:
        target_part = target
    else:
        target_part = "эту часть"

    if "피치" in tip or "높낮이" in tip or "억양" in tip:
        return (
            f"Обратите внимание на интонацию в слоге «{target_part}». "
            "Произнесите эту часть медленно и сравните с образцом."
        )

    if "강세" in tip or "세기" in tip or "에너지" in tip:
        return (
            f"Обратите внимание на силу и ударение в слоге «{target_part}». "
            "Произнесите этот слог более чётко."
        )

    if "받침" in tip or "끝소리" in tip or "종성" in tip or "마무리" in tip or "끝" in tip:
        return (
            f"Обратите внимание на конечный звук в слоге «{target_part}». "
            "Не проглатывайте конец слова, произнесите его чётко."
        )

    if "발음" in tip or "기준과 다릅니다" in tip:
        return (
            f"Обратите внимание на произношение слога «{target_part}». "
            "Сначала повторите его медленно, затем произнесите всё предложение целиком."
        )

    return (
        f"Обратите внимание на слог «{target_part}». "
        "Повторите эту часть медленно несколько раз."
    )


def _speech_synthesis_component(
    text: str,
    button_label: str,
    key_suffix: str,
) -> None:
    js_text = json.dumps(text, ensure_ascii=False)
    safe_label = html.escape(button_label)
    safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key_suffix)
    func_name = f"speakRu_{safe_key}"

    components.html(
        f"""
        <div style="font-family: sans-serif; line-height: 1.5; margin: 0.25rem 0 0.9rem 2.0rem;">
          <button onclick="{func_name}()" style="padding: 0.34rem 0.7rem; border-radius: 0.5rem; border: 1px solid #ccc; cursor: pointer; background: white;">
            {safe_label}
          </button>
          <button onclick="window.speechSynthesis.cancel()" style="padding: 0.34rem 0.7rem; border-radius: 0.5rem; border: 1px solid #ccc; cursor: pointer; margin-left: 0.4rem; background: white;">
            정지
          </button>
        </div>

        <script>
        function {func_name}() {{
            const text = {js_text};
            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "ru-RU";
            utterance.rate = 0.92;
            utterance.pitch = 1.0;

            const voices = window.speechSynthesis.getVoices();
            const ruVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith("ru"));

            if (ruVoice) {{
                utterance.voice = ruVoice;
            }}

            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=60,
    )


st.set_page_config(page_title="발음 피드백 대시보드", layout="wide")

st.title("발음 피드백 대시보드")
st.caption("`batch_compare_results.json` 기반 사용자 피드백 시각화")

st.markdown(
    """
    <style>
    .tip-card {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        background: #fffbea;
        border: 1px solid #fef3c7;
        margin: 0.75rem 0 0.35rem 0;
        font-size: 1.04rem;
        line-height: 1.9;
    }

    .tip-index {
        display: inline-block;
        min-width: 1.6rem;
        color: #6b7280;
    }

    .bad-syllable-inline {
        color: #b91c1c;
        background: #fee2e2;
        border-bottom: 2px solid #ef4444;
        border-radius: 0.25rem;
        padding: 0.03rem 0.12rem;
        font-weight: 800;
    }

    .ru-feedback {
        margin: 0.2rem 0 0.2rem 2.0rem;
        padding: 0.65rem 0.85rem;
        border-radius: 0.65rem;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #374151;
        font-size: 0.96rem;
        line-height: 1.65;
    }

    .ru-label {
        color: #6b7280;
        font-size: 0.86rem;
        margin-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

default_json = Path(__file__).resolve().parent / "batch_compare_results.json"
json_path = st.sidebar.text_input("결과 JSON 경로", value=str(default_json))

payload_path = Path(json_path)

if not payload_path.exists():
    st.error(f"파일이 없습니다: {payload_path}")
    st.stop()

try:
    payload = _load_payload(payload_path)
except Exception as e:
    st.error(f"JSON 로드 실패: {e}")
    st.stop()

rows = _to_rows(payload.get("results", []))

if not rows:
    st.warning("표시할 결과가 없습니다.")
    st.stop()

df = pd.DataFrame(rows)

with st.sidebar:
    st.subheader("필터")

    people = sorted([p for p in df["person"].dropna().unique().tolist()])
    selected_people = st.multiselect("사람 번호", options=people, default=people)

    sentences = sorted([s for s in df["sentence"].dropna().unique().tolist()])
    selected_sentences = st.multiselect("문장 번호", options=sentences, default=sentences)

    max_score = st.slider(
        "최대 Overall 점수",
        min_value=0,
        max_value=100,
        value=100,
        step=1,
    )

fdf = df.copy()

if selected_people:
    fdf = fdf[fdf["person"].isin(selected_people)]

if selected_sentences:
    fdf = fdf[fdf["sentence"].isin(selected_sentences)]

fdf = fdf[fdf["overall_score_100"] <= max_score]

if fdf.empty:
    st.warning("필터 결과가 없습니다.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

col1.metric("샘플 수", len(fdf))
col2.metric("평균 Overall", f"{fdf['overall_score_100'].mean():.1f}점")
col3.metric("최저 Overall", f"{fdf['overall_score_100'].min():.1f}점")
col4.metric("정렬 고신뢰 비율", f"{(fdf['align_confidence'] == 'high').mean() * 100:.1f}%")

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("샘플별 Overall 점수")

    chart_df = fdf.sort_values(["person", "sentence"])

    fig = px.bar(
        chart_df,
        x="real_file",
        y="overall_score_100",
        color="person",
        hover_data=["ref_file", "sentence", "align_backend_used", "align_confidence"],
        labels={
            "overall_score_100": "Overall 점수",
            "real_file": "Real 파일",
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=420, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("분석기 평균 점수")

    analyzer_score_cols = [
        f"{name}_100"
        for name in ANALYZER_COLS
        if f"{name}_100" in fdf.columns
    ]

    mean_scores = fdf[analyzer_score_cols].mean().reset_index()
    mean_scores.columns = ["analyzer", "score"]
    mean_scores["analyzer"] = mean_scores["analyzer"].str.replace("_100", "", regex=False)

    fig2 = px.bar(
        mean_scores,
        x="analyzer",
        y="score",
        text=mean_scores["score"].round(1),
        labels={
            "analyzer": "분석기",
            "score": "평균 점수",
        },
    )

    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=420, yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("샘플 상세 피드백")

options = fdf.sort_values(["person", "sentence"])["real_file"].tolist()
selected_real = st.selectbox("샘플 선택", options=options, index=0)

row = fdf[fdf["real_file"] == selected_real].iloc[0]

tips = row["targeted_tips"] if isinstance(row["targeted_tips"], list) else []

meta1, meta2, meta3, meta4 = st.columns(4)

meta1.metric(
    "Overall",
    f"{row['overall_score_100']:.1f}점",
    _style_score(row["overall_score_100"]),
)
meta2.metric("사람", int(row["person"]) if pd.notna(row["person"]) else -1)
meta3.metric("문장", int(row["sentence"]) if pd.notna(row["sentence"]) else -1)
meta4.metric("정렬", f"{row['align_backend_used']} / {row['align_confidence']}")

st.write(f"**대응 Ref 파일:** `{row['ref_file']}`")
st.info(row["summary"] or "요약 없음")

diag = row["diagnosis"] if isinstance(row["diagnosis"], list) else []
coach = row["coaching"] if isinstance(row["coaching"], list) else []

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 왜 이렇게 나왔나요?")

    if diag:
        for d in diag:
            st.write(f"- {d}")
    else:
        st.write("- 진단 정보 없음")

with col_b:
    st.markdown("### 이렇게 연습해보세요")

    if coach:
        for c in coach:
            st.write(f"- {c}")
    else:
        st.write("- 코칭 정보 없음")

st.markdown("### 어디를 고치면 좋나요? (정확한 위치)")
st.caption("피드백 문장에서 지적한 글자를 원문 문장 안에 붉은색으로 표시합니다.")

if tips:
    for i, tip in enumerate(tips, start=1):
        tip_text = str(tip)
        ru_feedback = _tip_to_russian_feedback(tip_text)

        st.markdown(
            (
                f"<div class='tip-card'>"
                f"<span class='tip-index'>{i}.</span> "
                f"{_highlight_tip_sentence(tip_text)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            (
                f"<div class='ru-feedback'>"
                f"<span class='ru-label'>러시아어 피드백:</span>"
                f"{html.escape(ru_feedback)}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

        _speech_synthesis_component(
            ru_feedback,
            button_label=f"{i}번 피드백 러시아어로 듣기",
            key_suffix=f"{selected_real}_{i}",
        )
else:
    st.write("- targeted_tips 정보 없음")

st.markdown("### 분석기 점수")

score_table = pd.DataFrame(
    [
        {
            "분석기": name,
            "점수": row.get(f"{name}_100"),
        }
        for name in ANALYZER_COLS
        if f"{name}_100" in row.index
    ]
)

st.dataframe(score_table, use_container_width=True, hide_index=True)

st.markdown("---")

st.subheader("전체 샘플 Targeted Tips 한눈에 보기")

tips_rows: List[Dict[str, Any]] = []

for _, r in fdf.sort_values(["person", "sentence"]).iterrows():
    rtips = r["targeted_tips"] if isinstance(r["targeted_tips"], list) else []

    if not rtips:
        tips_rows.append(
            {
                "real_file": r["real_file"],
                "person": r["person"],
                "sentence": r["sentence"],
                "tip": "(없음)",
            }
        )
    else:
        for tip in rtips:
            tips_rows.append(
                {
                    "real_file": r["real_file"],
                    "person": r["person"],
                    "sentence": r["sentence"],
                    "tip": tip,
                }
            )

tips_df = pd.DataFrame(tips_rows)
st.dataframe(tips_df, use_container_width=True, hide_index=True)