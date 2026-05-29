"""batch_compare_results.json 시각화 대시보드 (Streamlit).

실행:
  streamlit run view_feedback.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st


REAL_RE = re.compile(r"hospital_(\d+)-(\d+)_real\.wav$")


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        rows.append(
            {
                "real_file": real,
                "ref_file": item.get("ref_file"),
                "person": person,
                "sentence": sentence,
                "overall_score": item.get("overall_score"),
                "formants": analyzers.get("formants"),
                "mfcc_dtw": analyzers.get("mfcc_dtw"),
                "pitch": analyzers.get("pitch"),
                "energy": analyzers.get("energy"),
                "vot": analyzers.get("vot"),
                "summary": feedback.get("summary"),
                "diagnosis": feedback.get("diagnosis", []),
                "coaching": feedback.get("coaching", []),
                "targeted_tips": feedback.get("targeted_tips", []),
                "align_backend_used": align.get("used_backend"),
                "align_confidence": align.get("confidence"),
            }
        )
    return rows


def _style_score(v: float | None) -> str:
    if v is None:
        return "정보 없음"
    if v >= 0.7:
        return "좋음"
    if v >= 0.5:
        return "보통"
    return "개선 필요"


st.set_page_config(page_title="발음 피드백 대시보드", layout="wide")
st.title("발음 피드백 대시보드")
st.caption("`batch_compare_results.json` 기반 사용자 피드백 시각화")

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
    max_score = st.slider("최대 overall_score", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

fdf = df.copy()
if selected_people:
    fdf = fdf[fdf["person"].isin(selected_people)]
if selected_sentences:
    fdf = fdf[fdf["sentence"].isin(selected_sentences)]
fdf = fdf[fdf["overall_score"] <= max_score]

if fdf.empty:
    st.warning("필터 결과가 없습니다.")
    st.stop()

# 상단 요약
col1, col2, col3, col4 = st.columns(4)
col1.metric("샘플 수", len(fdf))
col2.metric("평균 Overall", f"{fdf['overall_score'].mean():.3f}")
col3.metric("최저 Overall", f"{fdf['overall_score'].min():.3f}")
col4.metric("정렬 고신뢰 비율", f"{(fdf['align_confidence'] == 'high').mean() * 100:.1f}%")

st.markdown("---")

# 점수 차트
left, right = st.columns(2)
with left:
    st.subheader("샘플별 Overall 점수")
    chart_df = fdf.sort_values(["person", "sentence"])
    fig = px.bar(
        chart_df,
        x="real_file",
        y="overall_score",
        color="person",
        hover_data=["ref_file", "sentence", "align_backend_used", "align_confidence"],
        labels={"overall_score": "Overall", "real_file": "Real 파일"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("분석기 평균 점수")
    analyzer_cols = ["formants", "mfcc_dtw", "pitch", "energy", "vot"]
    mean_scores = fdf[analyzer_cols].mean().reset_index()
    mean_scores.columns = ["analyzer", "score"]
    fig2 = px.bar(
        mean_scores,
        x="analyzer",
        y="score",
        text=mean_scores["score"].round(3),
        labels={"analyzer": "분석기", "score": "평균 점수"},
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=420, yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# 상세 카드
st.subheader("샘플 상세 피드백")
options = fdf.sort_values(["person", "sentence"])["real_file"].tolist()
selected_real = st.selectbox("샘플 선택", options=options, index=0)
row = fdf[fdf["real_file"] == selected_real].iloc[0]

meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Overall", f"{row['overall_score']:.3f}", _style_score(row["overall_score"]))
meta2.metric("사람", int(row["person"]) if pd.notna(row["person"]) else -1)
meta3.metric("문장", int(row["sentence"]) if pd.notna(row["sentence"]) else -1)
meta4.metric("정렬", f"{row['align_backend_used']} / {row['align_confidence']}")

st.write(f"**대응 Ref 파일:** `{row['ref_file']}`")
st.info(row["summary"] or "요약 없음")

diag = row["diagnosis"] if isinstance(row["diagnosis"], list) else []
coach = row["coaching"] if isinstance(row["coaching"], list) else []
tips = row["targeted_tips"] if isinstance(row["targeted_tips"], list) else []

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
if tips:
    for i, tip in enumerate(tips, start=1):
        st.warning(f"{i}. {tip}")
else:
    st.write("- targeted_tips 정보 없음")

st.markdown("### 분석기 점수")
score_table = pd.DataFrame(
    [
        {"분석기": "formants", "점수": row["formants"]},
        {"분석기": "mfcc_dtw", "점수": row["mfcc_dtw"]},
        {"분석기": "pitch", "점수": row["pitch"]},
        {"분석기": "energy", "점수": row["energy"]},
        {"분석기": "vot", "점수": row["vot"]},
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
