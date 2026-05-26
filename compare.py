"""고려인 발음 교정용 음성 비교 엔진.

입력: gt 오디오, sample 오디오, 발화 텍스트
출력: 분석기별 점수 + 음절별 세부 진단

분석기 추가/제거는 ENABLED_ANALYZERS 리스트에서 한다.
각 분석기는 analyzers/base.py의 인터페이스를 따른다.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from alignment import Alignment, align
from analyzers import energy, formants, mfcc_dtw, pitch, vot
from analyzers.base import AnalyzerResult
from audio_utils import load_audio


@dataclass
class CompareResult:
    overall_score: float
    analyzers: Dict[str, AnalyzerResult]
    gt_alignment: Alignment
    sample_alignment: Alignment


ENABLED_ANALYZERS = [formants, mfcc_dtw, pitch, energy, vot]


def compare(gt_path: str, sample_path: str, text: str) -> CompareResult:
    gt_audio = load_audio(gt_path)
    sample_audio = load_audio(sample_path)

    gt_alignment = align(gt_audio, text)
    sample_alignment = align(sample_audio, text)

    results = {}
    for module in ENABLED_ANALYZERS:
        r = module.analyze(
            gt_audio, sample_audio, gt_alignment, sample_alignment, text
        )
        results[r.name] = r

    scores = [r.score for r in results.values()]
    overall = sum(scores) / len(scores) if scores else 0.0

    return CompareResult(
        overall_score=overall,
        analyzers=results,
        gt_alignment=gt_alignment,
        sample_alignment=sample_alignment,
    )


def _build_user_feedback(result: CompareResult) -> Dict[str, Any]:
    """분석 결과를 사용자 노출용 rule-based 피드백으로 변환."""
    analyzers = result.analyzers
    formants_score = analyzers.get("formants").score if "formants" in analyzers else None
    mfcc_score = analyzers.get("mfcc_dtw").score if "mfcc_dtw" in analyzers else None
    pitch_score = analyzers.get("pitch").score if "pitch" in analyzers else None
    energy_score = analyzers.get("energy").score if "energy" in analyzers else None
    vot_score = analyzers.get("vot").score if "vot" in analyzers else None

    prosody_issue = (
        pitch_score is not None
        and energy_score is not None
        and ((pitch_score < 0.45 and energy_score < 0.45) or pitch_score < 0.35)
    )
    segmental_issue = (
        formants_score is not None
        and mfcc_score is not None
        and (formants_score < 0.45 or mfcc_score < 0.42)
    )

    summary = (
        "말의 높낮이와 힘 조절에 오류가 있습니다. 이번에는 발음 자체보다 말투(리듬)를 먼저 고치는 것이 좋습니다."
        if prosody_issue
        else "발음과 말투를 함께 조금씩 다듬으면 전체 점수를 더 안정적으로 올릴 수 있습니다."
    )

    diagnosis: List[str] = []
    if prosody_issue:
        diagnosis.append(
            "문장 전체의 높낮이와 힘의 흐름이 기준 음성과 다르게 나왔습니다."
        )
    if segmental_issue:
        diagnosis.append(
            "일부 글자 소리(특히 모음)가 기준 음성과 다르게 들립니다."
        )

    pitch_details = analyzers.get("pitch").details if "pitch" in analyzers else {}
    gt_tail = pitch_details.get("gt_tail_slope_st_per_sec")
    sample_tail = pitch_details.get("sample_tail_slope_st_per_sec")
    if gt_tail is not None and sample_tail is not None:
        diagnosis.append(
            "문장 끝부분에서 목소리가 내려가거나 올라가는 방식이 기준 음성과 차이가 있습니다."
        )

    vot_details = analyzers.get("vot").details if "vot" in analyzers else {}
    stop_count = int(vot_details.get("stop_count", 0))
    valid_count = int(vot_details.get("valid_count", 0))
    if stop_count > 0 and valid_count < max(2, stop_count // 2):
        diagnosis.append(
            "자음 비교는 이번 녹음에서 측정 가능한 구간이 적어서 참고용으로만 보는 것이 좋습니다."
        )

    coaching: List[str] = []
    if prosody_issue:
        coaching.append(
            "문장 뒤로 갈수록 목소리가 너무 떨어지지 않게, 높낮이를 조금 더 살려서 말해보세요."
        )
        coaching.append(
            "힘이 빠진 부분은 소리를 조금 더 또렷하고 힘 있게 내보세요."
        )
    if segmental_issue:
        coaching.append(
            "점수가 낮은 글자 소리를 골라, 입모양을 분명히 하면서 천천히 반복 연습해보세요."
        )
    if stop_count > 0 and valid_count < max(2, stop_count // 2):
        coaching.append(
            "자음은 긴 문장보다 짧은 단어(예: 가/까/카)를 반복해 연습하면 더 정확하게 교정할 수 있습니다."
        )
    if not coaching:
        coaching.append("현재 패턴을 유지하면서 낮은 점수 음절만 선택적으로 교정해 보세요.")

    return {
        "summary": summary,
        "diagnosis": diagnosis,
        "coaching": coaching,
        "scores": {
            "overall": round(result.overall_score, 3),
            "formants": round(formants_score, 3) if formants_score is not None else None,
            "mfcc_dtw": round(mfcc_score, 3) if mfcc_score is not None else None,
            "pitch": round(pitch_score, 3) if pitch_score is not None else None,
            "energy": round(energy_score, 3) if energy_score is not None else None,
            "vot": round(vot_score, 3) if vot_score is not None else None,
        },
        "confidence": {
            "vot_valid_count": valid_count,
            "vot_stop_count": stop_count,
            "vot_confidence": (
                "low"
                if stop_count > 0 and valid_count < max(2, stop_count // 2)
                else "medium"
            ),
        },
    }


def _fmt_hz(v):
    return f"{v:.0f}" if v is not None else "  --"


def _print_formants(ar):
    print(f"  {'char':>4} {'vowel':>5} {'score':>6}  "
          f"{'gt(F1, F2)':>16}  {'sample(F1, F2)':>16}")
    for ps in ar.per_syllable:
        d = ps.details
        gt_str = f"({_fmt_hz(d.get('gt_f1'))}, {_fmt_hz(d.get('gt_f2'))})"
        sm_str = f"({_fmt_hz(d.get('sample_f1'))}, {_fmt_hz(d.get('sample_f2'))})"
        vowel = d.get("vowel") or "?"
        print(f"  {ps.char:>4} {vowel:>5} {ps.score:>6.3f}  "
              f"{gt_str:>16}  {sm_str:>16}")


def _print_mfcc_dtw(ar):
    print(f"  {'char':>4} {'score':>6}  {'dtw_dist':>9}  {'frames':>11}")
    for ps in ar.per_syllable:
        d = ps.details
        dist = d.get("dtw_distance")
        dist_str = f"{dist:.2f}" if dist is not None else "  --"
        frames = f"{d.get('gt_frames', 0)}/{d.get('sample_frames', 0)}"
        print(f"  {ps.char:>4} {ps.score:>6.3f}  {dist_str:>9}  {frames:>11}")


def _fmt_opt(v, spec=".1f"):
    if v is None:
        return "  --"
    return format(v, spec)


def _print_pitch(ar):
    d = ar.details
    print(f"  primary (alignment-locked): {d.get('alignment_locked_score', 0):.3f}")
    print(f"  secondary (contour DTW):    {d.get('contour_score', 0):.3f}"
          f"   [dist={_fmt_opt(d.get('contour_dtw_distance_st'), '.2f')} st]")
    print(f"  gt median F0: {_fmt_opt(d.get('gt_median_hz'), '.0f')}Hz, "
          f"sample median F0: {_fmt_opt(d.get('sample_median_hz'), '.0f')}Hz")
    print(f"  tail slope: gt {_fmt_opt(d.get('gt_tail_slope_st_per_sec'))} st/s, "
          f"sample {_fmt_opt(d.get('sample_tail_slope_st_per_sec'))} st/s")
    print(f"  {'char':>4} {'score':>6}  {'gt(st)':>7}  {'sample(st)':>10}  {'diff':>6}")
    for ps in ar.per_syllable:
        pd = ps.details
        print(f"  {ps.char:>4} {ps.score:>6.3f}  "
              f"{_fmt_opt(pd.get('gt_st')):>7}  "
              f"{_fmt_opt(pd.get('sample_st')):>10}  "
              f"{_fmt_opt(pd.get('diff_st')):>6}")


def _print_energy(ar):
    d = ar.details
    print(f"  primary (alignment-locked): {d.get('alignment_locked_score', 0):.3f}")
    print(f"  secondary (contour DTW):    {d.get('contour_score', 0):.3f}"
          f"   [dist={_fmt_opt(d.get('contour_dtw_distance_db'), '.2f')} dB]")
    print(f"  gt mean: {_fmt_opt(d.get('gt_mean_db'), '.1f')} dB, "
          f"sample mean: {_fmt_opt(d.get('sample_mean_db'), '.1f')} dB")
    print(f"  {'char':>4} {'score':>6}  {'gt(dB)':>7}  {'sample(dB)':>10}  {'diff':>6}")
    for ps in ar.per_syllable:
        pd = ps.details
        print(f"  {ps.char:>4} {ps.score:>6.3f}  "
              f"{_fmt_opt(pd.get('gt_db')):>7}  "
              f"{_fmt_opt(pd.get('sample_db')):>10}  "
              f"{_fmt_opt(pd.get('diff_db')):>6}")


def _print_vot(ar):
    d = ar.details
    note = d.get("note") or ""
    print(f"  stops analyzed: {d.get('valid_count', 0)}/{d.get('stop_count', 0)}"
          f"{('  — ' + note) if note else ''}")
    if not ar.per_syllable:
        return
    print(f"  {'char':>4} {'init':>5} {'score':>6}  "
          f"{'gt_VOT(ms)':>11}  {'sample_VOT(ms)':>15}  {'diff':>6}")
    for ps in ar.per_syllable:
        pd = ps.details
        print(f"  {ps.char:>4} {pd.get('initial', '?'):>5} {ps.score:>6.3f}  "
              f"{_fmt_opt(pd.get('gt_vot_ms')):>11}  "
              f"{_fmt_opt(pd.get('sample_vot_ms')):>15}  "
              f"{_fmt_opt(pd.get('diff_ms')):>6}")


_PRINTERS = {
    "formants": _print_formants,
    "mfcc_dtw": _print_mfcc_dtw,
    "pitch": _print_pitch,
    "energy": _print_energy,
    "vot": _print_vot,
}


if __name__ == "__main__":
    gt_sound = "hospital_0_ref.wav"
    sample_sound = "hospital_0_real.wav"
    text = "자꾸 배가 아프고 속이 쓰려요"

    result = compare(gt_sound, sample_sound, text)
    print(f"\n=== Overall: {result.overall_score:.3f} ===\n")

    for name, ar in result.analyzers.items():
        print(f"[{name}] score = {ar.score:.3f}")
        print(f"  metric: {ar.details.get('metric', '')}")
        _PRINTERS.get(name, lambda a: None)(ar)
        print()

    feedback = _build_user_feedback(result)
    print("=== User Feedback (JSON) ===")
    print(json.dumps(feedback, ensure_ascii=False, indent=2))
