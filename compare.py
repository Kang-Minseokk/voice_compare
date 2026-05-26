"""고려인 발음 교정용 음성 비교 엔진.

입력: gt 오디오, sample 오디오, 발화 텍스트
출력: 분석기별 점수 + 음절별 세부 진단

분석기 추가/제거는 ENABLED_ANALYZERS 리스트에서 한다.
각 분석기는 analyzers/base.py의 인터페이스를 따른다.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from alignment import Alignment, align, alignment_from_textgrid
from analyzers import energy, formants, mfcc_dtw, pitch, vot
from analyzers.base import AnalyzerResult
from audio_utils import load_audio


@dataclass
class CompareResult:
    overall_score: float
    analyzers: Dict[str, AnalyzerResult]
    gt_alignment: Alignment
    sample_alignment: Alignment
    alignment_backend_requested: str
    alignment_backend_used: str
    alignment_confidence: str
    warnings: List[str]


ENABLED_ANALYZERS = [formants, mfcc_dtw, pitch, energy, vot]


def _mfa_textgrid_for_audio(audio_path: str, mfa_textgrid_dir: str) -> str:
    stem = Path(audio_path).stem
    tg = Path(mfa_textgrid_dir) / f"{stem}.TextGrid"
    return str(tg)


def compare(
    gt_path: str,
    sample_path: str,
    text: str,
    align_backend: str = "wav2vec2",
    mfa_textgrid_dir: str = "mfa_work/output",
    fail_on_mfa_error: bool = False,
) -> CompareResult:
    gt_audio = load_audio(gt_path)
    sample_audio = load_audio(sample_path)
    warnings: List[str] = []
    backend_requested = align_backend
    backend_used = align_backend

    if align_backend == "mfa":
        try:
            gt_tg = _mfa_textgrid_for_audio(gt_path, mfa_textgrid_dir)
            sample_tg = _mfa_textgrid_for_audio(sample_path, mfa_textgrid_dir)
            if not Path(gt_tg).exists() or not Path(sample_tg).exists():
                raise FileNotFoundError(
                    f"MFA TextGrid not found (gt={gt_tg}, sample={sample_tg})"
                )
            gt_alignment = alignment_from_textgrid(gt_tg, text=text)
            sample_alignment = alignment_from_textgrid(sample_tg, text=text)
        except Exception as e:
            if fail_on_mfa_error:
                raise RuntimeError(
                    "MFA 정렬 실패로 중단했습니다. "
                    f"(원인: {type(e).__name__}: {e})"
                ) from e
            backend_used = "wav2vec2"
            warnings.append(
                "MFA 정렬을 사용할 수 없어 wav2vec2 정렬로 자동 전환했습니다: "
                f"{type(e).__name__}: {e}"
            )
            gt_alignment = align(gt_audio, text)
            sample_alignment = align(sample_audio, text)
    else:
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
    target_syllables = len([ch for ch in text if ch != " "])
    aligned_syllables = min(len(gt_alignment.spans), len(sample_alignment.spans))
    if warnings:
        alignment_confidence = "low"
    elif target_syllables == 0:
        alignment_confidence = "low"
    elif aligned_syllables < target_syllables:
        warnings.append(
            f"정렬된 음절 수({aligned_syllables})가 입력 텍스트 음절 수({target_syllables})보다 적습니다."
        )
        alignment_confidence = "medium"
    else:
        alignment_confidence = "high"

    return CompareResult(
        overall_score=overall,
        analyzers=results,
        gt_alignment=gt_alignment,
        sample_alignment=sample_alignment,
        alignment_backend_requested=backend_requested,
        alignment_backend_used=backend_used,
        alignment_confidence=alignment_confidence,
        warnings=warnings,
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
        "말의 높낮이와 소리의 세기가 기준과 다릅니다. 먼저 말의 흐름을 고치면 좋습니다."
        if prosody_issue
        else "전체 발음은 나쁘지 않습니다. 낮은 점수 부분만 천천히 고치면 됩니다."
    )

    diagnosis: List[str] = []
    if prosody_issue:
        diagnosis.append(
            "문장 전체의 높낮이와 소리 크기 흐름이 기준 음성과 다릅니다."
        )
    if segmental_issue:
        diagnosis.append(
            "몇몇 글자 소리(특히 모음)가 기준과 다르게 들립니다."
        )

    pitch_details = analyzers.get("pitch").details if "pitch" in analyzers else {}
    gt_tail = pitch_details.get("gt_tail_slope_st_per_sec")
    sample_tail = pitch_details.get("sample_tail_slope_st_per_sec")
    if gt_tail is not None and sample_tail is not None:
        diagnosis.append(
            "문장 끝부분의 말투(올라감/내려감)가 기준과 다릅니다."
        )

    vot_details = analyzers.get("vot").details if "vot" in analyzers else {}
    stop_count = int(vot_details.get("stop_count", 0))
    valid_count = int(vot_details.get("valid_count", 0))
    if stop_count > 0 and valid_count < max(2, stop_count // 2):
        diagnosis.append(
            "자음 비교는 이번 녹음에서 측정 가능한 구간이 적어서 정확도가 낮습니다."
        )

    coaching: List[str] = []
    if prosody_issue:
        coaching.append(
            "문장 뒤쪽에서 목소리가 너무 내려가지 않게, 높낮이를 조금 더 살려서 말해보세요."
        )
        coaching.append(
            "힘이 약한 부분은 소리를 조금 더 또렷하고 힘 있게 내보세요."
        )
    if segmental_issue:
        coaching.append(
            "점수가 낮은 글자만 골라서, 입모양을 크게 하고 천천히 반복해 보세요."
        )
    if stop_count > 0 and valid_count < max(2, stop_count // 2):
        coaching.append(
            "자음은 긴 문장보다 짧은 단어(예: 가/까/카)를 반복하면 더 잘 고칠 수 있습니다."
        )
    if not coaching:
        coaching.append("현재 패턴을 유지하면서 낮은 점수 음절만 선택적으로 교정해 보세요.")

    return {
        "summary": summary,
        "alignment": {
            "requested_backend": result.alignment_backend_requested,
            "used_backend": result.alignment_backend_used,
            "confidence": result.alignment_confidence,
            "warnings": result.warnings,
        },
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
    parser = argparse.ArgumentParser(description="Pronunciation compare CLI")
    parser.add_argument("--gt-audio", default="hospital_0_ref.wav")
    parser.add_argument("--sample-audio", default="hospital_0_real.wav")
    parser.add_argument("--text", default="자꾸 배가 아프고 속이 쓰려요")
    parser.add_argument(
        "--align-backend",
        default="wav2vec2",
        choices=["wav2vec2", "mfa"],
        help="alignment backend 선택 (기본: wav2vec2)",
    )
    parser.add_argument(
        "--mfa-textgrid-dir",
        default="mfa_work/output",
        help="MFA TextGrid 폴더 경로 (align-backend=mfa일 때 사용)",
    )
    parser.add_argument(
        "--fail-on-mfa-error",
        action="store_true",
        help="MFA 정렬 실패 시 fallback 하지 않고 바로 종료",
    )
    args = parser.parse_args()

    result = compare(
        args.gt_audio,
        args.sample_audio,
        args.text,
        align_backend=args.align_backend,
        mfa_textgrid_dir=args.mfa_textgrid_dir,
        fail_on_mfa_error=args.fail_on_mfa_error,
    )
    print(f"\n=== Overall: {result.overall_score:.3f} ===\n")
    print(
        f"[alignment] requested={result.alignment_backend_requested}, "
        f"used={result.alignment_backend_used}, "
        f"confidence={result.alignment_confidence}"
    )
    for w in result.warnings:
        print(f"  warning: {w}")
    print()

    for name, ar in result.analyzers.items():
        print(f"[{name}] score = {ar.score:.3f}")
        print(f"  metric: {ar.details.get('metric', '')}")
        _PRINTERS.get(name, lambda a: None)(ar)
        print()

    feedback = _build_user_feedback(result)
    print("=== User Feedback (JSON) ===")
    print(json.dumps(feedback, ensure_ascii=False, indent=2))
