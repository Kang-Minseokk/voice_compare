"""고려인 발음 교정용 음성 비교 엔진.

입력: gt 오디오, sample 오디오, 발화 텍스트
출력: 분석기별 점수 + 음절별 세부 진단

분석기 추가/제거는 ENABLED_ANALYZERS 리스트에서 한다.
각 분석기는 analyzers/base.py의 인터페이스를 따른다.
"""

from dataclasses import dataclass
from typing import Dict

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
