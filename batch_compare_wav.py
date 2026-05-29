"""WAV 전용 배치 비교 스크립트.

규칙:
  hospital_<person>-<ref_idx>_real.wav -> hospital_<ref_idx>_ref.wav

기본적으로 m4a/mp3는 무시하고 wav만 처리한다.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import compare as compare_module


REAL_RE = re.compile(r"hospital_(\d+)-(\d+)(?:-([A-Za-z0-9_]+))?_real\.wav$")


def _load_text_map(text: Optional[str], text_map_json: Optional[str]) -> Dict[str, str]:
    if text_map_json:
        p = Path(text_map_json)
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    if text:
        return {"*": text}
    raise ValueError("`--text` 또는 `--text-map-json` 중 하나는 반드시 필요합니다.")


def _resolve_text(text_map: Dict[str, str], ref_idx: str) -> Optional[str]:
    return text_map.get(ref_idx) or text_map.get("*")


def _run_one(
    real_path: str,
    ref_path: str,
    text: str,
    align_backend: str,
    mfa_textgrid_dir: str,
    fail_on_mfa_error: bool,
    trim_silence: bool,
    trim_db: float,
    target_rms_dbfs: float,
    overall_trim_ratio: float,
) -> Dict[str, Any]:
    result = compare_module.compare(
        gt_path=ref_path,
        sample_path=real_path,
        text=text,
        align_backend=align_backend,
        mfa_textgrid_dir=mfa_textgrid_dir,
        fail_on_mfa_error=fail_on_mfa_error,
        trim_silence=trim_silence,
        trim_db=trim_db,
        target_rms_dbfs=target_rms_dbfs,
        overall_trim_ratio=overall_trim_ratio,
    )
    feedback = compare_module._build_user_feedback(result)
    # 배치 JSON에서는 feedback 내부 warnings를 노출하지 않는다.
    if isinstance(feedback.get("alignment"), dict):
        feedback["alignment"].pop("warnings", None)
    analyzer_scores = {name: round(ar.score, 3) for name, ar in result.analyzers.items()}
    return {
        "real_file": Path(real_path).name,
        "ref_file": Path(ref_path).name,
        "overall_score": round(result.overall_score, 3),
        "analyzers": analyzer_scores,
        "alignment": {
            "requested_backend": result.alignment_backend_requested,
            "used_backend": result.alignment_backend_used,
            "confidence": result.alignment_confidence,
            "warnings": result.warnings,
        },
        "feedback": feedback,
    }


def _discover_jobs(
    real_dir: Path, ref_dir: Path, text_map: Dict[str, str]
) -> Tuple[List[Dict[str, str]], List[str]]:
    jobs: List[Dict[str, str]] = []
    errors: List[str] = []

    real_files = sorted(p for p in real_dir.iterdir() if p.is_file() and REAL_RE.match(p.name))
    for real in real_files:
        m = REAL_RE.match(real.name)
        assert m is not None
        person_idx, ref_idx = m.group(1), m.group(2)
        ref = ref_dir / f"hospital_{ref_idx}_ref.wav"
        if not ref.exists():
            errors.append(f"{real.name}: 매핑된 ref 파일 없음 ({ref.name})")
            continue
        text = _resolve_text(text_map, ref_idx)
        if not text:
            errors.append(f"{real.name}: ref_idx={ref_idx}에 해당하는 text 없음")
            continue
        jobs.append(
            {
                "person_idx": person_idx,
                "ref_idx": ref_idx,
                "take_id": m.group(3) or "base",
                "real_path": str(real),
                "ref_path": str(ref),
                "text": text,
            }
        )
    return jobs, errors


def _mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def _aggregate_repeat_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """반복 녹음(화자+ref) 그룹별 평균/분산 집계."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in results:
        m = REAL_RE.match(row.get("real_file", ""))
        if m is None:
            continue
        person_idx, ref_idx = m.group(1), m.group(2)
        grouped.setdefault((person_idx, ref_idx), []).append(row)

    groups_out: List[Dict[str, Any]] = []
    for (person_idx, ref_idx), rows in sorted(grouped.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        overall_vals = [float(r.get("overall_score", 0.0)) for r in rows]
        analyzer_names = sorted({k for r in rows for k in r.get("analyzers", {}).keys()})
        analyzer_stats = {}
        for name in analyzer_names:
            vals = [float(r["analyzers"][name]) for r in rows if name in r.get("analyzers", {})]
            analyzer_stats[name] = {
                "mean": round(_mean(vals), 4),
                "std": round(_std(vals), 4),
                "count": len(vals),
            }

        groups_out.append(
            {
                "person_idx": person_idx,
                "ref_idx": ref_idx,
                "repeat_count": len(rows),
                "real_files": [r.get("real_file") for r in rows],
                "overall": {
                    "mean": round(_mean(overall_vals), 4),
                    "std": round(_std(overall_vals), 4),
                    "min": round(min(overall_vals), 4) if overall_vals else 0.0,
                    "max": round(max(overall_vals), 4) if overall_vals else 0.0,
                },
                "analyzers": analyzer_stats,
            }
        )

    repeat_groups = [g for g in groups_out if g["repeat_count"] > 1]
    mean_repeat_count = _mean([float(g["repeat_count"]) for g in repeat_groups]) if repeat_groups else 0.0
    repeat_overall_std_mean = _mean([float(g["overall"]["std"]) for g in repeat_groups]) if repeat_groups else 0.0

    return {
        "group_count": len(groups_out),
        "repeat_group_count": len(repeat_groups),
        "single_group_count": len(groups_out) - len(repeat_groups),
        "summary": {
            "mean_repeat_count": round(mean_repeat_count, 4),
            "repeat_overall_std_mean": round(repeat_overall_std_mean, 4),
        },
        "groups": groups_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="병렬 WAV 배치 비교")
    parser.add_argument("--root", default=".", help="오디오 파일 폴더")
    parser.add_argument(
        "--real-dir",
        default=None,
        help="real wav 폴더 경로 (기본: <root>/hospital_real_audio, 없으면 <root>)",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="ref wav 폴더 경로 (기본: <root>/hospital_ref_audio, 없으면 <root>)",
    )
    parser.add_argument("--text", default=None, help="모든 비교에 공통으로 쓸 텍스트")
    parser.add_argument(
        "--text-map-json",
        default=None,
        help='ref 인덱스별 텍스트 맵 JSON 파일 경로 (예: {"1":"...", "2":"..."})',
    )
    parser.add_argument("--align-backend", default="wav2vec2", choices=["wav2vec2", "mfa"])
    parser.add_argument("--mfa-textgrid-dir", default="mfa_work/output")
    parser.add_argument("--fail-on-mfa-error", action="store_true")
    parser.add_argument(
        "--no-trim-silence",
        action="store_true",
        help="앞/뒤 무음 trim 전처리를 비활성화",
    )
    parser.add_argument(
        "--trim-db",
        type=float,
        default=35.0,
        help="무음 trim threshold (peak 대비 dB, 기본: 35)",
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=-24.0,
        help="RMS 정규화 목표 dBFS (기본: -24)",
    )
    parser.add_argument(
        "--overall-trim-ratio",
        type=float,
        default=0.15,
        help="overall 점수 robust trimmed mean 비율 (기본: 0.15)",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-json", default="batch_compare_results.json")
    parser.add_argument(
        "--repeat-report-json",
        default=None,
        help="반복 녹음 평균/분산 리포트를 별도 JSON으로 저장할 경로 (옵션)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    default_real_dir = root / "hospital_real_audio"
    default_ref_dir = root / "hospital_ref_audio"
    real_dir = (
        Path(args.real_dir).resolve()
        if args.real_dir
        else (default_real_dir if default_real_dir.exists() else root)
    )
    ref_dir = (
        Path(args.ref_dir).resolve()
        if args.ref_dir
        else (default_ref_dir if default_ref_dir.exists() else root)
    )
    text_map = _load_text_map(args.text, args.text_map_json)
    jobs, precheck_errors = _discover_jobs(real_dir, ref_dir, text_map)

    print(f"[info] root={root}")
    print(f"[info] real_dir={real_dir}")
    print(f"[info] ref_dir={ref_dir}")
    print(f"[info] jobs={len(jobs)}, precheck_errors={len(precheck_errors)}")
    for e in precheck_errors:
        print(f"[warn] {e}")

    results: List[Dict[str, Any]] = []
    runtime_errors: List[str] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            future_map = {
                ex.submit(
                    _run_one,
                    j["real_path"],
                    j["ref_path"],
                    j["text"],
                    args.align_backend,
                    args.mfa_textgrid_dir,
                    args.fail_on_mfa_error,
                    not args.no_trim_silence,
                    args.trim_db,
                    args.target_rms_dbfs,
                    args.overall_trim_ratio,
                ): j
                for j in jobs
            }
            for fut in as_completed(future_map):
                job = future_map[fut]
                tag = f"{Path(job['real_path']).name} -> {Path(job['ref_path']).name}"
                try:
                    row = fut.result()
                    results.append(row)
                    print(
                        f"[ok] {tag} | overall={row['overall_score']:.3f} "
                        f"| used={row['alignment']['used_backend']}"
                    )
                except Exception as e:
                    runtime_errors.append(f"{tag}: {type(e).__name__}: {e}")
                    print(f"[err] {runtime_errors[-1]}")

    results.sort(key=lambda x: x["real_file"])
    repeat_report = _aggregate_repeat_report(results)
    payload = {
        "root": str(root),
        "real_dir": str(real_dir),
        "ref_dir": str(ref_dir),
        "align_backend": args.align_backend,
        "preprocess": {
            "trim_silence": not args.no_trim_silence,
            "trim_db": args.trim_db,
            "target_rms_dbfs": args.target_rms_dbfs,
            "overall_trim_ratio": args.overall_trim_ratio,
        },
        "job_count": len(jobs),
        "success_count": len(results),
        "precheck_errors": precheck_errors,
        "runtime_errors": runtime_errors,
        "repeat_report": repeat_report,
        "results": results,
    }
    out = Path(args.output_json).resolve()
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")
    if args.repeat_report_json:
        repeat_out = Path(args.repeat_report_json).resolve()
        repeat_out.write_text(json.dumps(repeat_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] wrote repeat report {repeat_out}")


if __name__ == "__main__":
    main()
