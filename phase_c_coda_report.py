"""Phase C: coda 분석 분포/임계값 튜닝 보조 리포트 생성.

실행 예:
  python phase_c_coda_report.py --root . --text-map-json ref_texts.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import compare as compare_module


REAL_RE = re.compile(r"hospital_(\d+)-(\d+)_real\.wav$")


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


def _discover_jobs(real_dir: Path, ref_dir: Path, text_map: Dict[str, str]) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for real in sorted(p for p in real_dir.iterdir() if p.is_file() and REAL_RE.match(p.name)):
        m = REAL_RE.match(real.name)
        assert m is not None
        person_idx, ref_idx = m.group(1), m.group(2)
        ref = ref_dir / f"hospital_{ref_idx}_ref.wav"
        text = _resolve_text(text_map, ref_idx)
        if not ref.exists() or not text:
            continue
        jobs.append(
            {
                "person_idx": person_idx,
                "ref_idx": ref_idx,
                "real_path": str(real),
                "ref_path": str(ref),
                "text": text,
            }
        )
    return jobs


def _safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C coda 리포트")
    parser.add_argument("--root", default=".")
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--ref-dir", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-map-json", default=None)
    parser.add_argument("--align-backend", default="wav2vec2", choices=["wav2vec2", "mfa"])
    parser.add_argument("--mfa-textgrid-dir", default="mfa_work/output")
    parser.add_argument("--output-json", default="phase_c_coda_report.json")
    parser.add_argument("--output-md", default="phase_c_coda_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    default_real = root / "hospital_real_audio"
    default_ref = root / "hospital_ref_audio"
    real_dir = Path(args.real_dir).resolve() if args.real_dir else (default_real if default_real.exists() else root)
    ref_dir = Path(args.ref_dir).resolve() if args.ref_dir else (default_ref if default_ref.exists() else root)
    text_map = _load_text_map(args.text, args.text_map_json)
    jobs = _discover_jobs(real_dir, ref_dir, text_map)

    all_rows: List[Dict[str, Any]] = []
    by_coda: Dict[str, List[Dict[str, Any]]] = {}

    for j in jobs:
        result = compare_module.compare(
            gt_path=j["ref_path"],
            sample_path=j["real_path"],
            text=j["text"],
            align_backend=args.align_backend,
            mfa_textgrid_dir=args.mfa_textgrid_dir,
        )
        coda_ar = result.analyzers.get("coda")
        if not coda_ar:
            continue
        for ps in coda_ar.per_syllable:
            d = ps.details or {}
            row = {
                "real_file": Path(j["real_path"]).name,
                "ref_file": Path(j["ref_path"]).name,
                "char": ps.char,
                "coda_char": d.get("coda_char"),
                "status": d.get("status"),
                "score": ps.score,
                "gt_decay_ratio": d.get("gt_decay_ratio"),
                "sample_decay_ratio": d.get("sample_decay_ratio"),
                "gt_duration_ratio": d.get("gt_duration_ratio"),
                "sample_duration_ratio": d.get("sample_duration_ratio"),
                "gt_voicing_residue": d.get("gt_voicing_residue"),
                "sample_voicing_residue": d.get("sample_voicing_residue"),
                "gt_fallback": bool(d.get("gt_fallback", False)),
                "sample_fallback": bool(d.get("sample_fallback", False)),
                "gt_syllable_frames": d.get("gt_syllable_frames"),
                "sample_syllable_frames": d.get("sample_syllable_frames"),
                "gt_coda_frames": d.get("gt_coda_frames"),
                "sample_coda_frames": d.get("sample_coda_frames"),
            }
            all_rows.append(row)
            coda_char = row["coda_char"] or "unknown"
            by_coda.setdefault(coda_char, []).append(row)

    status_counts = {"ok": 0, "weakened": 0, "dropped": 0, "uncertain": 0, "not_applicable": 0}
    for r in all_rows:
        s = r["status"] or "not_applicable"
        status_counts[s] = status_counts.get(s, 0) + 1

    coda_summary = {}
    for coda_char, rows in by_coda.items():
        coda_summary[coda_char] = {
            "count": len(rows),
            "avg_score": round(_safe_mean([x["score"] for x in rows]), 4),
            "weakened_rate": round(sum(1 for x in rows if x["status"] == "weakened") / len(rows), 4),
            "dropped_rate": round(sum(1 for x in rows if x["status"] == "dropped") / len(rows), 4),
            "uncertain_rate": round(sum(1 for x in rows if x["status"] == "uncertain") / len(rows), 4),
            "avg_decay_diff": round(
                _safe_mean(
                    [
                        abs((x["sample_decay_ratio"] or 0.0) - (x["gt_decay_ratio"] or 0.0))
                        for x in rows
                        if x["sample_decay_ratio"] is not None and x["gt_decay_ratio"] is not None
                    ]
                ),
                4,
            ),
        }

    report = {
        "job_count": len(jobs),
        "coda_token_count": len(all_rows),
        "fallback_counts": {
            "both_fallback": sum(1 for r in all_rows if r["gt_fallback"] and r["sample_fallback"]),
            "one_side_fallback": sum(1 for r in all_rows if (r["gt_fallback"] != r["sample_fallback"])),
            "none_fallback": sum(1 for r in all_rows if not r["gt_fallback"] and not r["sample_fallback"]),
        },
        "status_counts": status_counts,
        "status_rates": {
            k: round(v / len(all_rows), 4) if all_rows else 0.0 for k, v in status_counts.items()
        },
        "coda_summary": coda_summary,
    }

    json_out = Path(args.output_json).resolve()
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Phase C Coda Report")
    lines.append("")
    lines.append(f"- Jobs: {len(jobs)}")
    lines.append(f"- Coda tokens: {len(all_rows)}")
    fb = report["fallback_counts"]
    lines.append(
        f"- Fallback tokens: both={fb['both_fallback']} one_side={fb['one_side_fallback']} none={fb['none_fallback']}"
    )
    lines.append("- Status counts:")
    for k, v in status_counts.items():
        rate = report["status_rates"][k]
        lines.append(f"  - {k}: {v} ({rate:.2%})")
    lines.append("")
    lines.append("## By Coda")
    lines.append("")
    lines.append("| coda | count | avg_score | weakened_rate | dropped_rate | uncertain_rate | avg_decay_diff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for coda_char in sorted(coda_summary.keys()):
        s = coda_summary[coda_char]
        lines.append(
            f"| {coda_char} | {s['count']} | {s['avg_score']:.3f} | "
            f"{s['weakened_rate']:.2%} | {s['dropped_rate']:.2%} | "
            f"{s['uncertain_rate']:.2%} | {s['avg_decay_diff']:.3f} |"
        )
    lines.append("")
    lines.append("## Next Tuning Suggestions")
    lines.append("")
    lines.append("- uncertain_rate가 높은 coda는 fallback 규칙 재조정 또는 해당 케이스 제외 기준을 강화")
    lines.append("- dropped_rate가 과도하게 높은 coda는 DROP_RATIO 완화(예: 0.4 -> 0.35) 여부 검토")
    lines.append("- weakened_rate가 과도하게 높은 coda는 WEAK_RATIO 재조정(예: 0.7 -> 0.65) 검토")

    md_out = Path(args.output_md).resolve()
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] json report: {json_out}")
    print(f"[done] md report:   {md_out}")


if __name__ == "__main__":
    main()
