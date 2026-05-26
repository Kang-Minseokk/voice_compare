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


def _run_one(
    real_path: str,
    ref_path: str,
    text: str,
    align_backend: str,
    mfa_textgrid_dir: str,
    fail_on_mfa_error: bool,
) -> Dict[str, Any]:
    result = compare_module.compare(
        gt_path=ref_path,
        sample_path=real_path,
        text=text,
        align_backend=align_backend,
        mfa_textgrid_dir=mfa_textgrid_dir,
        fail_on_mfa_error=fail_on_mfa_error,
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


def _discover_jobs(root: Path, text_map: Dict[str, str]) -> Tuple[List[Dict[str, str]], List[str]]:
    jobs: List[Dict[str, str]] = []
    errors: List[str] = []

    real_files = sorted(p for p in root.iterdir() if p.is_file() and REAL_RE.match(p.name))
    for real in real_files:
        m = REAL_RE.match(real.name)
        assert m is not None
        person_idx, ref_idx = m.group(1), m.group(2)
        ref = root / f"hospital_{ref_idx}_ref.wav"
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
                "real_path": str(real),
                "ref_path": str(ref),
                "text": text,
            }
        )
    return jobs, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="병렬 WAV 배치 비교")
    parser.add_argument("--root", default=".", help="오디오 파일 폴더")
    parser.add_argument("--text", default=None, help="모든 비교에 공통으로 쓸 텍스트")
    parser.add_argument(
        "--text-map-json",
        default=None,
        help='ref 인덱스별 텍스트 맵 JSON 파일 경로 (예: {"1":"...", "2":"..."})',
    )
    parser.add_argument("--align-backend", default="wav2vec2", choices=["wav2vec2", "mfa"])
    parser.add_argument("--mfa-textgrid-dir", default="mfa_work/output")
    parser.add_argument("--fail-on-mfa-error", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-json", default="batch_compare_results.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    text_map = _load_text_map(args.text, args.text_map_json)
    jobs, precheck_errors = _discover_jobs(root, text_map)

    print(f"[info] root={root}")
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
    payload = {
        "root": str(root),
        "align_backend": args.align_backend,
        "job_count": len(jobs),
        "success_count": len(results),
        "precheck_errors": precheck_errors,
        "runtime_errors": runtime_errors,
        "results": results,
    }
    out = Path(args.output_json).resolve()
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
