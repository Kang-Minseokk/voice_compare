"""고려인 발음 교정 비교 엔진 API.

POST /compare — 두 오디오 + 텍스트를 받아 분석기별 점수와 음절별 진단을 반환.

핵심 설계 원칙:
  - 음절 단위 진단을 그대로 노출 (각 분석기의 per_syllable 결과를 모두 응답에 포함)
  - 종합 점수는 참고용. 클라이언트가 어떤 분석기/음절에 주목할지 결정하도록.
  - 모델 로딩은 서버 시작 시 1회 (~1.2GB)

실행:
  python api.py            # uvicorn으로 :8000 띄움
  또는
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from alignment import Alignment, load_model
from audio_utils import FRAME_STRIDE_SEC
from compare import compare


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[startup] loading wav2vec2 model...")
    load_model()
    print("[startup] model ready.")
    yield


app = FastAPI(
    title="Pronunciation Comparison API",
    description="고려인 화자 발음 교정용 음성 비교 엔진. "
                "분절(formants, MFCC, VOT) + 초분절(pitch, energy) 5축 진단.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _alignment_to_list(alignment: Alignment) -> List[Dict[str, Any]]:
    return [
        {
            "char": s.char,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_sec": s.start_frame * FRAME_STRIDE_SEC,
            "end_sec": (s.end_frame + 1) * FRAME_STRIDE_SEC,
        }
        for s in alignment.spans
    ]


def _result_to_dict(result) -> Dict[str, Any]:
    return {
        "overall_score": result.overall_score,
        "analyzers": {
            name: {
                "score": ar.score,
                "details": ar.details,
                "per_syllable": [
                    {
                        "char": ps.char,
                        "score": ps.score,
                        "details": ps.details,
                    }
                    for ps in ar.per_syllable
                ],
            }
            for name, ar in result.analyzers.items()
        },
        "alignment": {
            "gt": _alignment_to_list(result.gt_alignment),
            "sample": _alignment_to_list(result.sample_alignment),
        },
    }


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    tmp = NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        data = await upload.read()
        tmp.write(data)
    finally:
        tmp.close()
    return Path(tmp.name)


@app.post(
    "/compare",
    summary="Compare two audio files against text",
    description=(
        "두 오디오 파일과 발화 텍스트를 받아 5개 분석기로 비교합니다.\n\n"
        "**입력**\n"
        "- `gt_audio`: 원어민 녹음 (multipart)\n"
        "- `sample_audio`: 사용자 녹음 (multipart)\n"
        "- `text`: 발화한 한국어 텍스트\n\n"
        "**출력**: overall_score + 분석기별 결과(score, details, per_syllable) "
        "+ 음절 alignment."
    ),
)
async def compare_endpoint(
    gt_audio: UploadFile = File(...),
    sample_audio: UploadFile = File(...),
    text: str = Form(...),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    gt_path = await _save_upload(gt_audio)
    sample_path = await _save_upload(sample_audio)

    try:
        result = compare(str(gt_path), str(sample_path), text)
        return _result_to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"comparison failed: {e}")
    finally:
        gt_path.unlink(missing_ok=True)
        sample_path.unlink(missing_ok=True)


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.get("/", summary="API info")
def root():
    return {
        "name": "Pronunciation Comparison API",
        "analyzers": ["formants", "mfcc_dtw", "pitch", "energy", "vot"],
        "endpoints": {
            "POST /compare": "Compare two audio files against text",
            "GET /health": "Health check",
            "GET /docs": "Interactive API docs (Swagger UI)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
