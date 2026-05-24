import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .models import HealthResponse, TranslationRequest, TranslationResponse
from .translator import TranslatorService

load_dotenv()

app = FastAPI(title="LangChain Translate MVP", version="0.1.0")
translator = TranslatorService()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=translator.model_name)


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    try:
        return await translator.translate(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
