import argparse
import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import TranslationRequest
from app.translator import TranslatorService


class DemoTranslatorService(TranslatorService):
    async def _translate_chunk(
        self,
        chunk: str,
        request: TranslationRequest,
        previous_summary: str,
        resolved_style_guide: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        # 오프라인 데모 실행용 간이 번역기: 실제 LLM 호출 없이 형태만 재현
        return "[DEMO-EN] " + chunk

    async def _summarize(
        self,
        translated: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        return translated[:120]

    async def _summarize_source_for_parallel(
        self,
        source_text: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        return source_text[:120]


async def run(use_real_llm: bool, provider: str, model_override: str | None) -> None:
    input_path = Path("rotary_constitution_ko.txt")
    mode_label = "real" if use_real_llm else "demo"
    output_path = Path(f"rotary_constitution_en_output_{mode_label}_{provider}.txt")

    source_text = input_path.read_text(encoding="utf-8")

    if model_override:
        resolved_model = model_override
    elif provider == "ollama":
        resolved_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    elif provider == "openai":
        resolved_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    else:
        resolved_model = None

    request = TranslationRequest(
        text=source_text,
        source_language="ko",
        target_language="en",
        chunk_size=2200,
        # 스키마 최소 제약(>=50) 범위에서 중복을 최소화한다.
        chunk_overlap=50,
        parallel_chunk_translation=False,
        llm_provider=provider,
        llm_model_override=resolved_model,
        llm_call_mode="ainvoke",
        llm_reasoning=False,
    )

    service = TranslatorService() if use_real_llm else DemoTranslatorService()
    response = await service.translate(request)

    output_path.write_text(response.translated_text, encoding="utf-8")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"chunks: {len(response.chunks)}")
    print(f"avg_quality: {response.average_quality_score}")
    print("preview:")
    print(response.translated_text[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a text file and run translation example")
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real provider API calls (requires API key and model access)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google", "ollama"],
        default="openai",
        help="LLM provider for real mode (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. If omitted, provider defaults are used.",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            use_real_llm=args.real_llm,
            provider=args.provider,
            model_override=args.model,
        )
    )
