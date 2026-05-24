import argparse
import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import TranslationRequest
from app.translator import TranslatorService, _build_splitter, _resolve_effective_chunk_overlap


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


async def run(
    use_real_llm: bool,
    provider: str,
    model_override: str | None,
    fidelity_first: bool,
    parallel: bool,
    parallel_context_strategy: str,
    auto_overlap: bool,
    auto_overlap_ratio: float,
    auto_overlap_max: int,
    chunk_overlap: int,
    compare_overlap_demo: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "rotary_constitution_ko.txt"
    mode_label = "real" if use_real_llm else "demo"
    fidelity_label = "fidelity" if fidelity_first else "standard"
    parallel_label = "parallel" if parallel else "serial"
    output_path = script_dir / f"rotary_constitution_en_output_{mode_label}_{fidelity_label}_{parallel_label}_{provider}.txt"

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
        chunk_overlap=chunk_overlap,
        auto_chunk_overlap=auto_overlap,
        auto_chunk_overlap_ratio=auto_overlap_ratio,
        auto_chunk_overlap_max=auto_overlap_max,
        parallel_chunk_translation=parallel,
        parallel_context_strategy=parallel_context_strategy,
        use_fidelity_first_preset=fidelity_first,
        llm_provider=provider,
        llm_model_override=resolved_model,
        llm_call_mode="ainvoke",
        llm_reasoning=False,
    )

    service = TranslatorService() if use_real_llm else DemoTranslatorService()

    if compare_overlap_demo and not use_real_llm:
        effective_overlap = _resolve_effective_chunk_overlap(request)
        splitter = _build_splitter(request.chunk_size, effective_overlap)
        raw_chunks = splitter.split_text(source_text)
        legacy_text = "\n".join("[LEGACY-EN] " + c for c in raw_chunks)
        legacy_output = script_dir / "rotary_constitution_en_output_demo_legacy_overlap.txt"
        legacy_output.write_text(legacy_text, encoding="utf-8")

    response = await service.translate(request)

    output_path.write_text(response.translated_text, encoding="utf-8")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"chunks: {len(response.chunks)}")
    if request.auto_chunk_overlap:
        print(
            "effective_overlap:",
            _resolve_effective_chunk_overlap(request),
            f"(ratio={request.auto_chunk_overlap_ratio}, max={request.auto_chunk_overlap_max})",
        )
    if compare_overlap_demo and not use_real_llm:
        print("legacy_output:", script_dir / "rotary_constitution_en_output_demo_legacy_overlap.txt")
        print("compare_note: legacy는 오버랩 중복 포함, current는 중복 prefix 제거")
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
    parser.add_argument(
        "--fidelity-first",
        action="store_true",
        help="Bias the translation toward literal fidelity and glossary-only context",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel chunk translation to test glossary-only context",
    )
    parser.add_argument(
        "--parallel-context-strategy",
        choices=["none", "glossary-only"],
        default="none",
        help="Context strategy for parallel mode (default: none)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Requested chunk overlap size (default: 120)",
    )
    parser.add_argument(
        "--auto-overlap",
        action="store_true",
        help="Enable automatic effective overlap tuning",
    )
    parser.add_argument(
        "--auto-overlap-ratio",
        type=float,
        default=0.15,
        help="Overlap ratio used when --auto-overlap is enabled (default: 0.15)",
    )
    parser.add_argument(
        "--auto-overlap-max",
        type=int,
        default=450,
        help="Maximum overlap cap when --auto-overlap is enabled (default: 450)",
    )
    parser.add_argument(
        "--compare-overlap-demo",
        action="store_true",
        help="In demo mode, write legacy overlap output for before/after comparison",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            use_real_llm=args.real_llm,
            provider=args.provider,
            model_override=args.model,
            fidelity_first=args.fidelity_first,
            parallel=args.parallel,
            parallel_context_strategy=args.parallel_context_strategy,
            auto_overlap=args.auto_overlap,
            auto_overlap_ratio=args.auto_overlap_ratio,
            auto_overlap_max=args.auto_overlap_max,
            chunk_overlap=args.chunk_overlap,
            compare_overlap_demo=args.compare_overlap_demo,
        )
    )
