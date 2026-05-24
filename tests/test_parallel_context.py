import asyncio
from types import MethodType

from app.models import TranslationRequest
from app.translator import (
    DOCUMENT_TYPE_STYLE_PRESETS,
    PARALLEL_SUMMARY_MAX_CHARS,
    TranslatorService,
    _clamp_summary_text,
    _resolve_model_name,
    _resolve_parallel_summary_max_chars,
    _resolve_style_guide,
)


def _build_long_text() -> str:
    part_a = "A" * 700
    part_b = "B" * 700
    part_c = "C" * 700
    return f"{part_a}\n\n{part_b}\n\n{part_c}"


def test_parallel_source_summary_context_is_applied() -> None:
    service = object.__new__(TranslatorService)
    service.default_model_name = "gpt-4.1-mini"
    seen_previous_summaries: list[str] = []

    async def fake_translate_chunk(
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
        seen_previous_summaries.append(previous_summary)
        return chunk

    async def fake_summarize_source_for_parallel(
        self,
        source_text: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        return f"S:{source_text[:8]}"

    service._translate_chunk = MethodType(fake_translate_chunk, service)
    service._summarize_source_for_parallel = MethodType(fake_summarize_source_for_parallel, service)

    request = TranslationRequest(
        text=_build_long_text(),
        target_language="ko",
        chunk_size=1200,
        chunk_overlap=50,
        max_retries_per_chunk=0,
        parallel_chunk_translation=True,
        max_concurrency=4,
        parallel_context_strategy="source-summary",
        preserve_markdown_structures=False,
    )

    response = asyncio.run(service.translate(request))

    assert len(response.chunks) >= 2
    assert any(summary.startswith("S:") for summary in seen_previous_summaries[1:])


def test_parallel_none_context_stays_empty() -> None:
    service = object.__new__(TranslatorService)
    service.default_model_name = "gpt-4.1-mini"
    seen_previous_summaries: list[str] = []

    async def fake_translate_chunk(
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
        seen_previous_summaries.append(previous_summary)
        return chunk

    async def fake_summarize_source_for_parallel(
        self,
        source_text: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        return "S:unused"

    service._translate_chunk = MethodType(fake_translate_chunk, service)
    service._summarize_source_for_parallel = MethodType(fake_summarize_source_for_parallel, service)

    request = TranslationRequest(
        text=_build_long_text(),
        target_language="ko",
        chunk_size=1200,
        chunk_overlap=50,
        max_retries_per_chunk=0,
        parallel_chunk_translation=True,
        max_concurrency=4,
        parallel_context_strategy="none",
        preserve_markdown_structures=False,
    )

    response = asyncio.run(service.translate(request))

    assert len(response.chunks) >= 2
    assert all(summary == "" for summary in seen_previous_summaries)


def test_clamp_parallel_summary_text_limits_length() -> None:
    raw = "X" * (PARALLEL_SUMMARY_MAX_CHARS + 40)
    clamped = _clamp_summary_text(raw)

    assert len(clamped) == PARALLEL_SUMMARY_MAX_CHARS


def test_clamp_parallel_summary_text_normalizes_whitespace() -> None:
    raw = "alpha   beta\n\n gamma\t delta"
    clamped = _clamp_summary_text(raw)

    assert clamped == "alpha beta gamma delta"


def test_parallel_context_summary_uses_request_max_chars() -> None:
    service = object.__new__(TranslatorService)

    async def fake_summarize_source_for_parallel(
        self,
        source_text: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        return "Z" * 80

    service._summarize_source_for_parallel = MethodType(fake_summarize_source_for_parallel, service)

    contexts = asyncio.run(
        service._build_parallel_context_summaries(
            chunks=["chunk-1", "chunk-2"],
            max_concurrency=2,
            max_summary_chars=25,
            resolved_model_name="test-model",
            llm_provider="openai",
            llm_call_mode="ainvoke",
            llm_reasoning=False,
        )
    )

    assert contexts[0] == ""
    assert len(contexts[1]) == 25


def test_parallel_context_summary_uses_document_type_preset_when_enabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        parallel_summary_max_chars=77,
        document_type="legal",
        use_document_type_summary_preset=True,
    )

    assert _resolve_parallel_summary_max_chars(request) == 220


def test_parallel_context_summary_uses_request_value_when_preset_disabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        parallel_summary_max_chars=77,
        document_type="legal",
        use_document_type_summary_preset=False,
    )

    assert _resolve_parallel_summary_max_chars(request) == 77


def test_style_guide_uses_document_type_preset_when_enabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        style_guide="기술 문서 스타일, 의미 보존, 과도한 의역 금지",
        document_type="legal",
        use_document_type_style_preset=True,
    )

    assert _resolve_style_guide(request) == DOCUMENT_TYPE_STYLE_PRESETS["legal"]


def test_style_guide_merges_custom_style_when_preset_enabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        style_guide="숫자 표기는 원문과 동일하게 유지",
        document_type="technical",
        use_document_type_style_preset=True,
    )

    resolved = _resolve_style_guide(request)
    assert DOCUMENT_TYPE_STYLE_PRESETS["technical"] in resolved
    assert "숫자 표기는 원문과 동일하게 유지" in resolved


def test_model_name_uses_override_first() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        llm_model_override="gpt-4.1",
        use_document_type_model_preset=True,
        document_type="legal",
    )

    assert _resolve_model_name(request, "gpt-4.1-mini") == "gpt-4.1"


def test_model_name_uses_default_when_preset_disabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        use_document_type_model_preset=False,
        document_type="technical",
    )

    assert _resolve_model_name(request, "gpt-4.1-mini") == "gpt-4.1-mini"
