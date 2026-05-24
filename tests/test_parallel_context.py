import asyncio
from types import MethodType

from app.models import TranslationRequest
from app.translator import (
    DOCUMENT_TYPE_STYLE_PRESETS,
    FIDELITY_FIRST_STYLE_GUIDE,
    TranslatorService,
    _build_overlap_context,
    _resolve_effective_chunk_overlap,
    _resolve_model_name,
    _resolve_parallel_context_text,
    _resolve_style_guide,
)


def _build_long_text() -> str:
    part_a = "A" * 700
    part_b = "B" * 700
    part_c = "C" * 700
    return f"{part_a}\n\n{part_b}\n\n{part_c}"


def test_parallel_glossary_only_context_is_applied_with_fidelity_first() -> None:
    service = object.__new__(TranslatorService)
    service.default_model_name = "gpt-4.1-mini"
    seen_previous_summaries: list[str] = []
    seen_retry_settings: list[tuple[int, bool]] = []

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
        seen_retry_settings.append((request.max_retries_per_chunk, llm_reasoning))
        return chunk

    service._translate_chunk = MethodType(fake_translate_chunk, service)

    request = TranslationRequest(
        text=_build_long_text(),
        target_language="ko",
        glossary={"Rotary": "로타리", "club": "클럽"},
        chunk_size=1200,
        chunk_overlap=50,
        max_retries_per_chunk=2,
        parallel_chunk_translation=True,
        max_concurrency=4,
        parallel_context_strategy="glossary-only",
        use_fidelity_first_preset=True,
        llm_reasoning=True,
        preserve_markdown_structures=False,
    )

    response = asyncio.run(service.translate(request))

    assert len(response.chunks) >= 2
    assert seen_previous_summaries[0] == "- Rotary: 로타리\n- club: 클럽"
    assert all("- Rotary: 로타리\n- club: 클럽" in summary for summary in seen_previous_summaries[1:])
    assert any(summary.startswith("A") or summary.startswith("B") or summary.startswith("C") for summary in seen_previous_summaries[1:])
    assert all(retries == 0 for retries, _ in seen_retry_settings)
    assert all(reasoning is False for _, reasoning in seen_retry_settings)


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

    service._translate_chunk = MethodType(fake_translate_chunk, service)

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
    assert seen_previous_summaries[0] == ""
    assert all(summary != "" for summary in seen_previous_summaries[1:])


def test_fidelity_first_style_guide_overrides_document_preset() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        style_guide="숫자 표기는 원문과 동일하게 유지",
        document_type="legal",
        use_document_type_style_preset=True,
        use_fidelity_first_preset=True,
    )

    resolved = _resolve_style_guide(request)
    assert FIDELITY_FIRST_STYLE_GUIDE in resolved
    assert "숫자 표기는 원문과 동일하게 유지" in resolved


def test_parallel_context_text_uses_glossary_only_when_enabled() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        glossary={"Rotary": "로타리", "club": "클럽"},
        parallel_context_strategy="glossary-only",
    )

    assert _resolve_parallel_context_text(request) == "- Rotary: 로타리\n- club: 클럽"


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


def test_auto_chunk_overlap_uses_ratio_and_max_cap() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        chunk_size=2000,
        chunk_overlap=500,
        auto_chunk_overlap=True,
        auto_chunk_overlap_ratio=0.1,
        auto_chunk_overlap_max=180,
    )

    assert _resolve_effective_chunk_overlap(request) == 180


def test_auto_chunk_overlap_keeps_more_context_for_fidelity_first() -> None:
    request = TranslationRequest(
        text="dummy",
        target_language="ko",
        chunk_size=1200,
        chunk_overlap=220,
        auto_chunk_overlap=True,
        auto_chunk_overlap_ratio=0.05,
        auto_chunk_overlap_max=150,
        use_fidelity_first_preset=True,
    )

    assert _resolve_effective_chunk_overlap(request) == 120


def test_overlap_context_prefers_sentence_boundary() -> None:
    previous_chunk = (
        "첫 문장입니다. "
        "두 번째 문장은 더 길고 문맥을 충분히 제공합니다. "
        "세 번째 문장도 충분히 길어서 경계 선택 조건을 만족합니다."
    )
    context = _build_overlap_context(previous_chunk, 80)

    assert context.startswith("두 번째")
