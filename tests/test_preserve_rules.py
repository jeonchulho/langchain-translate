import asyncio
from types import MethodType

from app.models import TranslationRequest
from app.translator import (
    TranslatorService,
    _protect_non_translatable,
    _restore_preserved,
)


def test_protect_and_restore_special_entities_roundtrip() -> None:
    text = (
        "See https://api.example.com/docs and contact admin@example.com. "
        "Keep user_id and requestPayload unchanged."
    )

    masked, mapping = _protect_non_translatable(text, protect_special_entities=True)

    assert masked != text
    assert any("https://api.example.com/docs" == value for value in mapping.values())
    assert any("admin@example.com" == value for value in mapping.values())

    restored = _restore_preserved(masked, mapping)
    assert restored == text


def test_allowlist_and_denylist_filtering() -> None:
    text = "Use https://api.example.com and https://docs.myservice.io in docs."

    masked, mapping = _protect_non_translatable(
        text,
        protect_special_entities=True,
        allowlist=[r"^https?://"],
        denylist=[r"example\.com"],
    )

    assert "https://api.example.com" in masked
    assert any("https://docs.myservice.io" == value for value in mapping.values())


def test_markdown_table_cell_translation_preserves_structure() -> None:
    service = object.__new__(TranslatorService)

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
        return f"KR:{chunk}"

    service._translate_chunk = MethodType(fake_translate_chunk, service)

    request = TranslationRequest(text="dummy", target_language="ko")
    table = (
        "| Field | Description | Default |\n"
        "| --- | --- | --- |\n"
        "| timeout_ms | Request timeout in milliseconds | 3000 |\n"
        "| retry_count | Number of retries | 2 |\n"
    )

    out = asyncio.run(
        service._translate_markdown_table(
            table,
            request,
            resolved_style_guide=request.style_guide,
            resolved_model_name="test-model",
            llm_provider="openai",
            llm_call_mode="ainvoke",
            llm_reasoning=False,
        )
    )

    assert "| KR:Field | KR:Description | KR:Default |" in out
    assert "| --- | --- | --- |" in out
    assert "| KR:timeout_ms | KR:Request timeout in milliseconds | 3000 |" in out
    assert "| KR:retry_count | KR:Number of retries | 2 |" in out
