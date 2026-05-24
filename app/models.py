from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DEFAULT_STYLE_GUIDE = "기술 문서 스타일, 의미 보존, 과도한 의역 금지"


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source document text")
    source_language: str = Field(default="auto", description="e.g. en, ko, ja")
    target_language: str = Field(..., min_length=2, description="e.g. ko, en")
    glossary: Dict[str, str] = Field(default_factory=dict, description="term -> translation")
    style_guide: str = Field(
        default=DEFAULT_STYLE_GUIDE,
        description="Translation style requirements",
    )
    chunk_size: int = Field(default=3500, ge=1200, le=7000)
    chunk_overlap: int = Field(default=350, ge=50, le=1000)
    max_retries_per_chunk: int = Field(default=1, ge=0, le=3)
    parallel_chunk_translation: bool = Field(
        default=False,
        description="Translate chunks in parallel using asyncio.gather (disables chained previous-chunk summaries)",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent chunk translation tasks when parallel_chunk_translation=true",
    )
    parallel_context_strategy: Literal["none", "source-summary"] = Field(
        default="none",
        description="Context strategy for parallel mode: none | source-summary",
    )
    parallel_summary_max_chars: int = Field(
        default=120,
        ge=40,
        le=400,
        description="Maximum character length for source-summary context in parallel mode",
    )
    document_type: Literal["general", "technical", "legal", "marketing"] = Field(
        default="general",
        description="Document type used for optional summary-length presets",
    )
    use_document_type_summary_preset: bool = Field(
        default=False,
        description="When true, override parallel_summary_max_chars with a document-type preset",
    )
    use_document_type_style_preset: bool = Field(
        default=False,
        description="When true, inject document-type style guide preset into translation prompt",
    )
    llm_model_override: Optional[str] = Field(
        default=None,
        description="If set, use this model name for translation regardless of presets",
    )
    llm_provider: Literal["openai", "anthropic", "google", "ollama"] = Field(
        default="openai",
        description="LLM provider to use: openai | anthropic | google | ollama",
    )
    llm_call_mode: Literal["ainvoke", "astream"] = Field(
        default="ainvoke",
        description="LLM call mode: ainvoke for single response, astream for streamed chunks",
    )
    llm_reasoning: bool = Field(
        default=False,
        description="Enable model thinking mode via model_kwargs.extra_body.chat_template_kwargs.enable_thinking",
    )
    use_document_type_model_preset: bool = Field(
        default=False,
        description="When true, select model by document_type preset",
    )
    preserve_markdown_structures: bool = Field(
        default=True,
        description="Protect markdown code fences, inline code, and tables during translation",
    )
    translate_markdown_tables: bool = Field(
        default=True,
        description="Translate markdown table cells while keeping table structure",
    )
    preserve_special_entities: bool = Field(
        default=True,
        description="Protect URL, email, file path, and code-like identifiers during translation",
    )
    special_entity_allowlist_regex: List[str] = Field(
        default_factory=list,
        description="If provided, only entities matching any regex in this list are protected",
    )
    special_entity_denylist_regex: List[str] = Field(
        default_factory=list,
        description="Entities matching any regex in this list are not protected",
    )


class ChunkResult(BaseModel):
    index: int
    source_text: str
    translated_text: str
    quality_score: float
    issues: List[str] = Field(default_factory=list)
    retried: bool = False


class TranslationResponse(BaseModel):
    translated_text: str
    chunks: List[ChunkResult]
    average_quality_score: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: Optional[str] = None
