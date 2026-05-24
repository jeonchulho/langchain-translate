import asyncio
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import DEFAULT_STYLE_GUIDE, ChunkResult, TranslationRequest, TranslationResponse
from .prompt_templates import (
    CHUNK_HUMAN_TEMPLATE,
    CHUNK_SYSTEM_TEMPLATE,
    PARALLEL_SOURCE_SUMMARY_HUMAN_TEMPLATE,
    PARALLEL_SOURCE_SUMMARY_SYSTEM_TEMPLATE,
    SUMMARY_HUMAN_TEMPLATE,
    SUMMARY_SYSTEM_TEMPLATE,
)
from .preserve_rules import DEFAULT_SPECIAL_ENTITY_PATTERNS


@dataclass
class ChunkQuality:
    """청크 번역 품질 평가 결과를 담는 데이터 클래스.

    - score: 0.0~1.0 범위의 정규화된 품질 점수
    - issues: 감지된 품질 이슈 목록(예: 숫자 불일치, 길이 비율 이상)
    """

    score: float
    issues: List[str]


PRESERVE_TOKEN_PREFIX = "__LC_PRESERVE_"
PRESERVE_TOKEN_SUFFIX = "__"
PARALLEL_SUMMARY_MAX_CHARS = 120
DOCUMENT_TYPE_SUMMARY_PRESETS: Dict[str, int] = {
    "general": 120,
    "technical": 140,
    "legal": 220,
    "marketing": 100,
}
DOCUMENT_TYPE_STYLE_PRESETS: Dict[str, str] = {
    "general": "중립적이고 명확한 문체, 과도한 수식어 최소화",
    "technical": "기술 문서 문체, 용어 일관성 최우선, 정의와 절차를 명확히 유지",
    "legal": "법률 문서 문체, 조항 번호/용어 정의/조건 표현을 엄격히 보존",
    "marketing": "마케팅 문체, 설득력과 가독성을 유지하되 사실/수치 왜곡 금지",
}
DOCUMENT_TYPE_MODEL_PRESETS_ENV: Dict[str, str] = {
    "general": "OPENAI_MODEL_GENERAL",
    "technical": "OPENAI_MODEL_TECHNICAL",
    "legal": "OPENAI_MODEL_LEGAL",
    "marketing": "OPENAI_MODEL_MARKETING",
}
FIDELITY_FIRST_STYLE_GUIDE = "원문 구조, 숫자, 고유명사, 고정 용어를 우선 보존하고 의역을 최소화하는 직역 중심 문체"


def _new_token(index: int) -> str:
    """보존 토큰을 생성한다.

    번역 전에 보호해야 하는 영역(코드, 표, URL 등)을 치환할 때 사용하는
    고유 토큰 문자열을 만든다.
    """

    return f"{PRESERVE_TOKEN_PREFIX}{index:04d}{PRESERVE_TOKEN_SUFFIX}"


def _protect_code_fences(text: str, start_index: int = 0) -> Tuple[str, Dict[str, str], int]:
    """마크다운 코드 펜스를 보호 토큰으로 치환한다.

    반환값:
    - 치환된 텍스트
    - 토큰 -> 원문 코드 블록 매핑
    - 다음 단계에서 사용할 토큰 인덱스
    """

    mapping: Dict[str, str] = {}
    counter = start_index

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        token = _new_token(counter)
        mapping[token] = match.group(0)
        counter += 1
        return token

    masked = re.sub(r"```[\s\S]*?```", repl, text)
    return masked, mapping, counter


def _protect_inline_code(text: str, start_index: int = 0) -> Tuple[str, Dict[str, str], int]:
    """인라인 코드(`...`)를 보호 토큰으로 치환한다.

    코드 식별자/명령어가 번역되는 문제를 막기 위해 먼저 마스킹한다.
    """

    mapping: Dict[str, str] = {}
    counter = start_index

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        token = _new_token(counter)
        mapping[token] = match.group(0)
        counter += 1
        return token

    masked = re.sub(r"`[^`\n]+`", repl, text)
    return masked, mapping, counter


def _protect_by_pattern(
    text: str,
    pattern: str,
    start_index: int = 0,
    flags: int = 0,
    should_protect: Callable[[str], bool] | None = None,
) -> Tuple[str, Dict[str, str], int]:
    """정규식 패턴으로 매칭되는 텍스트를 토큰으로 치환한다.

    should_protect 콜백이 있으면 매칭 값별 보호 여부를 추가로 결정한다.
    여러 보호 규칙(URL, 이메일, 경로 등)의 공통 유틸로 사용한다.
    """

    mapping: Dict[str, str] = {}
    counter = start_index

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        value = match.group(0)
        if should_protect and not should_protect(value):
            return value
        token = _new_token(counter)
        mapping[token] = value
        counter += 1
        return token

    masked = re.sub(pattern, repl, text, flags=flags)
    return masked, mapping, counter


def _should_protect_entity(value: str, allowlist: List[str], denylist: List[str]) -> bool:
    """allowlist/denylist 규칙으로 엔티티 보호 여부를 판단한다.

    - allowlist가 비어있지 않으면, 하나라도 매칭되어야 보호
    - denylist에 하나라도 매칭되면 보호하지 않음
    """

    if allowlist and not any(re.search(pattern, value) for pattern in allowlist):
        return False
    if denylist and any(re.search(pattern, value) for pattern in denylist):
        return False
    return True


def _protect_special_entities(
    text: str,
    start_index: int = 0,
    allowlist: List[str] | None = None,
    denylist: List[str] | None = None,
) -> Tuple[str, Dict[str, str], int]:
    """특수 엔티티(URL/이메일/경로/식별자 등)를 일괄 보호한다.

    DEFAULT_SPECIAL_ENTITY_PATTERNS를 순회하며 보호 토큰 치환을 수행하고,
    치환 매핑을 병합해서 반환한다.
    """

    mapping: Dict[str, str] = {}
    masked = text
    counter = start_index
    allow_patterns = allowlist or []
    deny_patterns = denylist or []

    def decision(value: str) -> bool:
        return _should_protect_entity(value, allow_patterns, deny_patterns)

    for pattern, flags in DEFAULT_SPECIAL_ENTITY_PATTERNS:
        masked, current_map, counter = _protect_by_pattern(
            masked,
            pattern,
            counter,
            flags,
            should_protect=decision,
        )
        mapping.update(current_map)

    return masked, mapping, counter


def _is_table_separator(line: str) -> bool:
    """마크다운 표 구분선 행인지 확인한다.

    예: | --- | :---: | ---: |
    """

    stripped = line.strip()
    if "|" not in stripped:
        return False
    return bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", stripped))


def _protect_markdown_tables(text: str, start_index: int = 0) -> Tuple[str, Dict[str, str], int]:
    """마크다운 표 블록 전체를 보호 토큰으로 치환한다.

    헤더+구분선 패턴을 기준으로 표를 인식해 블록 단위로 보존하며,
    이후 표 셀 번역이 필요한 경우 별도 단계에서 복원/재번역한다.
    """

    lines = text.splitlines(keepends=True)
    out: List[str] = []
    mapping: Dict[str, str] = {}
    counter = start_index
    i = 0

    while i < len(lines):
        current = lines[i]

        if i + 1 < len(lines) and "|" in current and _is_table_separator(lines[i + 1]):
            block = [current, lines[i + 1]]
            i += 2

            while i < len(lines):
                if "|" not in lines[i].strip():
                    break
                block.append(lines[i])
                i += 1

            token = _new_token(counter)
            mapping[token] = "".join(block)
            out.append(token)
            counter += 1
            continue

        out.append(current)
        i += 1

    return "".join(out), mapping, counter


def _protect_non_translatable(
    text: str,
    protect_special_entities: bool = True,
    allowlist: List[str] | None = None,
    denylist: List[str] | None = None,
) -> Tuple[str, Dict[str, str]]:
    """번역하면 안 되는 영역을 순차적으로 보호한다.

    처리 순서:
    1) 코드 펜스
    2) 인라인 코드
    3) 특수 엔티티(옵션)
    """

    masked, map_code_fence, idx = _protect_code_fences(text, 0)
    masked, map_inline_code, idx = _protect_inline_code(masked, idx)
    map_special: Dict[str, str] = {}

    if protect_special_entities:
        masked, map_special, idx = _protect_special_entities(masked, idx, allowlist, denylist)

    mapping: Dict[str, str] = {}
    mapping.update(map_code_fence)
    mapping.update(map_inline_code)
    mapping.update(map_special)
    return masked, mapping


def _is_simple_non_translatable_cell(text: str) -> bool:
    """표 셀이 번역 불필요한 단순 값인지 판단한다.

    빈 셀, 숫자/기호 위주 셀은 번역하지 않고 원문 유지한다.
    """

    value = text.strip()
    if not value:
        return True
    if re.fullmatch(r"[\d\s.,:%()+\-/*]+", value):
        return True
    return False


def _parse_markdown_row(line: str) -> List[str]:
    """마크다운 표 행을 셀 리스트로 파싱한다."""

    work = line.strip()
    if work.startswith("|"):
        work = work[1:]
    if work.endswith("|"):
        work = work[:-1]
    return [cell.strip() for cell in work.split("|")]


def _build_markdown_row(cells: List[str]) -> str:
    """셀 리스트를 마크다운 표 행 문자열로 조합한다."""

    return "| " + " | ".join(cells) + " |"


def _restore_preserved(text: str, mapping: Dict[str, str]) -> str:
    """보호 토큰을 원문으로 복원한다.

    번역 완료 후 마스킹했던 코드/엔티티/표 블록을 다시 원래 값으로 되돌린다.
    """

    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    return restored


def _extract_numbers(text: str) -> List[str]:
    """텍스트에서 숫자 토큰을 추출한다.

    품질 검사에서 원문/번역문의 숫자 보존 여부를 확인할 때 사용한다.
    """

    return re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)", text)


def _quality_check(source: str, translated: str) -> ChunkQuality:
    """청크 번역 품질을 휴리스틱으로 평가한다.

    운영 관점 포인트:
    - 완전한 의미 평가 대신 빠른 1차 게이트로 사용해 지연 시간을 최소화한다.
    - 숫자 보존 실패/미번역 등 치명적인 케이스를 우선 탐지한다.
    - 점수는 재시도 여부 판단 기준으로 사용되며, 모델별 편차가 존재할 수 있다.

    현재 검사 규칙:
    - 숫자 불일치
    - 길이 비율 이상(과도하게 짧거나 김)
    - 원문과 동일(미번역 가능성)
    """

    issues: List[str] = []

    src_nums = sorted(_extract_numbers(source))
    tgt_nums = sorted(_extract_numbers(translated))
    if src_nums != tgt_nums:
        issues.append("숫자 불일치")

    src_len = max(len(source.strip()), 1)
    tgt_len = max(len(translated.strip()), 1)
    ratio = tgt_len / src_len
    if ratio < 0.45 or ratio > 1.9:
        issues.append("길이 비율 이상")

    if translated.strip() == source.strip():
        issues.append("미번역 가능성")

    score = max(0.0, 1.0 - (0.34 * len(issues)))
    return ChunkQuality(score=round(score, 3), issues=issues)


def _build_glossary_text(glossary: Dict[str, str]) -> str:
    """용어집 딕셔너리를 프롬프트 주입용 텍스트로 변환한다."""

    if not glossary:
        return "없음"
    return "\n".join(f"- {k}: {v}" for k, v in glossary.items())


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """문서 분할기(RecursiveCharacterTextSplitter)를 구성한다.

    문장/개행 우선 분리 후 필요 시 더 작은 단위로 분할하도록 separators를 배치한다.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", "。", "！", "？", " ", ""],
    )


def _resolve_effective_chunk_overlap(request: TranslationRequest) -> int:
    """요청 설정에서 실제 사용할 오버랩 길이를 계산한다."""

    if not request.auto_chunk_overlap:
        return request.chunk_overlap

    ratio_overlap = int(request.chunk_size * request.auto_chunk_overlap_ratio)
    effective = max(50, min(request.chunk_overlap, request.auto_chunk_overlap_max, ratio_overlap))

    # fidelity-first에서는 문맥 손실을 줄이기 위해 최소 문맥량을 보장한다.
    if request.use_fidelity_first_preset:
        effective = max(effective, min(request.chunk_overlap, 120))

    return effective


def _clamp_summary_text(text: str, max_chars: int = PARALLEL_SUMMARY_MAX_CHARS) -> str:
    """요약 텍스트 길이를 최대 문자 수로 제한한다.

    병렬 컨텍스트 전략에서 과도한 요약 길이가 프롬프트를 압박하지 않도록 정규화/절단한다.
    """

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def _normalize_llm_content(content: object) -> str:
    """LLM 응답 content를 문자열로 정규화한다.

    provider에 따라 content 타입이 문자열 또는 list(dict/text)로 올 수 있어
    호출부에서 일관되게 문자열로 처리할 수 있도록 변환한다.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


async def _run_llm_with_mode(
    llm: Any,
    messages: List[SystemMessage | HumanMessage],
    call_mode: str,
) -> str:
    """지정된 호출 모드로 LLM을 실행한다.

    운영 관점 포인트:
    - astream은 긴 출력에서 초기 응답 체감 속도를 높일 수 있다.
    - ainvoke는 구현/디버깅이 단순해 기본 모드로 유지하기 쉽다.
    - provider별 content 포맷 차이는 _normalize_llm_content에서 흡수한다.

    모드:
    - astream: 스트리밍 청크를 이어붙여 최종 문자열 생성
    - ainvoke: 단일 응답을 받아 문자열로 변환
    """

    if call_mode == "astream":
        chunks: List[str] = []
        async for chunk in llm.astream(messages):
            chunks.append(_normalize_llm_content(chunk.content))
        return "".join(chunks)

    response = await llm.ainvoke(messages)
    return _normalize_llm_content(response.content)


def _resolve_parallel_summary_max_chars(request: TranslationRequest) -> int:
    """병렬 컨텍스트 요약 최대 길이를 결정한다.

    문서 타입 프리셋 사용 여부에 따라 preset 값 또는 요청값을 반환한다.
    """

    if request.use_document_type_summary_preset:
        return DOCUMENT_TYPE_SUMMARY_PRESETS.get(request.document_type, PARALLEL_SUMMARY_MAX_CHARS)
    return request.parallel_summary_max_chars


def _resolve_style_guide(request: TranslationRequest) -> str:
    """최종 스타일 가이드를 계산한다.

    문서 타입 스타일 프리셋 사용 시 preset과 사용자 추가 요구사항을 병합한다.
    """

    if request.use_fidelity_first_preset:
        custom = request.style_guide.strip()
        if not custom or custom == DEFAULT_STYLE_GUIDE:
            return FIDELITY_FIRST_STYLE_GUIDE
        return f"{FIDELITY_FIRST_STYLE_GUIDE}; 추가 스타일 요구사항: {custom}"

    if not request.use_document_type_style_preset:
        return request.style_guide

    preset = DOCUMENT_TYPE_STYLE_PRESETS.get(request.document_type, DOCUMENT_TYPE_STYLE_PRESETS["general"])
    custom = request.style_guide.strip()

    if not custom or custom == DEFAULT_STYLE_GUIDE:
        return preset

    return f"{preset}; 추가 스타일 요구사항: {custom}"


def _resolve_parallel_context_text(request: TranslationRequest) -> str:
    """병렬 번역에서 사용할 보조 문맥 텍스트를 결정한다.

    - none: 빈 문자열
    - glossary-only: 용어집만 주입
    - fidelity-first 프리셋이 켜진 경우에도 glossary-only를 우선 사용한다.
    """

    if not request.glossary:
        return ""
    if request.use_fidelity_first_preset or request.parallel_context_strategy == "glossary-only":
        return _build_glossary_text(request.glossary)
    return ""


def _resolve_model_name(request: TranslationRequest, default_model_name: str) -> str:
    """요청에 사용할 최종 모델명을 결정한다.

    우선순위:
    1) llm_model_override
    2) 문서 타입 모델 프리셋(환경변수)
    3) 기본 모델
    """

    if request.llm_model_override:
        return request.llm_model_override.strip()

    if not request.use_document_type_model_preset:
        return default_model_name

    env_key = DOCUMENT_TYPE_MODEL_PRESETS_ENV.get(request.document_type)
    if env_key:
        return os.getenv(env_key, default_model_name)

    return default_model_name


def _find_sentence_safe_context_start(text: str, min_context_chars: int) -> int:
    """문장 경계에 맞는 컨텍스트 시작 지점을 찾는다.

    반환값은 text의 시작 인덱스이다.
    - 문장 끝(., !, ?, 。, ！, ？, 줄바꿈) 직후를 우선한다.
    - 충분히 긴 문맥을 확보할 수 있는 가장 앞쪽 경계를 고른다.
    - 적절한 경계를 못 찾으면 0을 반환해 원문 suffix 전체를 사용한다.
    """

    if not text:
        return 0

    boundary_pattern = re.compile(r"[.!?。！？\n]")
    for match in boundary_pattern.finditer(text):
        start = match.end()
        if len(text) - start >= min_context_chars:
            while start < len(text) and text[start].isspace():
                start += 1
            return start

    return 0


def _build_overlap_context(previous_chunk: str, chunk_overlap: int) -> str:
    """이전 청크 끝부분에서 현재 청크용 문맥을 만든다.

    오버랩은 실제 번역 대상이 아니라 문맥 힌트로만 사용한다.
    문장 경계에 맞는 시작점을 우선하고, 없으면 요청된 오버랩 길이만큼의 suffix를 사용한다.
    """

    if not previous_chunk or chunk_overlap <= 0:
        return ""

    overlap_length = min(chunk_overlap, len(previous_chunk))
    suffix = previous_chunk[-overlap_length:]
    min_context_chars = max(40, overlap_length // 2)
    start = _find_sentence_safe_context_start(suffix, min_context_chars)
    context = suffix[start:] if start < len(suffix) else suffix
    return context.lstrip()


def _strip_overlap_prefix(chunk: str, chunk_overlap: int, is_first_chunk: bool) -> str:
    """중복 출력을 막기 위해 현재 청크의 오버랩 prefix를 제거한다."""

    if is_first_chunk or chunk_overlap <= 0:
        return chunk

    cut = min(chunk_overlap, len(chunk))
    body = chunk[cut:]
    return body if body else chunk


class TranslatorService:
    """긴 문서 번역 파이프라인을 제공하는 서비스 클래스.

        운영 관점 포인트:
        - 대용량 문서 처리 시 안정성(보호/복원), 비용(재시도/요약 길이),
            지연 시간(병렬/스트리밍)의 균형을 맞추는 것을 목표로 한다.
        - provider 분기와 캐시를 통해 다중 모델 운영과 설정 실험을 쉽게 만든다.

    주요 역할:
    - 텍스트 보호/복원
    - 청크 번역 및 품질 검사/재시도
    - 병렬 번역/요약 컨텍스트 관리
    - provider별 LLM 인스턴스 생성/캐시
    """

    def __init__(self) -> None:
        """기본 모델/프롬프트/캐시를 초기화한다."""

        self.default_model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.model_name = self.default_model_name
        self.llm_cache: Dict[str, Any] = {}

        self.chunk_system = CHUNK_SYSTEM_TEMPLATE
        self.chunk_human = CHUNK_HUMAN_TEMPLATE
        self.summary_system = SUMMARY_SYSTEM_TEMPLATE
        self.summary_human = SUMMARY_HUMAN_TEMPLATE
        self.parallel_source_summary_system = PARALLEL_SOURCE_SUMMARY_SYSTEM_TEMPLATE
        self.parallel_source_summary_human = PARALLEL_SOURCE_SUMMARY_HUMAN_TEMPLATE

    @staticmethod
    def _instantiate_with_optional_reasoning(
        cls: Any,
        base_kwargs: Dict[str, Any],
        reasoning_kwargs: Dict[str, Any],
    ) -> Any:
        # provider SDK 버전에 따라 reasoning/thinking 관련 파라미터 지원 여부가 다르다.
        # 따라서 reasoning 파라미터를 우선 포함해 생성해보고,
        # TypeError(미지원 인자) 발생 시 기본 파라미터만으로 안전하게 폴백한다.
        #
        # 이 방식으로 API 요청 단의 `llm_reasoning` 플래그는 유지하면서도,
        # 특정 SDK 버전 차이로 서비스 전체가 실패하는 것을 방지할 수 있다.
        if reasoning_kwargs:
            try:
                return cls(**base_kwargs, **reasoning_kwargs)
            except TypeError:
                # Older provider SDKs may not support reasoning-related params.
                pass
        return cls(**base_kwargs)

    def _get_llm(self, provider: str, model_name: str, llm_reasoning: bool) -> Any:
        """provider와 모델명에 맞는 LLM 클라이언트를 반환한다.

        운영 관점 포인트:
        - cache_key에 provider/model/reasoning을 포함해 설정 오염을 방지한다.
        - provider 패키지 미설치 시 즉시 명확한 RuntimeError를 내어
          장애 원인 파악 시간을 줄인다.
        - reasoning 파라미터는 SDK 버전 차이를 고려해 best-effort로 적용한다.
        """

        cache_key = f"{provider}::{model_name}::reasoning={llm_reasoning}"
        if cache_key not in self.llm_cache:
            if provider == "openai":
                self.llm_cache[cache_key] = ChatOpenAI(
                    model=model_name,
                    temperature=0,
                    model_kwargs={
                        "extra_body": {
                            "chat_template_kwargs": {
                                "enable_thinking": llm_reasoning,
                            }
                        }
                    },
                )
            elif provider == "anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic
                except ImportError as exc:
                    raise RuntimeError(
                        "langchain-anthropic is required for llm_provider='anthropic'. "
                        "Install it with: pip install langchain-anthropic"
                    ) from exc
                self.llm_cache[cache_key] = self._instantiate_with_optional_reasoning(
                    ChatAnthropic,
                    {
                        "model": model_name,
                        "temperature": 0,
                    },
                    (
                        # Anthropic reasoning(extended thinking) 활성화 시도.
                        # 모델/SDK 조합에 따라 `thinking` 인자를 받지 않을 수 있으므로
                        # 헬퍼에서 자동 폴백 처리한다.
                        {
                            "thinking": {
                                "type": "enabled",
                                "budget_tokens": 1024,
                            }
                        }
                        if llm_reasoning
                        else {}
                    ),
                )
            elif provider == "google":
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                except ImportError as exc:
                    raise RuntimeError(
                        "langchain-google-genai is required for llm_provider='google'. "
                        "Install it with: pip install langchain-google-genai"
                    ) from exc
                self.llm_cache[cache_key] = self._instantiate_with_optional_reasoning(
                    ChatGoogleGenerativeAI,
                    {
                        "model": model_name,
                        "temperature": 0,
                    },
                    # Google Gemini 계열 reasoning budget 전달 시도.
                    # 미지원 버전에서는 헬퍼가 TypeError를 흡수하고 기본 생성으로 대체한다.
                    ({"thinking_budget": 1024} if llm_reasoning else {}),
                )
            elif provider == "ollama":
                try:
                    from langchain_ollama import ChatOllama
                except ImportError as exc:
                    raise RuntimeError(
                        "langchain-ollama is required for llm_provider='ollama'. "
                        "Install it with: pip install langchain-ollama"
                    ) from exc

                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.llm_cache[cache_key] = self._instantiate_with_optional_reasoning(
                    ChatOllama,
                    {
                        "model": model_name,
                        "temperature": 0,
                        "base_url": base_url,
                    },
                    # Ollama의 reasoning 옵션은 모델/버전별 지원 여부가 달라 best-effort 적용.
                    ({"reasoning": llm_reasoning} if llm_reasoning else {}),
                )
            else:
                raise ValueError(f"Unsupported llm_provider: {provider}")
        return self.llm_cache[cache_key]

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
        """단일 청크를 번역한다.

        운영 관점 포인트:
        - 이전 청크 요약(previous_summary)을 주입해 청크 경계 단절을 완화한다.
        - glossary/style을 매 호출에 명시해 문체/용어 일관성을 높인다.
        - 실제 호출은 _run_llm_with_mode로 위임해 실행 경로를 단순화한다.
        """

        ph = {
            "source_language": request.source_language,
            "target_language": request.target_language,
            "glossary": _build_glossary_text(request.glossary),
            "style_guide": resolved_style_guide,
            "previous_summary": previous_summary or "없음",
        }
        chunk_ph = {"chunk": chunk}

        messages = [
            SystemMessage(content=self.chunk_system.format(**ph)),
            HumanMessage(content=self.chunk_human.format(**chunk_ph)),
        ]
        return await _run_llm_with_mode(
            self._get_llm(llm_provider, resolved_model_name, llm_reasoning),
            messages,
            llm_call_mode,
        )

    async def _translate_markdown_table(
        self,
        table_text: str,
        request: TranslationRequest,
        resolved_style_guide: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> str:
        """마크다운 표를 셀 단위로 번역한다.

        표 구조(파이프/구분선)는 유지하고 셀 내용만 번역한다.
        숫자/기호 위주 단순 셀은 원문을 그대로 둔다.
        """

        lines = table_text.splitlines()
        if len(lines) < 2:
            return table_text

        translated_lines: List[str] = [lines[0], lines[1]]
        header_cells = _parse_markdown_row(lines[0])
        translated_header: List[str] = []
        for cell in header_cells:
            if _is_simple_non_translatable_cell(cell):
                translated_header.append(cell)
            else:
                translated_header.append(
                    (
                        await self._translate_chunk(
                            cell,
                            request,
                            "없음",
                            resolved_style_guide,
                            resolved_model_name,
                            llm_provider,
                            llm_call_mode,
                            llm_reasoning,
                        )
                    ).strip()
                )
        translated_lines[0] = _build_markdown_row(translated_header)

        for row in lines[2:]:
            if "|" not in row:
                translated_lines.append(row)
                continue

            row_cells = _parse_markdown_row(row)
            translated_cells: List[str] = []
            for cell in row_cells:
                if _is_simple_non_translatable_cell(cell):
                    translated_cells.append(cell)
                else:
                    translated_cells.append(
                        (
                            await self._translate_chunk(
                                cell,
                                request,
                                "없음",
                                resolved_style_guide,
                                resolved_model_name,
                                llm_provider,
                                llm_call_mode,
                                llm_reasoning,
                            )
                        ).strip()
                    )
            translated_lines.append(_build_markdown_row(translated_cells))

        trailing_newline = "\n" if table_text.endswith("\n") else ""
        return "\n".join(translated_lines) + trailing_newline

    async def _translate_chunk_with_retry(
        self,
        chunk: str,
        request: TranslationRequest,
        previous_summary: str,
        preserve_mapping: Dict[str, str],
        resolved_style_guide: str,
        resolved_model_name: str,
        llm_provider: str,
        llm_call_mode: str,
        llm_reasoning: bool,
    ) -> ChunkResult:
        """청크 번역 후 품질 평가를 수행하고 필요 시 재시도한다.

        운영 관점 포인트:
        - 재시도는 비용/지연 시간을 증가시키므로 max_retries_per_chunk로 제한한다.
        - 품질 점수 개선 시에만 교체해 불필요한 출력 변동성을 줄인다.
        - preserve 복원 후 품질 비교를 수행해 실제 사용자 출력 기준으로 판단한다.
        """

        translated = await self._translate_chunk(
            chunk,
            request,
            previous_summary,
            resolved_style_guide,
            resolved_model_name,
            llm_provider,
            llm_call_mode,
            llm_reasoning,
        )
        source_chunk_display = _restore_preserved(chunk, preserve_mapping)
        translated_display = _restore_preserved(translated, preserve_mapping)
        quality = _quality_check(source_chunk_display, translated_display)
        retried = False

        if quality.issues and request.max_retries_per_chunk > 0:
            for _ in range(request.max_retries_per_chunk):
                retried_candidate = await self._translate_chunk(
                    chunk,
                    request,
                    previous_summary,
                    resolved_style_guide,
                    resolved_model_name,
                    llm_provider,
                    llm_call_mode,
                    llm_reasoning,
                )
                retried_candidate_display = _restore_preserved(retried_candidate, preserve_mapping)
                quality_candidate = _quality_check(source_chunk_display, retried_candidate_display)
                if quality_candidate.score > quality.score:
                    translated_display = retried_candidate_display
                    quality = quality_candidate
                    retried = True
                if not quality.issues:
                    break

        return ChunkResult(
            index=-1,
            source_text=source_chunk_display,
            translated_text=translated_display,
            quality_score=quality.score,
            issues=quality.issues,
            retried=retried,
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """전체 번역 파이프라인을 실행한다.

        운영 관점 포인트:
        - 보호/복원 단계로 구조 손상을 방지하고, 표는 셀 단위 처리로 가독성을 유지한다.
        - 병렬 모드에서 처리량을 높이되 컨텍스트 전략으로 품질 저하를 완화한다.
        - 최종 응답에 청크별 점수와 평균 점수를 포함해 모니터링 지표로 활용 가능하다.

        흐름:
        1) 옵션 해석(provider/model/style/mode/reasoning)
        2) 보호 대상 마스킹 및 표 처리
        3) 청크 분할
        4) 순차 또는 병렬 번역(+품질검사/재시도)
        5) 결과 병합 및 평균 품질 점수 계산
        """

        effective_request = (
            request.model_copy(update={"max_retries_per_chunk": 0, "llm_reasoning": False})
            if request.use_fidelity_first_preset
            else request
        )

        source_text = effective_request.text
        preserve_mapping: Dict[str, str] = {}
        masked_text = source_text
        resolved_style_guide = _resolve_style_guide(effective_request)
        resolved_model_name = _resolve_model_name(effective_request, self.default_model_name)
        llm_provider = effective_request.llm_provider
        llm_call_mode = effective_request.llm_call_mode
        llm_reasoning = effective_request.llm_reasoning

        if request.preserve_markdown_structures:
            masked_text, preserve_mapping = _protect_non_translatable(
                source_text,
                protect_special_entities=effective_request.preserve_special_entities,
                allowlist=effective_request.special_entity_allowlist_regex,
                denylist=effective_request.special_entity_denylist_regex,
            )

            masked_text, table_mapping, _ = _protect_markdown_tables(masked_text, len(preserve_mapping))
            for token, table_block in table_mapping.items():
                if effective_request.translate_markdown_tables:
                    preserve_mapping[token] = await self._translate_markdown_table(
                        table_block,
                        effective_request,
                        resolved_style_guide,
                        resolved_model_name,
                        llm_provider,
                        llm_call_mode,
                        llm_reasoning,
                    )
                else:
                    preserve_mapping[token] = table_block

            source_text = masked_text

        effective_overlap = _resolve_effective_chunk_overlap(effective_request)
        splitter = _build_splitter(effective_request.chunk_size, effective_overlap)
        chunks = splitter.split_text(source_text)

        chunk_results: List[ChunkResult] = []

        if effective_request.parallel_chunk_translation:
            semaphore = asyncio.Semaphore(effective_request.max_concurrency)
            context_text = _resolve_parallel_context_text(effective_request)

            async def worker(idx: int, chunk: str) -> ChunkResult:
                async with semaphore:
                    chunk_body = _strip_overlap_prefix(chunk, effective_overlap, idx == 0)
                    overlap_context = (
                        _build_overlap_context(chunks[idx - 1], effective_overlap)
                        if idx > 0
                        else ""
                    )
                    combined_context = "\n\n".join(part for part in [overlap_context, context_text] if part)
                    result = await self._translate_chunk_with_retry(
                        chunk=chunk_body,
                        request=effective_request,
                        previous_summary=combined_context,
                        preserve_mapping=preserve_mapping,
                        resolved_style_guide=resolved_style_guide,
                        resolved_model_name=resolved_model_name,
                        llm_provider=llm_provider,
                        llm_call_mode=llm_call_mode,
                        llm_reasoning=llm_reasoning,
                    )
                    result.index = idx
                    return result

            chunk_results = await asyncio.gather(
                *(worker(idx, chunk) for idx, chunk in enumerate(chunks))
            )
        else:
            for idx, chunk in enumerate(chunks):
                chunk_body = _strip_overlap_prefix(chunk, effective_overlap, idx == 0)
                overlap_context = (
                    _build_overlap_context(chunks[idx - 1], effective_overlap)
                    if idx > 0
                    else ""
                )
                result = await self._translate_chunk_with_retry(
                    chunk=chunk_body,
                    request=effective_request,
                    previous_summary=overlap_context,
                    preserve_mapping=preserve_mapping,
                    resolved_style_guide=resolved_style_guide,
                    resolved_model_name=resolved_model_name,
                    llm_provider=llm_provider,
                    llm_call_mode=llm_call_mode,
                    llm_reasoning=llm_reasoning,
                )
                result.index = idx
                chunk_results.append(result)

        merged = "\n".join(c.translated_text for c in chunk_results)
        avg_score = round(mean(c.quality_score for c in chunk_results), 3) if chunk_results else 0.0

        return TranslationResponse(
            translated_text=merged,
            chunks=chunk_results,
            average_quality_score=avg_score,
        )
