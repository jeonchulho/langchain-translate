# langchain-translate

LangChain 기반 긴 문서 번역 MVP입니다. 문서를 청크 단위로 번역하고, 청크별 품질 점수를 계산해 필요 시 재시도합니다.

## Features

- 긴 문서 청킹 번역 (문단/문장 우선 분할)
- 이전 청크 요약 기반 문맥 전달
- 옵션 기반 청크 병렬 번역(asyncio.gather + 동시성 제한)
	- parallel_chunk_translation=true 일 때 max_concurrency 만큼 병렬 처리
	- parallel_context_strategy=source-summary 일 때 인접 청크 소스 요약을 컨텍스트로 사용
	- source-summary는 엔티티/수치 중심의 짧은 요약(최대 1문장)으로 생성
	- parallel_summary_max_chars로 source-summary 길이를 조절 가능(기본 120)
	- use_document_type_summary_preset=true 이면 document_type별 preset 자동 적용
		- technical=140, legal=220, marketing=100, general=120
- 문서 타입 기반 스타일 프리셋
	- use_document_type_style_preset=true 이면 document_type 스타일 가이드 자동 주입
		- technical: 용어/절차 일관성, legal: 조항/조건 표현 보존, marketing: 설득력 우선
- 문서 타입 기반 모델 프리셋
	- use_document_type_model_preset=true 이면 document_type별 모델 환경변수에서 자동 선택
		- OPENAI_MODEL_GENERAL, OPENAI_MODEL_TECHNICAL, OPENAI_MODEL_LEGAL, OPENAI_MODEL_MARKETING
- LLM 실행 모드 선택
	- llm_call_mode: "ainvoke" | "astream"
- LLM provider 선택
	- llm_provider: "openai" | "anthropic" | "google" | "ollama"
- LLM thinking 모드
	- llm_reasoning provider별 동작
		- openai: model_kwargs.extra_body.chat_template_kwargs.enable_thinking 전달
		- anthropic: thinking 파라미터를 시도(미지원 SDK 버전이면 자동 폴백)
		- google: thinking_budget 파라미터를 시도(미지원 SDK 버전이면 자동 폴백)
		- ollama: reasoning 파라미터를 시도(모델/버전 미지원 시 자동 폴백)
- 용어집 강제 반영
- 마크다운 구조 보존
	- 코드펜스(```...```), 인라인 코드(`...`), 표를 보호 후 복원
	- 표는 셀 단위 번역으로 구조를 유지하면서 내용만 번역
- 특수 엔티티 보호
	- URL, 이메일, 파일 경로, 코드 식별자를 보호 후 복원
	- allowlist/denylist 정규식으로 보호 대상을 세밀하게 제어
- 청크별 품질 검사
	- 숫자 불일치 감지
	- 번역 길이 비율 이상 감지
	- 미번역 가능성 감지
- 품질 저하 시 청크 단위 재시도

## Quick Start

### 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 환경 변수

```bash
cp .env.example .env
```

`.env` 파일의 `OPENAI_API_KEY`를 설정하세요.

Ollama 사용 시에는 로컬 서버와 모델을 준비하세요.

```bash
ollama serve
ollama pull llama3.1:8b
```

### 3) 서버 실행

```bash
python -m app.main
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다.

## API

### Health

```bash
curl http://localhost:8000/health
```

### Translate

```bash
curl -X POST http://localhost:8000/translate \
	-H "Content-Type: application/json" \
	-d '{
		"text": "Chapter 1\nThis platform processes 12,500 documents per day.",
		"source_language": "en",
		"target_language": "ko",
		"glossary": {
			"platform": "플랫폼",
			"documents": "문서"
		},
		"style_guide": "기술 문서 톤, 간결하고 정확하게 번역",
		"chunk_size": 3500,
		"chunk_overlap": 350,
		"max_retries_per_chunk": 1,
		"parallel_chunk_translation": false,
		"max_concurrency": 4,
		"parallel_context_strategy": "none",
		"parallel_summary_max_chars": 120,
		"document_type": "general",
		"use_document_type_summary_preset": false,
		"use_document_type_style_preset": false,
		"llm_model_override": null,
		"llm_provider": "openai",
		"llm_call_mode": "ainvoke",
		"llm_reasoning": false,
		"use_document_type_model_preset": false,
		"preserve_markdown_structures": true,
		"translate_markdown_tables": true,
		"preserve_special_entities": true,
		"special_entity_allowlist_regex": ["^https?://", "_"],
		"special_entity_denylist_regex": ["example\\.com"]
	}'
```

응답에는 최종 번역문, 청크별 번역 결과, 품질 점수가 포함됩니다.

## File Translation Example

긴 문서를 텍스트 파일로 저장한 뒤, 파일을 읽어 번역 결과 파일로 저장하는 예제입니다.

입력 파일:
- `examples/rotary_constitution_ko.txt`

실행 스크립트:
- `examples/translate_file_example.py`

출력 파일:
- 데모 모드: `examples/rotary_constitution_en_output_demo_<provider>.txt`
- 실제 모드: `examples/rotary_constitution_en_output_real_<provider>.txt`

### 데모 모드(오프라인)

실제 LLM API 호출 없이 파이프라인 동작(파일 읽기 -> 청킹/처리 -> 파일 저장)만 확인합니다.
데모 모드는 실제 번역이 아니라 `[DEMO-EN]` 접두어를 붙이는 샘플 출력입니다.

```bash
python examples/translate_file_example.py
```

### 실제 LLM 모드

실제 provider API를 호출해 번역합니다. (API 키/모델 접근 권한 필요)

```bash
python examples/translate_file_example.py --real-llm
```

### Ollama로 실제 테스트

```bash
python examples/translate_file_example.py --real-llm --provider ollama --model llama3.1:8b
```

## Structure

```text
app/
	main.py        # FastAPI 엔드포인트
	models.py      # 요청/응답 스키마
	translator.py  # 청킹/번역/품질검사 핵심 로직
examples/
	sample_input.md
	rotary_constitution_ko.txt
	translate_file_example.py
	rotary_constitution_en_output.txt
tests/
	test_preserve_rules.py
requirements.txt
.env.example
```

## Test

```bash
pytest -q
```

## Notes

- 현재 MVP는 OpenAI/Anthropic/Google/Ollama provider를 지원합니다.
- 정확도 개선을 위해 추후 다음을 권장합니다.
	- 표/코드 블록 전용 파서 추가
	- 도메인별 용어집 저장소 및 검색(RAG) 연동
	- 역번역 기반 자동 평가 지표 추가