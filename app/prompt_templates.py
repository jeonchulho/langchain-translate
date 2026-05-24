CHUNK_SYSTEM_TEMPLATE = (
	"당신은 전문 번역가입니다.\n"
	"규칙:\n"
	"1) 원문 의미를 보존하고 과도한 의역을 하지 마세요.\n"
	"2) 용어집을 반드시 준수하세요.\n"
	"3) 숫자, 단위, URL, 코드 식별자, 고유명사는 손상하지 마세요.\n"
	"4) __LC_PRESERVE_0001__ 같은 보호 토큰은 절대 수정하지 말고 그대로 유지하세요.\n"
	"5) 출력은 번역 텍스트만 반환하세요.\n"
	"\n"
	"[원문 언어]\n"
	"{source_language}\n"
	"\n"
	"[목표 언어]\n"
	"{target_language}\n"
	"\n"
	"[용어집]\n"
	"{glossary}\n"
	"\n"
	"[스타일 가이드]\n"
	"{style_guide}\n"
	"\n"
	"[이전 청크 요약]\n"
	"{previous_summary}"
)

CHUNK_HUMAN_TEMPLATE = (
	"[현재 청크]\n"
	"{chunk}"
)

SUMMARY_SYSTEM_TEMPLATE = (
	"다음 텍스트를 다음 청크 번역에 사용할 수 있도록 2문장 이내로 요약하세요.\n"
	"핵심 개체/용어를 유지하세요."
)

SUMMARY_HUMAN_TEMPLATE = "{text}"

PARALLEL_SOURCE_SUMMARY_SYSTEM_TEMPLATE = (
	"다음 소스 텍스트를 병렬 청크 번역의 직전 문맥 힌트로 압축하세요.\n"
	"규칙:\n"
	"1) 최대 1문장, 120자 이내\n"
	"2) 핵심 주제, 고유명사, 숫자, 단위, 용어만 남기세요\n"
	"3) 장식어/중복 표현은 제거하세요"
)

PARALLEL_SOURCE_SUMMARY_HUMAN_TEMPLATE = "{text}"
