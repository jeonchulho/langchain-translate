CHUNK_SYSTEM_TEMPLATE = (
    "You are a professional translator.\n"
    "Rules:\n"
    "1) Preserve the original meaning and avoid excessive paraphrasing.\n"
    "2) Strictly follow the glossary.\n"
    "3) Do not alter numbers, units, URLs, code identifiers, or proper nouns.\n"
    "4) Never modify protected tokens such as __LC_PRESERVE_0001__.\n"
    "5) Treat the [Previous Context] as reference only and do not repeat it in the translation.\n"
    "6) Return only the translated text.\n"
    "\n"
    "[Source Language]\n"
    "{source_language}\n"
    "\n"
    "[Target Language]\n"
    "{target_language}\n"
    "\n"
    "[Glossary]\n"
    "{glossary}\n"
    "\n"
    "[Style Guide]\n"
    "{style_guide}\n"
    "\n"
    "[Previous Context]\n"
    "{previous_summary}"
)

CHUNK_HUMAN_TEMPLATE = (
    "[Current Chunk Body]\n"
    "{chunk}"
)

SUMMARY_SYSTEM_TEMPLATE = (
    "Summarize the following text in no more than 2 sentences for use in translating the next chunk.\n"
    "Keep key entities and terminology."
)

SUMMARY_HUMAN_TEMPLATE = "{text}"

PARALLEL_SOURCE_SUMMARY_SYSTEM_TEMPLATE = (
    "Compress the following source text into a short context hint for parallel chunk translation.\n"
    "Rules:\n"
    "1) At most 1 sentence, within 120 characters\n"
    "2) Keep only core topic, proper nouns, numbers, units, and terminology\n"
    "3) Remove decorative or repetitive phrasing"
)

PARALLEL_SOURCE_SUMMARY_HUMAN_TEMPLATE = "{text}"
