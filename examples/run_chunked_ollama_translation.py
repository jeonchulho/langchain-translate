import json
from pathlib import Path
from urllib import request

SRC_PATH = Path("examples/rotary_constitution_ko.txt")
OUT_PATH = Path("examples/rotary_constitution_en_output_real_qwen25_3b_chunked.txt")
MODEL = "qwen2.5:3b"
CHUNK_SIZE = 1800


def call_ollama(prompt: str) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1400,
            },
        }
    ).encode("utf-8")

    req = request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data.get("response", "").strip()


def main() -> None:
    src = SRC_PATH.read_text(encoding="utf-8")
    chunks = [src[i : i + CHUNK_SIZE] for i in range(0, len(src), CHUNK_SIZE)]

    translated_parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            "Translate the following Korean text into clear natural English. "
            "Output only the English translation for this chunk.\n\n"
            + chunk
        )
        translated = call_ollama(prompt)
        translated_parts.append(translated)
        print(f"chunk {idx}/{len(chunks)} done, chars={len(translated)}", flush=True)

    final_text = "\n\n".join(translated_parts)
    OUT_PATH.write_text(final_text, encoding="utf-8")

    print("CHUNKED_LONG_TRANSLATION_OK")
    print(f"output={OUT_PATH}")
    print(f"chars={len(final_text)}")


if __name__ == "__main__":
    main()
