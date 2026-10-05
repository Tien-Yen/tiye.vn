from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sentence_transformers import SentenceTransformer

from retrieval import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    load_collection,
    load_all_chunks,
    semantic_search,
    bm25_search,
    select_document_first_evidence,
)
from timetable_engine import TimetableEngine


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_TIMEOUT = 300

EVIDENCE_TOP_K = 6
MAX_EVIDENCE_CHARS = 12000
MAX_CHARS_PER_EVIDENCE = 2000


def build_evidence(rows: list[dict]) -> tuple[str, int]:
    pieces: list[str] = []
    total_chars = 0

    for index, row in enumerate(rows[:EVIDENCE_TOP_K], start=1):
        metadata = row.get("metadata") or {}
        source = metadata.get("source", "Không rõ")

        location_parts = []
        for key in ("section", "page", "table", "sheet", "row"):
            value = metadata.get(key)
            if value not in (None, "", []):
                location_parts.append(f"{key}={value}")
        location = ", ".join(location_parts) or "Không rõ"

        text = (row.get("document") or "").strip()
        if len(text) > MAX_CHARS_PER_EVIDENCE:
            text = text[:MAX_CHARS_PER_EVIDENCE].rstrip() + " ..."

        piece = (
            f"[BẰNG CHỨNG {index}]\n"
            f"Nguồn: {source}\n"
            f"Vị trí: {location}\n"
            f"{text}\n"
        )

        if total_chars + len(piece) > MAX_EVIDENCE_CHARS:
            break

        pieces.append(piece)
        total_chars += len(piece)

    return "\n".join(pieces), total_chars


def build_system_message() -> str:
    return """/no_think

Bạn là trợ lý AI của Trường THCS&THPT Tiên Yên.

NHIỆM VỤ:
- Chỉ trả lời dựa trên BẰNG CHỨNG được cung cấp trong lượt hiện tại.
- Có thể hiểu cách diễn đạt tự nhiên và từ đồng nghĩa.
- Có thể kết hợp nhiều bằng chứng khi chúng thực sự cùng mô tả thông tin cần trả lời.
- Không sử dụng kiến thức bên ngoài tài liệu của nhà trường.

QUY TẮC CHÍNH XÁC:
- Không bịa số liệu, ngày tháng, thời gian, tên người, lớp, môn học hoặc quy định.
- Không lấy thông tin của đối tượng này gán cho đối tượng khác.
- Khi hỏi một thông tin cụ thể (ví dụ hạn đăng ký, ngày, giờ, số lượng), ưu tiên bằng chứng trực tiếp chứa đúng thông tin đó; không lấy một con số/ngày chỉ vì nó xuất hiện trong cùng tài liệu.
- Một giá trị chỉ được dùng khi bằng chứng gắn rõ giá trị đó với đúng đối tượng và đúng nội dung câu hỏi.
- Khi nhiều bằng chứng có giá trị khác nhau, không chọn một giá trị chỉ vì nó xuất hiện trong evidence; phải ưu tiên evidence trực tiếp nói về chính nội dung được hỏi.
- Không dùng một ngày/thời gian của hoạt động khác để trả lời câu hỏi hiện tại.
- Không biến suy đoán thành sự thật.
- Nếu bằng chứng chỉ xác nhận được một phần, trả lời phần xác nhận được và nói rõ phần còn thiếu.
- Nếu bằng chứng không đủ, nói rõ rằng chưa có đủ căn cứ để xác nhận.

CÁCH TRẢ LỜI:
- Tiếng Việt.
- Ngắn gọn, trực tiếp, dễ hiểu với học sinh.
- Không nhắc đến embedding, RAG, BM25, evidence hoặc cấu trúc kỹ thuật nội bộ.
"""


def clean_qwen_output(content: str) -> str:
    text = (content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[0].strip()
    return text.strip()


def ask_qwen(question: str, evidence: str, chat_messages: list[dict]) -> str:
    current_message = f"""CÂU HỎI CỦA HỌC SINH:
{question}

BẰNG CHỨNG CỦA LƯỢT HIỆN TẠI:
{evidence}

Chỉ trả lời bằng Tiếng Việt, tối đa 1-2 câu. Không trình bày phân tích hoặc quá trình suy luận."""

    messages = [
        {"role": "system", "content": build_system_message()},
        *chat_messages,
        {"role": "user", "content": current_message},
    ]

    answer_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "maxLength": 300},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": answer_schema,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "num_predict": 4096,
        },
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Không kết nối được Ollama tại http://127.0.0.1:11434"
        ) from exc

    content = ((result.get("message") or {}).get("content") or "").strip()
    if content:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                answer = data.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return clean_qwen_output(answer)
        except json.JSONDecodeError:
            pass
    return clean_qwen_output(content)


def print_retrieval(rows: list[dict]) -> None:
    print("\nTOP EVIDENCE ĐƯỢC CHỌN:")
    for index, row in enumerate(rows[:EVIDENCE_TOP_K], start=1):
        metadata = row.get("metadata") or {}
        source = metadata.get("source", "")
        location_parts = []
        for key in ("section", "page", "table", "sheet", "row"):
            value = metadata.get(key)
            if value not in (None, "", []):
                location_parts.append(f"{key}={value}")
        location = ", ".join(location_parts) or "Không rõ"
        print(
            f"  {index}. {source} | {location} | "
            f"Final={row.get('final_score', row.get('rrf_score', 0.0)):.6f} | "
            f"Global RRF={row.get('global_rrf_score', 0.0):.6f} | "
            f"Document={row.get('document_rank')} | "
            f"Semantic local={row.get('semantic_rank_local')} / global={row.get('semantic_rank_global')} | "
            f"BM25 local={row.get('lexical_rank_local')} / global={row.get('lexical_rank_global')}"
        )


def retrieve_prose(question: str, collection, all_chunks: list[dict], model) -> list[dict]:
    semantic_rows = semantic_search(
        question,
        collection,
        model,
        file_types={"pdf", "docx"},
    )
    lexical_rows = bm25_search(
        question,
        all_chunks,
        file_types={"pdf", "docx"},
    )
    return select_document_first_evidence(
        question=question,
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        all_chunks=all_chunks,
        model=model,
        collection=collection,
    )


def main() -> None:
    print("=" * 60)
    print("SCHOOL AI - RAG + STRUCTURED TKB")
    print("PDF/DOCX -> Retrieval -> Qwen")
    print("XLSX/TKB -> Python xử lý trực tiếp")
    print("=" * 60)

    print("Đang tải embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model đã sẵn sàng.")

    collection = load_collection()
    all_chunks = load_all_chunks(collection)
    timetable = TimetableEngine()

    print(f"Chroma collection: {COLLECTION_NAME} | {len(all_chunks)} chunks")
    print(f"TKB records: {len(timetable.lessons)} tiết")
    print(f"Qwen model: {OLLAMA_MODEL}")
    print(f"Evidence gửi Qwen: Top {EVIDENCE_TOP_K}")
    print("Lệnh: 'moi' để reset hội thoại/context | 'thoat' để thoát")
    print()

    chat_messages: list[dict] = []

    while True:
        question = input("Học sinh hỏi: ").strip()
        if not question:
            continue

        command = question.lower()
        if command == "thoat":
            print("Đã thoát.")
            break
        if command == "moi":
            chat_messages.clear()
            timetable.context.clear()
            print("Đã reset ngữ cảnh.\n")
            continue

        try:
            # TKB đi hoàn toàn bằng Python.
            if timetable.is_timetable_query(question):
                result = timetable.answer(question)
                print("\nTrợ lý:")
                print(result["message"])
                print()
                continue

            # PDF/DOCX đi qua hybrid retrieval rồi mới tới Qwen.
            rows = retrieve_prose(question, collection, all_chunks, model)
            if not rows:
                print("\nTrợ lý:")
                print("Không tìm thấy bằng chứng phù hợp cho câu hỏi này.")
                print()
                continue

            print_retrieval(rows)
            evidence, evidence_chars = build_evidence(rows)
            print(f"Độ dài evidence gửi Qwen: {evidence_chars:,} ký tự")
            print("Đang để Qwen đọc và trả lời...")

            answer = ask_qwen(question, evidence, chat_messages)
            if not answer:
                answer = "Qwen không trả về nội dung."

            current_message = (
                f"CÂU HỎI CỦA HỌC SINH:\n{question}\n\n"
                f"BẰNG CHỨNG CỦA LƯỢT HIỆN TẠI:\n{evidence}"
            )
            chat_messages.append({"role": "user", "content": current_message})
            chat_messages.append({"role": "assistant", "content": answer})

            print("\nTrợ lý:")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nĐã dừng.")
            break
        except Exception as exc:
            print(f"\nLỖI: {exc}")


if __name__ == "__main__":
    main() 
