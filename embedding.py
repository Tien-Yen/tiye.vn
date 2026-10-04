from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/processed/chunks.jsonl")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "school_documents"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 16


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Không tìm thấy {INPUT_PATH}. Hãy chạy: python chunking.py"
        )

    chunks = load_jsonl(INPUT_PATH)
    if not chunks:
        raise SystemExit("chunks.jsonl đang rỗng.")

    print("=" * 60)
    print("SCHOOL AI - BƯỚC 3: EMBEDDING + CHROMADB")
    print("Chưa gọi Qwen / chưa thực hiện retrieval")
    print("=" * 60)
    print(f"Số chunk đầu vào: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    print("\nĐang tải embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model đã sẵn sàng.")

    # Embed bản có thêm tên file/section để retrieval biết chunk thuộc tài liệu nào.
    # Nội dung gốc `text` vẫn được lưu vào Chroma để Qwen đọc không bị thêm metadata.
    texts = [chunk.get("retrieval_text") or chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    print(f"\nĐang tạo embedding cho {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    print("\nKết nối ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Đã xóa collection cũ.")
    except Exception:
        print("Không có collection cũ để xóa.")

    # Dùng cosine distance để khoảng cách có ý nghĩa rõ ràng hơn cho embedding.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    metadatas = []
    for chunk in chunks:
        location = chunk.get("location", {}) or {}
        metadatas.append(
            {
                "file_id": str(chunk.get("file_id", "")),
                "source": str(chunk.get("source", "")),
                "relative_path": str(chunk.get("relative_path", "")),
                "file_type": str(chunk.get("file_type", "")),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "block_types": ",".join(chunk.get("block_types", [])),
                "section": str(location.get("section", "")),
                "page": str(location.get("page", "")),
                "table": str(location.get("table", "")),
                "sheet": str(location.get("sheet", "")),
                "row": str(location.get("row", "")),
                "char_count": int(chunk.get("char_count", len(chunk.get("text", "")))),
            }
        )

    # Chroma có giới hạn kích thước batch ở một số phiên bản.
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        print(f"Đã lưu {end}/{len(chunks)} chunks...")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH BƯỚC 3")
    print("=" * 60)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Chunks     : {collection.count()}")
    print(f"Database   : {CHROMA_PATH}")
    print("Metric     : cosine")

    # Kiểm tra tối thiểu: dữ liệu gốc quan trọng vẫn tồn tại trong Chroma.
    result = collection.get(
        where={"file_type": "docx"},
        include=["documents"],
    )
    docs = result.get("documents", [])
    checks = {
        "Trương Thị Thanh Châm": any("Trương Thị Thanh Châm" in d for d in docs),
        "HIỆU TRƯỞNG": any("HIỆU TRƯỞNG" in d for d in docs),
        "Tien Yen’s Got Talent": any("Tien Yen’s Got Talent" in d for d in docs),
    }

    print("\nKiểm tra dữ liệu trong Chroma:")
    for key, ok in checks.items():
        print(f"  '{key}': {'CÓ' if ok else 'KHÔNG'}")


if __name__ == "__main__":
    main()
