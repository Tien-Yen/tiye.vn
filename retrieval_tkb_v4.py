import re
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# SCHOOL AI - BƯỚC 4: HYBRID RETRIEVAL
# Semantic + BM25 + Reciprocal Rank Fusion (RRF)
# Chưa gọi Qwen / chưa trả lời câu hỏi
# ============================================================

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "school_documents"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_TOP_K = 30
LEXICAL_TOP_K = 30
FINAL_TOP_K = 12
RRF_K = 60

# Một số từ rất phổ biến, ít giúp phân biệt tài liệu.
# Đây là stopword tổng quát, không phụ thuộc vào câu hỏi cụ thể.
STOPWORDS = {
    "la", "là", "cua", "của", "truong", "trường", "hoc", "học",
    "sinh", "sinh", "bao", "bao nhiêu", "nhung", "những", "nao", "nào",
    "gi", "gì", "ai", "the", "thế", "nao", "nào", "duoc", "được",
    "cho", "trong", "tai", "tại", "ve", "về", "co", "có", "khong", "không",
    "la", "một", "mot", "cac", "các", "moi", "mỗi", "nay", "này",
    "khi", "khi", "bao", "nhiêu", "dien", "diễn", "ra", "theo", "doi",
    "đời", "thuoc", "thuộc", "nam", "năm"
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^\w\sÀ-ỹđĐ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = normalized.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]



DAY_NAMES = {
    "hai": "2", "ba": "3", "tư": "4", "tu": "4", "bốn": "4", "bon": "4",
    "năm": "5", "nam": "5", "sáu": "6", "sau": "6",
    "bảy": "7", "bay": "7",
}


def _find_days(text: str) -> list[str]:
    q = normalize_text(text)
    found: list[str] = []

    # Range: "thứ 2 đến thứ 7" / "thứ hai tới thứ bảy".
    range_match = re.search(
        r"\bthứ\s*(hai|ba|tư|tu|bốn|bon|năm|nam|sáu|sau|bảy|bay|2|3|4|5|6|7)"
        r"\s*(?:đến|tới|->|-)\s*"
        r"(?:thứ\s*)?(hai|ba|tư|tu|bốn|bon|năm|nam|sáu|sau|bảy|bay|2|3|4|5|6|7)\b",
        q,
    )
    if range_match:
        def to_num(value: str) -> int:
            return int(value) if value.isdigit() else int(DAY_NAMES[value])
        start_day = to_num(range_match.group(1))
        end_day = to_num(range_match.group(2))
        if start_day <= end_day:
            return [str(n) for n in range(start_day, end_day + 1)]

    patterns = [
        (r"\bthứ\s*(hai|2)\b", "2"),
        (r"\bthứ\s*(ba|3)\b", "3"),
        (r"\bthứ\s*(tư|4)\b", "4"),
        (r"\bthứ\s*(năm|5)\b", "5"),
        (r"\bthứ\s*(sáu|6)\b", "6"),
        (r"\bthứ\s*(bảy|7)\b", "7"),
    ]
    for pattern, value in patterns:
        if re.search(pattern, q):
            found.append(value)

    # Bỏ trùng nhưng giữ thứ tự.
    return list(dict.fromkeys(found))


def _find_periods(text: str) -> list[str]:
    q = normalize_text(text)
    periods: list[str] = []

    # "tiết 4 và 5", "tiết 2,3", "tiết 1-5".
    for group in re.findall(
        r"\btiết\s*(\d+(?:\s*(?:,|và|&|-|đến)\s*\d+)*)",
        q,
    ):
        numbers = [int(x) for x in re.findall(r"\d+", group)]
        # Nếu có từ "đến" hoặc dấu gạch nối giữa hai số -> mở rộng dãy.
        if len(numbers) == 2 and re.search(r"(?:-|đến)", group):
            numbers = list(range(numbers[0], numbers[1] + 1))
        for number in numbers:
            value = str(number)
            if value not in periods:
                periods.append(value)

    return periods


def parse_timetable_query(question: str) -> dict:
    """Phân tích truy vấn TKB thành các điều kiện cấu trúc, không phụ thuộc embedding."""
    q = normalize_text(question)

    info = {
        "class_name": None,
        "grade": None,
        "days": [],
        "day": None,
        "session": None,
        "periods": [],
        "period_count": None,
    }

    class_match = re.search(r"\b(6|7|8|9|10|11|12)a\d+\b", q, re.IGNORECASE)
    if class_match:
        info["class_name"] = class_match.group(0).upper()
        grade_match = re.match(r"(6|7|8|9|10|11|12)", info["class_name"])
        if grade_match:
            info["grade"] = grade_match.group(1)
    else:
        grade_match = re.search(r"\bkhối\s*(6|7|8|9|10|11|12)\b", q, re.IGNORECASE)
        if grade_match:
            info["grade"] = grade_match.group(1)

    info["days"] = _find_days(q)
    if len(info["days"]) == 1:
        info["day"] = info["days"][0]

    if re.search(r"\bb[uù]ổi\s+s[aá]ng\b|\bsáng\b", q, re.IGNORECASE):
        info["session"] = "S"
    elif re.search(r"\bb[uù]ổi\s+chi[eề]u\b|\bchi[eề]u\b", q, re.IGNORECASE):
        info["session"] = "C"

    info["periods"] = _find_periods(q)

    # "5 tiết buổi sáng" = muốn các tiết 1..5, không phải chỉ tiết 5.
    count_match = re.search(
        r"\b(?:gồm|có|có|đủ|trong)\s*(\d+)\s*tiết\b",
        q,
    )
    if count_match and not info["periods"]:
        count = int(count_match.group(1))
        if 1 <= count <= 10:
            info["period_count"] = count
            info["periods"] = [str(n) for n in range(1, count + 1)]

    return info


def _header_value(document: str, field: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(field)}\s*:\s*([^\n]+)",
        document,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _document_periods(document: str) -> list[str]:
    periods = []
    for raw in re.findall(r"^\s*Tiết\s*(\d+)\s*:", document, re.MULTILINE | re.IGNORECASE):
        if raw not in periods:
            periods.append(raw)
    return periods


def _document_class(document: str) -> str:
    return _header_value(document, "Lớp").upper()


def _document_grade(document: str) -> str:
    grade = _header_value(document, "Khối")
    if grade:
        match = re.search(r"\d+", grade)
        if match:
            return match.group(0)
    class_name = _document_class(document)
    match = re.match(r"(6|7|8|9|10|11|12)A", class_name, re.IGNORECASE)
    return match.group(1) if match else ""


def structured_timetable_search(question: str, all_chunks: list[dict]) -> list[dict]:
    """
    Tìm trên chunk TKB mới: một chunk = một lớp + một ngày + một buổi.
    Vì vậy truy vấn tiết 1 không làm mất các tiết 2..5 của cùng buổi.
    """
    info = parse_timetable_query(question)
    has_signal = bool(
        info["class_name"]
        or info["grade"]
        or info["days"]
        or info["session"]
        or info["periods"]
    )
    if not has_signal:
        return []

    candidates = []
    wanted_days = set(info["days"])
    wanted_periods = set(info["periods"])

    for row in all_chunks:
        metadata = row.get("metadata") or {}
        if metadata.get("file_type") != "xlsx":
            continue

        document = row.get("document", "")
        class_name = _document_class(document)
        grade = _document_grade(document)
        day_text = _header_value(document, "Thứ")
        session_text = _header_value(document, "Buổi")
        periods = _document_periods(document)

        day_match = re.search(r"\((2|3|4|5|6|7)\)", day_text)
        day = day_match.group(1) if day_match else ""
        session = ""
        if re.search(r"sáng|\(s\)", session_text, re.IGNORECASE):
            session = "S"
        elif re.search(r"chiều|\(c\)", session_text, re.IGNORECASE):
            session = "C"

        if info["class_name"] and class_name != info["class_name"].upper():
            continue
        if info["grade"] and grade != info["grade"]:
            continue
        if wanted_days and day not in wanted_days:
            continue
        if info["session"] and session != info["session"]:
            continue
        if wanted_periods and not (wanted_periods & set(periods)):
            continue

        score = 0
        score += 6 if info["class_name"] and class_name == info["class_name"].upper() else 0
        score += 3 if info["grade"] and grade == info["grade"] else 0
        score += 3 if day in wanted_days else 0
        score += 3 if info["session"] and session == info["session"] else 0
        score += 2 if wanted_periods and (wanted_periods & set(periods)) else 0

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": None,
            "bm25_score": None,
            "structured_score": score,
            "query_info": info,
            "_day": int(day) if day.isdigit() else 99,
            "_session": 0 if session == "S" else 1,
            "_class": class_name,
        })

    candidates.sort(
        key=lambda item: (
            item["_day"],
            item["_session"],
            item["_class"],
            item["metadata"].get("chunk_index", 0),
        )
    )

    result = []
    for rank, row in enumerate(candidates, start=1):
        item = dict(row)
        item["rank"] = rank
        item["rrf_score"] = 1.0 / (RRF_K + rank)
        item.pop("_day", None)
        item.pop("_session", None)
        item.pop("_class", None)
        result.append(item)
    return result

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return collection


def load_all_chunks(collection):
    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows = []
    for i, (document, metadata) in enumerate(zip(documents, metadatas)):
        rows.append({
            "index": i,
            "id": (result.get("ids") or [None] * len(documents))[i],
            "document": document or "",
            "metadata": metadata or {},
        })
    return rows


def semantic_search(question: str, collection, model):
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(SEMANTIC_TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        rows.append({
            "document": document or "",
            "metadata": metadata or {},
            "distance": float(distance),
            "rank": rank,
        })
    return rows


def bm25_search(question: str, all_chunks):
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    tokenized_docs = []
    document_frequency = Counter()

    for row in all_chunks:
        tokens = tokenize(row["document"])
        tokenized_docs.append(tokens)
        for token in set(tokens):
            document_frequency[token] += 1

    n_docs = len(all_chunks)
    avgdl = (
        sum(len(tokens) for tokens in tokenized_docs) / n_docs
        if n_docs else 0.0
    )

    k1 = 1.5
    b = 0.75

    scored = []
    query_counter = Counter(query_tokens)

    for row, tokens in zip(all_chunks, tokenized_docs):
        if not tokens:
            continue

        tf = Counter(tokens)
        doc_length = len(tokens)
        score = 0.0

        for term in query_counter:
            df = document_frequency.get(term, 0)
            if df == 0:
                continue

            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue

            numerator = term_tf * (k1 + 1)
            denominator = (
                term_tf
                + k1 * (1 - b + b * (doc_length / avgdl if avgdl else 1))
            )
            score += idf * (numerator / denominator)

        # Thưởng nhẹ cho cụm từ nhiều từ xuất hiện liền nhau.
        # Đây vẫn là lexical matching tổng quát, không biết nội dung câu hỏi.
        normalized_doc = normalize_text(row["document"])
        normalized_question = normalize_text(question)
        question_terms = [t for t in query_tokens if len(t) > 2]
        if len(question_terms) >= 2:
            phrase = " ".join(question_terms)
            if phrase and phrase in normalized_doc:
                score += 1.0

        # Chỉ giữ các chunk thực sự có lexical match.
        # Chunk BM25 = 0 không mang bằng chứng từ từ khóa của câu hỏi,
        # nên không được đưa vào lexical ranking/RRF.
        if score <= 0:
            continue

        scored.append({
            "document": row["document"],
            "metadata": row["metadata"],
            "distance": None,
            "bm25_score": score,
        })

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return [dict(row, rank=rank) for rank, row in enumerate(scored[:LEXICAL_TOP_K], start=1)]


def key_for(row):
    metadata = row.get("metadata") or {}
    document = row.get("document", "")

    # IDs do not need to match between the two retrieval paths.
    # source + location + text is stable enough within this KB.
    return (
        metadata.get("source", ""),
        metadata.get("block_id", ""),
        metadata.get("chunk_id", ""),
        document,
    )


def rrf_fusion(semantic_rows, lexical_rows, structured_rows=None):
    fused = defaultdict(lambda: {
        "rrf_score": 0.0,
        "semantic_rank": None,
        "lexical_rank": None,
        "distance": None,
        "bm25_score": None,
        "document": "",
        "metadata": {},
    })

    for row in semantic_rows:
        key = key_for(row)
        item = fused[key]
        item["rrf_score"] += 1.0 / (RRF_K + row["rank"])
        item["semantic_rank"] = row["rank"]
        item["distance"] = row["distance"]
        item["document"] = row["document"]
        item["metadata"] = row["metadata"]

    for row in lexical_rows:
        key = key_for(row)
        item = fused[key]
        item["rrf_score"] += 1.0 / (RRF_K + row["rank"])
        item["lexical_rank"] = row["rank"]
        item["bm25_score"] = row["bm25_score"]
        item["document"] = row["document"]
        item["metadata"] = row["metadata"]

    # Structured retrieval được cộng điểm riêng khi câu hỏi có các
    # điều kiện rõ ràng của thời khóa biểu. Đây là cơ chế tổng quát,
    # không phụ thuộc vào nội dung của một câu hỏi cụ thể.
    if structured_rows:
        for row in structured_rows:
            key = key_for(row)
            item = fused[key]
            item["rrf_score"] += 2.0 / (20 + row["rank"])
            item["structured_rank"] = row["rank"]
            item["structured_score"] = row.get("structured_score", 0.0)
            item["query_info"] = row.get("query_info", {})
            if not item["document"]:
                item["document"] = row["document"]
                item["metadata"] = row["metadata"]

    rows = list(fused.values())
    rows.sort(key=lambda x: x["rrf_score"], reverse=True)
    return rows[:FINAL_TOP_K]


def print_row(rank, row):
    metadata = row.get("metadata") or {}
    source = metadata.get("source", "")
    location = metadata.get("location", "")
    if isinstance(location, dict):
        location = ", ".join(f"{k}={v}" for k, v in location.items())

    print(f"[{rank}] RRF={row['rrf_score']:.6f}")
    print(f"Nguồn: {source}")
    print(f"Vị trí: {location}")
    print(f"Semantic rank  : {row.get('semantic_rank')}")
    print(f"Lexical rank   : {row.get('lexical_rank')}")
    print(f"Structured rank: {row.get('structured_rank')}")
    if row.get("distance") is not None:
        print(f"Distance     : {row['distance']:.6f}")
    if row.get("bm25_score") is not None:
        print(f"BM25         : {row['bm25_score']:.6f}")
    print(row["document"][:1200])
    print("-" * 60)


def main():
    print("=" * 60)
    print("SCHOOL AI - BƯỚC 4: HYBRID RETRIEVAL")
    print("Semantic + BM25 + RRF")
    print("Chưa gọi Qwen / chưa trả lời câu hỏi")
    print("=" * 60)

    print("Đang tải embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model đã sẵn sàng.")

    collection = load_collection()
    all_chunks = load_all_chunks(collection)

    print(f"Chroma collection: {COLLECTION_NAME} | {len(all_chunks)} chunks")
    print(
        f"SEMANTIC_TOP_K={SEMANTIC_TOP_K} | "
        f"LEXICAL_TOP_K={LEXICAL_TOP_K} | "
        f"FINAL_TOP_K={FINAL_TOP_K}"
    )
    print()

    while True:
        question = input("Học sinh hỏi: ").strip()
        if question.lower() == "thoat":
            print("Đã thoát.")
            break
        if not question:
            continue

        print("\n" + "-" * 60)
        print("SEMANTIC SEARCH")
        print("-" * 60)
        semantic_rows = semantic_search(question, collection, model)
        for i, row in enumerate(semantic_rows[:5], start=1):
            print(
                f"Semantic [{i}] distance={row['distance']:.6f} | "
                f"{row['metadata'].get('source', '')} | "
                f"{row['metadata'].get('location', '')}"
            )

        print("\n" + "-" * 60)
        print("LEXICAL SEARCH (BM25)")
        print("-" * 60)
        lexical_rows = bm25_search(question, all_chunks)
        for i, row in enumerate(lexical_rows[:5], start=1):
            print(
                f"BM25 [{i}] score={row['bm25_score']:.6f} | "
                f"{row['metadata'].get('source', '')} | "
                f"{row['metadata'].get('location', '')}"
            )

        print("\n" + "-" * 60)
        print("STRUCTURED TIMETABLE SEARCH")
        print("-" * 60)
        structured_rows = structured_timetable_search(question, all_chunks)
        if structured_rows:
            info = structured_rows[0].get("query_info", {})
            print(f"Điều kiện: {info}")
            for i, row in enumerate(structured_rows[:5], start=1):
                print(
                    f"Structured [{i}] score={row.get('structured_score', 0):.1f} | "
                    f"{row['metadata'].get('source', '')} | "
                    f"{row['metadata'].get('location', '')}"
                )
        else:
            print("Không phát hiện truy vấn thời khóa biểu có cấu trúc.")

        # Nếu đã xác định được truy vấn TKB có cấu trúc, KHÔNG trộn
        # semantic/BM25 vào nữa. Structured retrieval là nguồn dữ liệu
        # chính xác theo khóa lớp + ngày + buổi + tiết.
        if structured_rows:
            final_rows = structured_rows
            print("\nĐÃ XÁC ĐỊNH TRUY VẤN TKB CÓ CẤU TRÚC")
            print("Semantic/BM25 chỉ tham khảo, không được trộn vào kết quả TKB.")
        else:
            final_rows = rrf_fusion(semantic_rows, lexical_rows, None)

        print("\n" + "=" * 60)
        if structured_rows:
            print("TOP KẾT QUẢ STRUCTURED TKB")
        else:
            print("TOP EVIDENCE SAU KHI RRF")
        print("=" * 60)
        for rank, row in enumerate(final_rows, start=1):
            print_row(rank, row)

        print("=" * 60)
        print("TỔNG HỢP THEO FILE")
        print("=" * 60)
        file_groups = defaultdict(list)
        for row in final_rows:
            source = row["metadata"].get("source", "")
            file_groups[source].append(row)

        for source, rows in sorted(
            file_groups.items(),
            key=lambda item: item[1][0]["rrf_score"],
            reverse=True,
        ):
            print(
                f"- {source} | "
                f"{len(rows)} evidence trong TOP {len(final_rows)} | "
                f"RRF tốt nhất={rows[0]['rrf_score']:.6f}"
            )

        print()


if __name__ == "__main__":
    main()
