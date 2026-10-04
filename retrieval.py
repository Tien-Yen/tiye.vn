import re
import math
import unicodedata

import numpy as np
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
SEMANTIC_TOP_K = 100
LEXICAL_TOP_K = 100
FINAL_TOP_K = 12
RRF_K = 60
DOCUMENT_TOP_K = 5
MAX_CHUNKS_PER_DOCUMENT = 4
SECONDARY_DOCUMENT_MAX_GLOBAL_RANK = 5
LEXICAL_RRF_WEIGHT = 1.5
SEMANTIC_RRF_WEIGHT = 1.0
AGREEMENT_RRF_WEIGHT = 0.25

# Pipeline prose mặc định chỉ làm việc với PDF/DOCX.
# XLSX/TKB đi theo structured retrieval riêng.
PROSE_FILE_TYPES = {"pdf", "docx"}

# BM25 index được cache theo identity của all_chunks trong suốt phiên chạy.
# Corpus chỉ được xây một lần khi chương trình khởi động, không xây lại sau mỗi query.
_BM25_CACHE = {}

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


def parse_timetable_query(question: str) -> dict:
    """Phân tích các điều kiện cấu trúc chung cho câu hỏi thời khóa biểu."""
    q = normalize_text(question)

    info = {
        "class_name": None,
        "grade": None,
        "day": None,
        "session": None,
        "periods": [],
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

    day_patterns = [
        (r"\bthứ\s*(hai|2)\b", "2"),
        (r"\bthứ\s*(ba|3)\b", "3"),
        (r"\bthứ\s*(tư|4)\b", "4"),
        (r"\bthứ\s*(năm|5)\b", "5"),
        (r"\bthứ\s*(sáu|6)\b", "6"),
        (r"\bthứ\s*(bảy|7)\b", "7"),
    ]
    for pattern, value in day_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            info["day"] = value
            break

    if re.search(r"\bsáng\b", q, re.IGNORECASE):
        info["session"] = "S"
    elif re.search(r"\bchiều\b", q, re.IGNORECASE):
        info["session"] = "C"

    # Hỗ trợ: "tiết 1", "tiết 4 và 5", "tiết 2,3".
    matches = re.findall(
        r"\btiết\s*(\d+(?:\s*(?:,|và|&|- )\s*\d+)*)",
        q,
        re.IGNORECASE,
    )
    for group in matches:
        for number in re.findall(r"\d+", group):
            if number not in info["periods"]:
                info["periods"].append(number)

    return info


def _header_value(document: str, field: str) -> str | None:
    match = re.search(
        rf"^\s*{re.escape(field)}\s*:\s*([^\n]+)",
        document,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _has_class(document: str, class_name: str) -> bool:
    return bool(
        re.search(
            rf"^\s*{re.escape(class_name)}\s*:",
            document,
            re.MULTILINE | re.IGNORECASE,
        )
    )


def _has_grade_class(document: str, grade: str) -> bool:
    return bool(
        re.search(
            rf"^\s*{re.escape(grade)}a\d+\s*:",
            document,
            re.MULTILINE | re.IGNORECASE,
        )
    )


def structured_timetable_search(question: str, all_chunks: list[dict]) -> list[dict]:
    """Tìm chính xác các chunk XLSX theo lớp/khối/thứ/buổi/tiết."""
    info = parse_timetable_query(question)

    has_timetable_signal = any([
        info["class_name"],
        info["grade"],
        info["day"],
        info["session"],
        info["periods"],
    ])

    if not has_timetable_signal:
        return []

    candidates = []

    for row in all_chunks:
        metadata = row.get("metadata") or {}
        if metadata.get("file_type") != "xlsx":
            continue

        document = row.get("document", "")

        day = _header_value(document, "Thứ")
        session = _header_value(document, "Buổi")
        period = _header_value(document, "Tiết")

        if info["day"] and day != info["day"]:
            continue
        if info["session"] and session != info["session"]:
            continue
        if info["periods"] and period not in info["periods"]:
            continue

        if info["class_name"] and not _has_class(document, info["class_name"]):
            continue
        if info["grade"] and not _has_grade_class(document, info["grade"]):
            continue

        # Nếu chỉ có khối/ngày/buổi/tiết, một chunk đã chứa toàn bộ lớp của khối.
        # Nếu có tên lớp cụ thể, chunk được xem là khớp chính xác hơn.
        exactness = 0
        exactness += 4 if info["class_name"] else 0
        exactness += 2 if info["grade"] else 0
        exactness += 2 if info["day"] else 0
        exactness += 2 if info["session"] else 0
        exactness += 2 if info["periods"] else 0

        candidates.append({
            "document": document,
            "metadata": metadata,
            "distance": None,
            "bm25_score": None,
            "structured_score": exactness,
            "query_info": info,
        })

    # Ưu tiên các chunk khớp nhiều điều kiện hơn, sau đó theo tiết.
    candidates.sort(
        key=lambda row: (
            -row["structured_score"],
            int(_header_value(row["document"], "Tiết") or 0),
            row["metadata"].get("source", ""),
            row["metadata"].get("chunk_index", 0),
        )
    )

    return [dict(row, rank=rank) for rank, row in enumerate(candidates, start=1)]


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


def semantic_search(
    question: str,
    collection,
    model,
    file_types=None,
):
    """Semantic retrieval bằng Chroma ANN, chỉ lấy candidate cần thiết."""
    if file_types is None:
        file_types = PROSE_FILE_TYPES
    file_types = set(file_types)

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    # Chroma lọc metadata trước khi truy vấn, sau đó dùng ANN index.
    where = {
        "file_type": {"$in": list(file_types)}
    } if len(file_types) > 1 else {"file_type": next(iter(file_types))}

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(SEMANTIC_TOP_K, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    rows = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        rows.append({
            "document": document or "",
            "metadata": metadata or {},
            "distance": float(distance),
            "semantic_score": 1.0 - float(distance),
            "rank": rank,
        })

    return rows

def bm25_search(question: str, all_chunks, file_types=None):
    """BM25 search với index được xây một lần và tái sử dụng."""
    if file_types is None:
        file_types = PROSE_FILE_TYPES
    file_types = frozenset(file_types)

    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    cache_key = (id(all_chunks), file_types)
    index = _BM25_CACHE.get(cache_key)

    if index is None:
        candidate_chunks = [
            row
            for row in all_chunks
            if (row.get("metadata") or {}).get("file_type") in file_types
        ]

        tokenized_docs = []
        document_frequency = Counter()
        for row in candidate_chunks:
            metadata = row.get("metadata") or {}
            retrieval_text = " ".join(
                part
                for part in (
                    metadata.get("source", ""),
                    metadata.get("section", ""),
                    row.get("document", ""),
                )
                if part
            )
            tokens = tokenize(retrieval_text)
            tokenized_docs.append(tokens)
            for token in set(tokens):
                document_frequency[token] += 1

        n_docs = len(candidate_chunks)
        avgdl = (
            sum(len(tokens) for tokens in tokenized_docs) / n_docs
            if n_docs else 0.0
        )

        index = {
            "rows": candidate_chunks,
            "tokenized_docs": tokenized_docs,
            "document_frequency": document_frequency,
            "n_docs": n_docs,
            "avgdl": avgdl,
        }
        _BM25_CACHE[cache_key] = index

    candidate_chunks = index["rows"]
    tokenized_docs = index["tokenized_docs"]
    document_frequency = index["document_frequency"]
    n_docs = index["n_docs"]
    avgdl = index["avgdl"]

    k1 = 1.5
    b = 0.75
    scored = []
    query_counter = Counter(query_tokens)

    normalized_question = normalize_text(question)
    question_terms = [t for t in query_tokens if len(t) > 2]
    phrase = " ".join(question_terms) if len(question_terms) >= 2 else ""

    for row, tokens in zip(candidate_chunks, tokenized_docs):
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
            denominator = term_tf + k1 * (
                1 - b + b * (doc_length / avgdl if avgdl else 1)
            )
            score += idf * (numerator / denominator)

        metadata = row.get("metadata") or {}
        retrieval_text = " ".join(
            part
            for part in (
                metadata.get("source", ""),
                metadata.get("section", ""),
                row.get("document", ""),
            )
            if part
        )
        if phrase and phrase in normalize_text(retrieval_text):
            score += 1.0

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

def retrieve_prose(question: str, collection, all_chunks: list[dict], model) -> list[dict]:
    """Hybrid retrieval dành riêng cho PDF/DOCX.

    Không trộn XLSX vào semantic/BM25/RRF của văn bản prose.
    Python chỉ tách kho dữ liệu theo file_type; không diễn giải ý nghĩa câu hỏi.
    """
    semantic_rows = semantic_search(
        question,
        collection,
        model,
        file_types=PROSE_FILE_TYPES,
    )
    lexical_rows = bm25_search(
        question,
        all_chunks,
        file_types=PROSE_FILE_TYPES,
    )
    return rrf_fusion(semantic_rows, lexical_rows)



def source_key(row):
    """Định danh tài liệu ổn định, ưu tiên file_id để tránh trùng tên file."""
    metadata = row.get("metadata") or {}
    file_id = metadata.get("file_id", "")
    relative_path = metadata.get("relative_path", "")
    source = metadata.get("source", "")
    return (file_id, relative_path, source)


def bm25_scores_for_rows(question, rows, all_chunks, file_types=None):
    """Tính BM25 cho một nhóm chunk đã được chọn, dùng DF/TF index đã cache.

    Không rebuild tokenization hay document-frequency. Hàm này dùng để
    rerank các chunk bên trong các document ứng viên mà không bị giới hạn
    bởi LEXICAL_TOP_K toàn corpus.
    """
    if file_types is None:
        file_types = PROSE_FILE_TYPES
    file_types = frozenset(file_types)
    query_tokens = tokenize(question)
    if not query_tokens or not rows:
        return {}

    # Dùng đúng BM25 index của corpus hiện tại.
    index = _BM25_CACHE.get((id(all_chunks), file_types))

    if index is None:
        tokenized_docs = []
        for r in rows:
            metadata = r.get("metadata") or {}
            retrieval_text = " ".join(
                part
                for part in (
                    metadata.get("source", ""),
                    metadata.get("section", ""),
                    r.get("document", ""),
                )
                if part
            )
            tokenized_docs.append(tokenize(retrieval_text))
        document_frequency = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                document_frequency[token] += 1
        n_docs = len(rows)
        avgdl = sum(map(len, tokenized_docs)) / n_docs if n_docs else 0.0
        local_docs = rows
    else:
        # Dùng DF toàn corpus để IDF nhất quán giữa các query.
        token_map = {key_for(r): tokens for r, tokens in zip(index["rows"], index["tokenized_docs"])}
        tokenized_docs = []
        for r in rows:
            metadata = r.get("metadata") or {}
            retrieval_text = " ".join(
                part
                for part in (
                    metadata.get("source", ""),
                    metadata.get("section", ""),
                    r.get("document", ""),
                )
                if part
            )
            tokenized_docs.append(token_map.get(key_for(r), tokenize(retrieval_text)))
        document_frequency = index["document_frequency"]
        n_docs = index["n_docs"]
        avgdl = index["avgdl"]
        local_docs = rows

    k1 = 1.5
    b = 0.75
    query_counter = Counter(query_tokens)
    normalized_question = normalize_text(question)
    question_terms = [t for t in query_tokens if len(t) > 2]
    phrase = " ".join(question_terms) if len(question_terms) >= 2 else ""

    scores = {}
    for row, tokens in zip(local_docs, tokenized_docs):
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
            denominator = term_tf + k1 * (1 - b + b * (doc_length / avgdl if avgdl else 1))
            score += idf * (numerator / denominator)
        if phrase and phrase in normalize_text(row.get("document", "")):
            score += 1.0
        if score > 0:
            scores[key_for(row)] = score
    return scores


def document_first_rank(semantic_rows, lexical_rows):
    """Stage 1: tìm các tài liệu ứng viên, KHÔNG quyết định câu trả lời.

    Chỉ dùng top semantic/BM25 toàn cục để phát hiện tài liệu. Sau đó stage 2
    sẽ lấy lại TOÀN BỘ chunk của các tài liệu này và rerank trong từng tài liệu.
    """
    docs = {}

    semantic_stage1 = semantic_rows[:SEMANTIC_TOP_K]
    lexical_stage1 = lexical_rows[:LEXICAL_TOP_K]

    def ensure_doc(row):
        key = source_key(row)
        if key not in docs:
            metadata = row.get("metadata") or {}
            docs[key] = {
                "source_key": key,
                "source": metadata.get("source", ""),
                "relative_path": metadata.get("relative_path", ""),
                "file_id": metadata.get("file_id", ""),
                "semantic_ranks": [],
                "lexical_ranks": [],
                "semantic_hits": 0,
                "lexical_hits": 0,
            }
        return docs[key]

    for row in semantic_stage1:
        item = ensure_doc(row)
        item["semantic_hits"] += 1
        item["semantic_ranks"].append(row["rank"])

    for row in lexical_stage1:
        item = ensure_doc(row)
        item["lexical_hits"] += 1
        item["lexical_ranks"].append(row["rank"])

    ranked = []
    for item in docs.values():
        # Dùng vài hit tốt nhất thay vì chỉ nhìn duy nhất chunk Top 1.
        # Đây vẫn chỉ là rank fusion, không phải semantic rule.
        semantic_score = SEMANTIC_RRF_WEIGHT * sum(
            1.0 / (RRF_K + rank)
            for rank in sorted(item["semantic_ranks"])[:5]
        )
        lexical_score = LEXICAL_RRF_WEIGHT * sum(
            1.0 / (RRF_K + rank)
            for rank in sorted(item["lexical_ranks"])[:5]
        )

        agreement_bonus = (
            AGREEMENT_RRF_WEIGHT / (RRF_K + 1)
            if item["semantic_hits"] and item["lexical_hits"]
            else 0.0
        )

        item["document_score"] = semantic_score + lexical_score + agreement_bonus
        item["best_semantic_rank"] = (
            min(item["semantic_ranks"]) if item["semantic_ranks"] else None
        )
        item["best_lexical_rank"] = (
            min(item["lexical_ranks"]) if item["lexical_ranks"] else None
        )
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            -item["document_score"],
            item["best_semantic_rank"] if item["best_semantic_rank"] is not None else 10**9,
            item["best_lexical_rank"] if item["best_lexical_rank"] is not None else 10**9,
            item["source"],
        )
    )

    return ranked[:DOCUMENT_TOP_K]


def select_document_first_evidence(
    question,
    semantic_rows,
    lexical_rows,
    all_chunks,
    model,
    final_top_k=FINAL_TOP_K,
    collection=None,
):
    """Stage 2: rerank CHUNK BÊN TRONG các document đã được chọn.

    Đây là điểm quan trọng của kiến trúc:
    - Một chunk đứng hạng 150 toàn corpus vẫn được xét nếu document của nó
      được stage 1 chọn.
    - Semantic rank được tính lại trong chính document đó.
    - BM25 được fusion ở cấp chunk.

    Python vẫn chỉ làm similarity/ranking; không có luật kiểu "hiệu trưởng",
    "hạn đăng ký", "thời gian", v.v.
    """
    ranked_documents = document_first_rank(semantic_rows, lexical_rows)
    allowed = {item["source_key"] for item in ranked_documents}

    # Lấy lại TOÀN BỘ chunk của các document ứng viên.
    doc_chunks = defaultdict(list)
    for row in all_chunks:
        if source_key(row) in allowed and (row.get("metadata") or {}).get("file_type") in PROSE_FILE_TYPES:
            doc_chunks[source_key(row)].append(row)

    # Tra lexical score toàn cục của các chunk (có thể đứng rất thấp).
    lexical_map = {key_for(row): row for row in lexical_rows}

    selected_by_doc = {}

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    query_vector_np = np.asarray(query_vector, dtype=np.float32)

    for doc in ranked_documents:
        source = doc["source_key"]
        chunks = doc_chunks.get(source, [])
        if not chunks:
            selected_by_doc[source] = []
            continue

        # Ưu tiên lấy vector đã lưu trong Chroma thay vì encode lại bằng model.
        # Đây là tối ưu quan trọng nhất cho stage 2 khi corpus lớn.
        semantic_candidates = chunks
        if collection is not None and doc.get("file_id"):
            try:
                result = collection.get(
                    where={"file_id": doc["file_id"]},
                    include=["documents", "metadatas", "embeddings"],
                )
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                vectors = result.get("embeddings") or []
                if docs and metas and vectors:
                    semantic_candidates = []
                    for text, metadata, vector in zip(docs, metas, vectors):
                        semantic_candidates.append({
                            "document": text or "",
                            "metadata": metadata or {},
                        })
                    candidate_vectors = np.asarray(vectors, dtype=np.float32)
                else:
                    candidate_vectors = None
            except Exception:
                candidate_vectors = None
        else:
            candidate_vectors = None

        if candidate_vectors is None:
            texts = [row.get("document", "") for row in semantic_candidates]
            candidate_vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )
            candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)

        semantic_scores = candidate_vectors @ query_vector_np
        local_bm25 = bm25_scores_for_rows(
            question, semantic_candidates, all_chunks, file_types=PROSE_FILE_TYPES
        )

        scored_chunks = []
        doc_rank = ranked_documents.index(doc) + 1
        for row, semantic_score in zip(semantic_candidates, semantic_scores.tolist()):
            key = key_for(row)
            scored_chunks.append({
                "document": row.get("document", ""),
                "metadata": row.get("metadata") or {},
                "distance": 1.0 - float(semantic_score),
                "semantic_score_local": float(semantic_score),
                "bm25_score": float(local_bm25.get(key, 0.0)),
                "document_rank": doc_rank,
            })

        # Local semantic rank: chunk đứng thấp toàn corpus vẫn có thể đứng đầu
        # trong document đúng.
        semantic_order = sorted(
            range(len(scored_chunks)),
            key=lambda i: scored_chunks[i]["distance"],
        )
        local_semantic_rank = {}
        for rank, idx in enumerate(semantic_order, start=1):
            local_semantic_rank[idx] = rank

        lexical_candidates = [
            i for i, row in enumerate(scored_chunks) if row["bm25_score"] > 0
        ]
        lexical_order = sorted(
            lexical_candidates,
            key=lambda i: scored_chunks[i]["bm25_score"],
            reverse=True,
        )
        local_lexical_rank = {}
        for rank, idx in enumerate(lexical_order, start=1):
            local_lexical_rank[idx] = rank

        for idx, row in enumerate(scored_chunks):
            semantic_rrf = SEMANTIC_RRF_WEIGHT / (RRF_K + local_semantic_rank[idx])
            lexical_rrf = (
                LEXICAL_RRF_WEIGHT / (RRF_K + local_lexical_rank[idx])
                if idx in local_lexical_rank
                else 0.0
            )
            agreement_bonus = (
                AGREEMENT_RRF_WEIGHT / (RRF_K + 1)
                if idx in local_lexical_rank
                else 0.0
            )
            row["rrf_score"] = semantic_rrf + lexical_rrf + agreement_bonus
            row["semantic_rank_local"] = local_semantic_rank[idx]
            row["lexical_rank_local"] = local_lexical_rank.get(idx)
            row["semantic_rank_global"] = None
            row["lexical_rank_global"] = None

        # Lưu các chunk tốt nhất trong từng document.
        scored_chunks.sort(
            key=lambda row: (
                -row["rrf_score"],
                row["semantic_rank_local"],
                row["lexical_rank_local"] if row["lexical_rank_local"] is not None else 10**9,
            )
        )
        selected_by_doc[source] = scored_chunks[:MAX_CHUNKS_PER_DOCUMENT]

    # Gán lại global rank để in/log dễ đọc và interleave tài liệu.
    for row in semantic_rows:
        key = key_for(row)
        for source_rows in selected_by_doc.values():
            for item in source_rows:
                if key_for(item) == key:
                    item["semantic_rank_global"] = row.get("rank")
                    break

    for row in lexical_rows:
        key = key_for(row)
        for source_rows in selected_by_doc.values():
            for item in source_rows:
                if key_for(item) == key:
                    item["lexical_rank_global"] = row.get("rank")
                    break

    # Xếp hạng cuối dùng cả tín hiệu toàn corpus và local.
    # Global rank giữ lại thông tin “chunk này mạnh thế nào so với toàn KB”,
    # còn local rank bảo vệ trường hợp chunk nằm trong đúng document nhưng
    # đứng thấp toàn corpus.
    for source_rows in selected_by_doc.values():
        for item in source_rows:
            global_semantic = (
                SEMANTIC_RRF_WEIGHT / (RRF_K + item["semantic_rank_global"])
                if item.get("semantic_rank_global") is not None
                else 0.0
            )
            global_lexical = (
                LEXICAL_RRF_WEIGHT / (RRF_K + item["lexical_rank_global"])
                if item.get("lexical_rank_global") is not None
                else 0.0
            )
            global_agreement = (
                AGREEMENT_RRF_WEIGHT / (RRF_K + 1)
                if item.get("semantic_rank_global") is not None
                and item.get("lexical_rank_global") is not None
                else 0.0
            )
            item["global_rrf_score"] = (
                global_semantic + global_lexical + global_agreement
            )
            # Ưu tiên tín hiệu toàn corpus hơn local score để tránh một
            # tài liệu yếu nhưng có local rank 1 lấn át tài liệu thật sự mạnh.
            item["final_score"] = (
                2.0 * item["global_rrf_score"]
                + item["rrf_score"]
                + 0.02 / max(1, item["document_rank"])
            )

    # Document-first vẫn được giữ, nhưng KHÔNG ép mỗi document một chunk.
    # Với câu hỏi factual thông thường, document tốt nhất nên chiếm phần lớn
    # evidence để tài liệu phụ không làm Qwen nhiễu. Document phụ chỉ được
    # tham gia khi nó có ít nhất một hit global thật sự mạnh.
    primary_document = ranked_documents[0]["source_key"] if ranked_documents else None

    selected = []
    primary_rows = sorted(
        selected_by_doc.get(primary_document, []),
        key=lambda row: (-row["final_score"], row["semantic_rank_local"]),
    )
    selected.extend(dict(row) for row in primary_rows[:final_top_k])

    if len(selected) >= final_top_k:
        return selected[:final_top_k]

    for doc in ranked_documents[1:]:
        rows = selected_by_doc.get(doc["source_key"], [])
        if not rows:
            continue

        best_global_rank = min(
            [r for r in (
                *(row.get("semantic_rank_global") for row in rows),
                *(row.get("lexical_rank_global") for row in rows),
            ) if r is not None],
            default=10**9,
        )
        if best_global_rank > SECONDARY_DOCUMENT_MAX_GLOBAL_RANK:
            continue

        secondary_rows = sorted(
            rows,
            key=lambda row: (-row["final_score"], row["semantic_rank_local"]),
        )
        selected.extend(dict(row) for row in secondary_rows)
        if len(selected) >= final_top_k:
            break

    # Nếu document phụ không đủ mạnh, evidence sẽ chỉ đến từ document chính.
    selected.sort(key=lambda row: (-row["final_score"], row.get("document_rank", 10**9)))
    return selected[:final_top_k]


def key_for(row):
    metadata = row.get("metadata") or {}
    document = row.get("document", "")

    # IDs do not need to match between the two retrieval paths.
    # source + location + text is stable enough within this KB.
    return (
        metadata.get("file_id", ""),
        metadata.get("chunk_index", ""),
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

    score = row.get("rrf_score")
    if score is None:
        score = row.get("structured_score", 0.0)
    print(f"[{rank}] SCORE={score:.6f}")
    print(f"Nguồn: {source}")
    print(f"Vị trí: {location}")
    print(
        f"Semantic rank  : global={row.get('semantic_rank_global')} | "
        f"local={row.get('semantic_rank_local', row.get('semantic_rank'))}"
    )
    print(
        f"Lexical rank   : global={row.get('lexical_rank_global')} | "
        f"local={row.get('lexical_rank_local', row.get('lexical_rank'))}"
    )
    print(f"Structured rank: {row.get('structured_rank')}")
    print(f"Document rank  : {row.get('document_rank')}")
    if row.get("distance") is not None:
        print(f"Distance     : {row['distance']:.6f}")
    if row.get("bm25_score") is not None:
        print(f"BM25         : {row['bm25_score']:.6f}")
    print(row["document"][:1200])
    print("-" * 60)


def main():
    print("=" * 60)
    print("SCHOOL AI - BƯỚC 4: HYBRID RETRIEVAL")
    print("PDF/DOCX: Semantic + BM25 + RRF | XLSX/TKB: Structured")
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

            print("\nTKB dùng structured retrieval riêng; không trộn XLSX vào prose RAG.")
            final_rows = structured_rows[:FINAL_TOP_K]

        else:
            print("Không phát hiện truy vấn thời khóa biểu có cấu trúc.")
            print("\n" + "-" * 60)
            print("PROSE SEARCH (CHỈ PDF/DOCX)")
            print("-" * 60)

            semantic_rows = semantic_search(
                question, collection, model, file_types=PROSE_FILE_TYPES
            )
            print(f"Semantic candidate set: chỉ PDF/DOCX ({len(semantic_rows)} kết quả tối đa)")
            for i, row in enumerate(semantic_rows[:5], start=1):
                print(
                    f"Semantic [{i}] distance={row['distance']:.6f} | "
                    f"{row['metadata'].get('source', '')} | "
                    f"{row['metadata'].get('location', '')}"
                )

            lexical_rows = bm25_search(
                question, all_chunks, file_types=PROSE_FILE_TYPES
            )
            for i, row in enumerate(lexical_rows[:5], start=1):
                print(
                    f"BM25 [{i}] score={row['bm25_score']:.6f} | "
                    f"{row['metadata'].get('source', '')} | "
                    f"{row['metadata'].get('location', '')}"
                )

            ranked_documents = document_first_rank(semantic_rows, lexical_rows)
            print("\n" + "=" * 60)
            print("DOCUMENT CANDIDATES")
            print("=" * 60)
            for i, doc in enumerate(ranked_documents, start=1):
                print(
                    f"Document [{i}] score={doc['document_score']:.6f} | "
                    f"semantic_hits={doc['semantic_hits']} | "
                    f"lexical_hits={doc['lexical_hits']} | "
                    f"{doc['source']}"
                )

            final_rows = select_document_first_evidence(
                question,
                semantic_rows,
                lexical_rows,
                all_chunks,
                model,
            )

        print("\n" + "=" * 60)
        print("TOP EVIDENCE SAU KHI DOCUMENT-FIRST")
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
