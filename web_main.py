from __future__ import annotations

import copy
import hashlib
import mimetypes
import time
import uuid
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote
from threading import Lock

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Reuse the current school AI pipeline instead of duplicating the RAG logic.
# web_main.py should sit in the same folder as rag_mvp.py, retrieval.py,
# timetable_engine.py, data/, etc.
import rag_mvp
from rag_mvp import SentenceTransformer
from rag_mvp import COLLECTION_NAME, EMBEDDING_MODEL, load_collection, load_all_chunks
from timetable_engine import TimetableEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm"}

app = FastAPI(title="School AI - THCS&THPT Tiên Yên")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tien-yen.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# One embedding model + one Chroma connection for the whole process.
# This avoids reloading the model for every browser request.
model = None
collection = None
all_chunks: list[dict] = []
base_timetable = None
startup_lock = Lock()

# Prototype session memory. This is intentionally in RAM for v1.
# A future production version can move this to Redis/database.
sessions: dict[str, dict] = {}
sessions_lock = Lock()
MAX_HISTORY_TURNS = 6

# TiYe AI chỉ nhận câu hỏi trong một khung giờ liên tục mỗi ngày.
# 07:00 được tính là bắt đầu hoạt động; 22:00 là thời điểm đóng.
SERVICE_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
SERVICE_START = dtime(7, 0)
SERVICE_END = dtime(22, 0)

REFUSAL = "Không có đủ thông tin trong tài liệu nhà trường được cung cấp để xác nhận câu trả lời này."


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    mode: str
    sources: list[dict] = []
    elapsed_seconds: float


def service_status(now: datetime | None = None) -> tuple[bool, str]:
    """Kiểm tra TiYe AI có đang trong giờ hỗ trợ hay không."""
    current = now.astimezone(SERVICE_TZ) if now is not None else datetime.now(SERVICE_TZ)
    current_time = current.time().replace(tzinfo=None)
    is_open = SERVICE_START <= current_time < SERVICE_END
    label = "Đang hỗ trợ" if is_open else "Ngoài giờ hỗ trợ"
    return is_open, label


def initialize_pipeline() -> None:
    global model, collection, all_chunks, base_timetable

    if model is not None:
        return

    with startup_lock:
        if model is not None:
            return

        print("=" * 60)
        print("SCHOOL AI - WEB SERVER")
        print("Đang khởi động lõi RAG + TKB...")
        print("=" * 60)

        model = SentenceTransformer(EMBEDDING_MODEL)
        collection = load_collection()
        all_chunks = load_all_chunks(collection)
        base_timetable = TimetableEngine()

        print(f"Chroma collection: {COLLECTION_NAME} | {len(all_chunks)} chunks")
        print(f"TKB records: {len(base_timetable.lessons)} tiết")
        print(f"Qwen model: {rag_mvp.OLLAMA_MODEL}")
        print("Web server sẵn sàng.")


def get_session(session_id: str | None) -> tuple[str, dict]:
    sid = session_id or uuid.uuid4().hex
    with sessions_lock:
        session = sessions.get(sid)
        if session is None:
            # Each session gets its own timetable context, while lesson data
            # is copied from the already-parsed base timetable.
            timetable = copy.deepcopy(base_timetable)
            session = {
                "chat_messages": [],
                "timetable": timetable,
            }
            sessions[sid] = session
        return sid, session


def _safe_document_path(relative_path: str) -> Path:
    """Resolve a document path but never allow traversal outside data/documents."""
    raw = (relative_path or "").replace("\\", "/").lstrip("/")
    candidate = (DOCUMENTS_DIR / raw).resolve()
    root = DOCUMENTS_DIR.resolve()
    if candidate == root or root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Đường dẫn tài liệu không hợp lệ.")
    if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Loại tài liệu không được hỗ trợ.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    return candidate


def _document_id(relative_path: str) -> str:
    """Create a stable short id so downloads do not depend on URL-encoded Windows paths."""
    normalized = relative_path.replace("\\", "/").strip("/").lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def document_catalog() -> list[dict]:
    """Danh mục tải xuống lấy trực tiếp từ kho tài liệu thực tế trên máy chủ.

    Chroma chỉ phục vụ retrieval; không dùng metadata Chroma để quyết định
    đường dẫn file tải xuống. Cách này tránh lỗi khi Chroma cũ/mới lệch so với
    thư mục data/documents hoặc khi đường dẫn có Unicode.
    """
    if not DOCUMENTS_DIR.exists():
        return []

    items: list[dict] = []
    seen: set[str] = set()

    for path in DOCUMENTS_DIR.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        relative_path = path.relative_to(DOCUMENTS_DIR).as_posix()
        key = relative_path.replace('\\', '/').strip('/')
        if not key or key in seen:
            continue
        seen.add(key)

        ext = path.suffix.lower()
        type_label = {
            '.pdf': 'PDF',
            '.docx': 'Word',
            '.xlsx': 'Excel',
            '.xlsm': 'Excel',
        }.get(ext, ext.upper().lstrip('.'))

        doc_id = _document_id(key)
        source_group = 'Thời khóa biểu' if ext in {'.xlsx', '.xlsm'} else 'Tài liệu nhà trường'
        items.append({
            'id': doc_id,
            'name': path.name,
            'relative_path': key,
            'type': type_label,
            'download_url': f'/api/documents/{doc_id}/download',
            'size_bytes': path.stat().st_size,
            'source_group': source_group,
        })

    items.sort(key=lambda item: (item['type'], item['name'].casefold()))
    return items

def source_payload(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()

    for row in rows:
        metadata = row.get("metadata") or {}
        source = str(metadata.get("source", "Không rõ"))
        location_parts = []
        for key in ("section", "page", "table", "sheet", "row"):
            value = metadata.get(key)
            if value not in (None, "", []):
                location_parts.append(f"{key}={value}")
        location = ", ".join(location_parts) or "Không rõ"
        key = f"{source}|{location}"
        if key in seen:
            continue
        seen.add(key)
        result.append({"source": source, "location": location})
        if len(result) >= 4:
            break

    return result


def remember_turn(session: dict, question: str, evidence: str, answer: str) -> None:
    chat_messages = session["chat_messages"]
    chat_messages.append(
        {
            "role": "user",
            "content": (
                f"CÂU HỎI CỦA HỌC SINH:\n{question}\n\n"
                f"BẰNG CHỨNG CỦA LƯỢT HIỆN TẠI:\n{evidence}"
            ),
        }
    )
    chat_messages.append({"role": "assistant", "content": answer})

    # Keep the prompt bounded on the local 8 GB machine.
    max_messages = MAX_HISTORY_TURNS * 2
    if len(chat_messages) > max_messages:
        del chat_messages[:-max_messages]


@app.on_event("startup")
def startup_event() -> None:
    initialize_pipeline()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Không tìm thấy giao diện web.")
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health() -> dict:
    initialize_pipeline()
    is_open, label = service_status()
    return {
        "status": "ok",
        "service_open": is_open,
        "service_label": label,
        "service_hours": "07:00–22:00",
        "collection": COLLECTION_NAME,
        "chunks": len(all_chunks),
        "timetable_records": len(base_timetable.lessons),
        "model": rag_mvp.OLLAMA_MODEL,
    }


@app.get("/api/documents")
def documents() -> dict:
    initialize_pipeline()
    items = document_catalog()
    return {"documents": items, "count": len(items)}


@app.get("/api/documents/{document_id}/download")
def download_document(document_id: str) -> FileResponse:
    """Tải file bằng ID hash; tìm file thật trực tiếp trong kho tài liệu."""
    initialize_pipeline()

    if len(document_id) != 16 or any(ch not in "0123456789abcdefABCDEF" for ch in document_id):
        raise HTTPException(status_code=400, detail="Mã tài liệu không hợp lệ.")

    if not DOCUMENTS_DIR.exists():
        raise HTTPException(status_code=404, detail="Kho tài liệu không tồn tại.")

    for path in DOCUMENTS_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue
        relative_path = path.relative_to(DOCUMENTS_DIR).as_posix()
        if _document_id(relative_path) != document_id:
            continue

        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(
            path=path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="attachment",
            headers={"Cache-Control": "no-store"},
        )

    raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu cần tải xuống.")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    initialize_pipeline()

    is_open, _ = service_status()
    if not is_open:
        raise HTTPException(
            status_code=503,
            detail="TiYe AI hiện ngoài giờ hỗ trợ. Thời gian hoạt động: 07:00–22:00.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    sid, session = get_session(request.session_id)
    start = time.perf_counter()

    try:
        # Structured timetable stays exactly where it is today: Python, no Qwen.
        timetable: TimetableEngine = session["timetable"]
        if timetable.is_timetable_query(question):
            result = timetable.answer(question)
            answer = result.get("message", REFUSAL)
            elapsed = time.perf_counter() - start
            return ChatResponse(
                answer=answer,
                session_id=sid,
                mode="timetable",
                sources=[{"source": "TKB trực tiếp từ Excel", "location": "Python structured data"}],
                elapsed_seconds=round(elapsed, 2),
            )

        rows = rag_mvp.retrieve_prose(question, collection, all_chunks, model)
        if not rows:
            answer = REFUSAL
            elapsed = time.perf_counter() - start
            return ChatResponse(
                answer=answer,
                session_id=sid,
                mode="documents",
                sources=[],
                elapsed_seconds=round(elapsed, 2),
            )

        evidence, _ = rag_mvp.build_evidence(rows)
        answer = rag_mvp.ask_qwen(question, evidence, session["chat_messages"])
        answer = answer.strip() or REFUSAL

        # Only remember a non-empty answer; current architecture lets Qwen
        # decide whether evidence is sufficient.
        if answer:
            remember_turn(session, question, evidence, answer)

        elapsed = time.perf_counter() - start
        return ChatResponse(
            answer=answer,
            session_id=sid,
            mode="documents",
            sources=source_payload(rows),
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reset")
def reset_chat(request: ChatRequest) -> dict:
    sid = request.session_id
    if sid:
        with sessions_lock:
            session = sessions.get(sid)
            if session is not None:
                session["chat_messages"] = []
                session["timetable"].context.clear()
        return {"status": "ok", "session_id": sid}

    return {"status": "ok", "session_id": uuid.uuid4().hex}
