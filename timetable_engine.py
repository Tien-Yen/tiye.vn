from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

DEFAULT_SOURCE_DIR = Path("data/documents")
SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm"}

CLASS_RE = re.compile(r"\b(6|7|8|9|10|11|12)A\d+\b", re.IGNORECASE)
GRADE_RE = re.compile(r"\bkhối\s*(6|7|8|9|10|11|12)\b", re.IGNORECASE)
PERIOD_RE = re.compile(r"\btiết\s*(\d+)\b", re.IGNORECASE)

DAY_NAMES = {
    "hai": "2", "ba": "3", "tư": "4", "tu": "4", "bốn": "4", "bon": "4",
    "năm": "5", "nam": "5", "sáu": "6", "sau": "6", "bảy": "7", "bay": "7",
}

DAY_DISPLAY = {
    "2": "Thứ Hai", "3": "Thứ Ba", "4": "Thứ Tư",
    "5": "Thứ Năm", "6": "Thứ Sáu", "7": "Thứ Bảy",
}

SESSION_DISPLAY = {"S": "Sáng", "C": "Chiều"}


@dataclass
class Query:
    class_name: str | None = None
    grade: str | None = None
    days: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    periods: list[int] = field(default_factory=list)
    mode: str = "lookup"
    explicit_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Lesson:
    # value giữ nguyên toàn bộ nội dung ô Excel, gồm cả môn + giáo viên nếu file có.
    source: str
    relative_path: str
    title: str
    sheet: str
    class_name: str
    grade: str
    day: str
    session: str
    period: int
    value: str
    source_type: str = "cell"
    ambiguity: str = ""


class TimetableEngine:
    """TKB engine đọc trực tiếp XLSX/XLSM và hiểu merged cells.

    Excel được parse đúng một lần lúc khởi động; mỗi câu hỏi sau đó chỉ
    lọc dữ liệu đã chuẩn hóa trong RAM.

    Quy tắc merged-cell:
      - Merge 1 lớp + nhiều dòng: giá trị áp dụng cho mọi tiết trong vùng merge.
      - Merge nhiều lớp + nhiều dòng: TOÀN BỘ nội dung của ô merge được gán cho
        MỌI lớp có cột nằm trong vùng merge. Không suy đoán ánh xạ từng mục môn
        học sang từng lớp.
      - Merge nhiều lớp chứa một hoạt động chung (ví dụ ÔN HSG): áp dụng cho
        toàn bộ lớp trong vùng merge, giống đúng quy tắc trên.
    """

    def __init__(self, source_dir: Path | str = DEFAULT_SOURCE_DIR) -> None:
        self.source_dir = Path(source_dir)
        self.lessons: list[Lesson] = []
        self.context: dict[str, Any] = {}
        self._by_key: dict[tuple[str, str, str, int], list[Lesson]] = defaultdict(list)
        self._files_loaded: list[str] = []
        self._load()

    # ------------------------- text helpers -------------------------

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", str(text or "")).lower().strip()
        text = text.replace("’", "'").replace("`", "'")
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def clean_cell(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def parse_day_token(token: str) -> str | None:
        value = token.lower().strip()
        if value.isdigit() and value in {"2", "3", "4", "5", "6", "7"}:
            return value
        return DAY_NAMES.get(value)

    @staticmethod
    def parse_session_value(text: str) -> list[str]:
        q = TimetableEngine.normalize(text)
        sessions: list[str] = []
        if re.search(r"\bs[aá]ng\b", q):
            sessions.append("S")
        if re.search(r"\bchi[eề]u\b", q):
            sessions.append("C")
        if not sessions:
            if re.fullmatch(r"s", q):
                sessions.append("S")
            elif re.fullmatch(r"c", q):
                sessions.append("C")
        return sessions

    @staticmethod
    def parse_periods(text: str) -> list[int]:
        q = TimetableEngine.normalize(text)
        found: list[int] = []
        pattern = r"\btiết\s*(\d+)(?:\s*(?:-|–|—|đến|tới)\s*(\d+))?"
        for first, second in re.findall(pattern, q):
            a = int(first)
            b = int(second) if second else None
            values = range(a, b + 1) if b is not None and a <= b <= 20 else [a]
            for value in values:
                if 1 <= value <= 20 and value not in found:
                    found.append(value)
        return found

    @staticmethod
    def parse_days(text: str) -> list[str]:
        q = TimetableEngine.normalize(text)
        found: list[str] = []
        range_match = re.search(
            r"\bthứ\s*(hai|ba|tư|tu|bốn|bon|năm|nam|sáu|sau|bảy|bay|2|3|4|5|6|7)"
            r"\s*(?:đến|tới|->|-)\s*"
            r"(?:thứ\s*)?(hai|ba|tư|tu|bốn|bon|năm|nam|sáu|sau|bảy|bay|2|3|4|5|6|7)\b",
            q,
        )
        if range_match:
            start = TimetableEngine.parse_day_token(range_match.group(1))
            end = TimetableEngine.parse_day_token(range_match.group(2))
            if start and end and int(start) <= int(end):
                return [str(n) for n in range(int(start), int(end) + 1)]

        patterns = [
            r"\bthứ\s*(hai|2)\b", r"\bthứ\s*(ba|3)\b",
            r"\bthứ\s*(tư|4)\b", r"\bthứ\s*(năm|5)\b",
            r"\bthứ\s*(sáu|6)\b", r"\bthứ\s*(bảy|7)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                day = TimetableEngine.parse_day_token(match.group(1))
                if day and day not in found:
                    found.append(day)
        return found

    # ------------------------- Excel parsing -------------------------

    @staticmethod
    def _find_header_row(ws) -> int | None:
        for row_idx in range(1, min(ws.max_row, 15) + 1):
            values = [TimetableEngine.clean_cell(ws.cell(row_idx, col).value) for col in range(1, ws.max_column + 1)]
            nonempty = {v for v in values if v}
            if {"Thứ", "Buổi", "Tiết"}.issubset(nonempty):
                return row_idx
        return None

    @staticmethod
    def _sheet_title(ws) -> str:
        for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 3), values_only=True):
            for value in row:
                if value:
                    return TimetableEngine.clean_cell(value)
        return ""

    @staticmethod
    def _class_columns(ws, header_row: int) -> dict[int, str]:
        result: dict[int, str] = {}
        for col in range(1, ws.max_column + 1):
            value = TimetableEngine.clean_cell(ws.cell(header_row, col).value).upper()
            if re.fullmatch(r"(?:6|7|8|9|10|11|12)A\d+", value):
                result[col] = value
        return result

    @staticmethod
    def _merged_map(ws) -> dict[tuple[int, int], Any]:
        mapping: dict[tuple[int, int], Any] = {}
        for merged in ws.merged_cells.ranges:
            anchor = ws.cell(merged.min_row, merged.min_col)
            for row in range(merged.min_row, merged.max_row + 1):
                for col in range(merged.min_col, merged.max_col + 1):
                    mapping[(row, col)] = (merged, anchor)
        return mapping

    @staticmethod
    def _day_session_by_row(ws, header_row: int) -> dict[int, tuple[str, str]]:
        context: dict[int, tuple[str, str]] = {}
        current_day = ""
        current_session = ""
        for row in range(header_row + 1, ws.max_row + 1):
            day_raw = TimetableEngine.clean_cell(ws.cell(row, 1).value)
            session_raw = TimetableEngine.clean_cell(ws.cell(row, 2).value)
            if day_raw:
                parsed = TimetableEngine.parse_day_token(day_raw)
                if parsed:
                    current_day = parsed
            if session_raw:
                sessions = TimetableEngine.parse_session_value(session_raw)
                if sessions:
                    current_session = sessions[0]
            if current_day and current_session:
                context[row] = (current_day, current_session)
        return context

    @staticmethod
    def _split_multiple_assignments(value: str) -> list[tuple[str, str]]:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in re.split(r"\s*;\s*", text) if part.strip()]
        result: list[tuple[str, str]] = []
        for part in parts:
            part = re.sub(r"\s*\[[^\]]*\]\s*$", "", part).strip()
            # Một mục chuẩn: "Môn - Giáo viên"
            match = re.match(r"^(.+?)\s+-\s+(.+?)\s*$", part)
            if match:
                result.append((match.group(1).strip(), match.group(2).strip()))
            else:
                result.append((part, ""))
        return result

    def _add_lesson(
        self,
        source: str,
        relative_path: str,
        title: str,
        sheet: str,
        class_name: str,
        day: str,
        session: str,
        period: int,
        value: str,
        source_type: str = "cell",
        ambiguity: str = "",
    ) -> None:
        value = self.clean_cell(value)
        if not value:
            return
        value = self._format_subject(value)
        if not value:
            return
        grade_match = re.match(r"(6|7|8|9|10|11|12)A", class_name, re.IGNORECASE)
        grade = grade_match.group(1) if grade_match else ""
        lesson = Lesson(
            source=source,
            relative_path=relative_path,
            title=title,
            sheet=sheet,
            class_name=class_name,
            grade=grade,
            day=day,
            session=session,
            period=period,
            value=value,
            source_type=source_type,
            ambiguity=ambiguity,
        )
        self.lessons.append(lesson)
        self._by_key[(class_name, day, session, period)].append(lesson)

    def _load_workbook(self, path: Path) -> None:
        wb = load_workbook(filename=str(path), read_only=False, data_only=True)
        try:
            relative_path = path.relative_to(self.source_dir).as_posix()
        except ValueError:
            relative_path = path.name

        try:
            for ws in wb.worksheets:
                header_row = self._find_header_row(ws)
                if header_row is None:
                    continue
                class_columns = self._class_columns(ws, header_row)
                if not class_columns:
                    continue

                title = self._sheet_title(ws)
                row_context = self._day_session_by_row(ws, header_row)
                merged_map = self._merged_map(ws)
                processed_merges: set[str] = set()

                # 1) Các merged range liên quan đến cột lớp.
                for merged in ws.merged_cells.ranges:
                    if merged.max_row <= header_row:
                        continue
                    covered_class_cols = [
                        col for col in class_columns
                        if merged.min_col <= col <= merged.max_col
                    ]
                    if not covered_class_cols:
                        continue

                    anchor = ws.cell(merged.min_row, merged.min_col)
                    value = self.clean_cell(anchor.value)
                    if not value:
                        continue

                    key = str(merged)
                    if key in processed_merges:
                        continue
                    processed_merges.add(key)

                    periods: list[int] = []
                    for row in range(max(header_row + 1, merged.min_row), merged.max_row + 1):
                        period_raw = self.clean_cell(ws.cell(row, 3).value)
                        if period_raw.isdigit():
                            periods.append(int(period_raw))
                    periods = list(dict.fromkeys(periods))
                    if not periods:
                        continue

                    # Merge nhiều dòng nhưng chỉ cột Thứ/Buổi: không phải lesson.
                    lesson_classes = [class_columns[c] for c in covered_class_cols]
                    day, session = row_context.get(merged.min_row, ("", ""))
                    if not day or not session:
                        # Anchor có thể nằm trên dòng đầu của vùng mà A/B đang merge.
                        for row in range(merged.min_row, merged.max_row + 1):
                            if row in row_context:
                                day, session = row_context[row]
                                break
                    if not day or not session:
                        continue

                    assignments = self._split_multiple_assignments(value)

                    if len(lesson_classes) == 1:
                        class_name = lesson_classes[0]
                        for period in periods:
                            self._add_lesson(
                                path.name, relative_path, title, ws.title,
                                class_name, day, session, period, value,
                                source_type="merged_single_class",
                            )
                        continue

                    # Quy tắc của dự án: nếu một merged cell phủ qua nhiều cột lớp,
                    # TOÀN BỘ nội dung của ô đó thuộc về MỌI lớp mà vùng merge phủ qua.
                    # Không suy đoán kiểu "mục thứ 1 -> lớp 11A1, mục thứ 2 -> 11A2".
                    # Ví dụ một ô merge K16:P17 chứa nhiều môn cho nhóm 11A1-11A6:
                    # mỗi lớp trong K:P nhận nguyên văn toàn bộ nội dung của ô.
                    for class_name in lesson_classes:
                        for period in periods:
                            self._add_lesson(
                                path.name, relative_path, title, ws.title,
                                class_name, day, session, period, value,
                                source_type="merged_group",
                            )

                # 2) Cell thường: những ô không nằm trong merged range đã xử lý ở trên.
                for row in range(header_row + 1, ws.max_row + 1):
                    day, session = row_context.get(row, ("", ""))
                    period_raw = self.clean_cell(ws.cell(row, 3).value)
                    if not day or not session or not period_raw.isdigit():
                        continue
                    period = int(period_raw)
                    for col, class_name in class_columns.items():
                        merged_info = merged_map.get((row, col))
                        if merged_info is not None:
                            merged_range, _ = merged_info
                            if str(merged_range) in processed_merges:
                                continue
                        value = self.clean_cell(ws.cell(row, col).value)
                        if not value:
                            continue
                        self._add_lesson(
                            path.name, relative_path, title, ws.title,
                            class_name, day, session, period, value,
                            source_type="cell",
                        )
        finally:
            wb.close()

    def _load(self) -> None:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục TKB: {self.source_dir}")
        files = sorted(
            p for p in self.source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
        )
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file XLSX/XLSM trong {self.source_dir}")
        for path in files:
            self._load_workbook(path)
            self._files_loaded.append(path.name)

        self.lessons = self._dedupe_lessons(self.lessons)
        self._by_key = defaultdict(list)
        for lesson in self.lessons:
            self._by_key[(lesson.class_name, lesson.day, lesson.session, lesson.period)].append(lesson)

        self.lessons.sort(key=lambda x: (
            x.class_name, int(x.day), 0 if x.session == "S" else 1, x.period, x.source
        ))

    # ------------------------- query parsing -------------------------

    def parse_query(self, question: str, use_context: bool = True) -> Query:
        q = self.normalize(question)
        query = Query()

        class_match = CLASS_RE.search(q)
        if class_match:
            query.class_name = class_match.group(0).upper()
            query.explicit_fields.add("class_name")
        else:
            grade_match = GRADE_RE.search(q)
            if grade_match:
                query.grade = grade_match.group(1)
                query.explicit_fields.add("grade")

        days = self.parse_days(q)
        if days:
            query.days = days
            query.explicit_fields.add("days")

        sessions = self.parse_session_value(q)
        if sessions:
            query.sessions = sessions
            query.explicit_fields.add("sessions")

        periods = self.parse_periods(q)
        if periods:
            query.periods = periods
            query.explicit_fields.add("periods")

        if not query.periods:
            count_match = re.search(r"\b(?:cả|gồm|có|đủ|trong)\s*(?:toàn bộ\s*)?(\d+)\s*tiết\b", q)
            if count_match:
                count = int(count_match.group(1))
                if 1 <= count <= 20:
                    query.periods = list(range(1, count + 1))
                    query.explicit_fields.add("periods")

        timetable_words = re.search(r"\b(?:tkb|thời khóa biểu|thời khoá biểu)\b", q, re.IGNORECASE)
        explicit_signal = bool(
            timetable_words or query.class_name or query.grade or query.periods or query.sessions
            or (query.days and re.search(r"\bhọc\b|\bmôn\b|\btiết\b", q))
        )

        # Follow-up chỉ kế thừa khi câu hỏi thật sự có dạng nối tiếp.
        followup_words = re.search(r"\b(?:vậy|thế|còn|nữa|tiếp|thì sao)\b", q)
        is_followup = (
            use_context and explicit_signal and not timetable_words
            and (bool(followup_words) or (query.periods and not query.class_name and not query.grade and not query.days and not query.sessions))
        )
        if is_followup:
            self._apply_context(query)

        if timetable_words or len(query.days) > 1 or len(query.sessions) > 1 or len(query.periods) > 1:
            query.mode = "timetable"
        elif query.class_name and not query.days and not query.sessions and not query.periods:
            query.mode = "timetable"
        elif query.class_name and query.days and query.sessions and not query.periods:
            query.mode = "timetable"
        elif query.class_name and query.days and query.sessions and query.periods:
            query.mode = "lookup" if len(query.periods) == 1 else "timetable"
        elif query.periods and query.class_name and not query.days and not query.sessions:
            query.mode = "lookup"
        elif query.periods and self.context.get("class_name"):
            query.mode = "lookup"
        else:
            query.mode = "lookup"
        return query

    def _apply_context(self, query: Query) -> None:
        if query.class_name is None:
            query.class_name = self.context.get("class_name")
        if query.grade is None:
            query.grade = self.context.get("grade")
        if not query.days and self.context.get("days"):
            query.days = list(self.context["days"])
        if not query.sessions and self.context.get("sessions"):
            query.sessions = list(self.context["sessions"])

    def is_timetable_query(self, question: str) -> bool:
        q = self.normalize(question)
        parsed = self.parse_query(question, use_context=False)
        if re.search(r"\b(?:tkb|thời khóa biểu|thời khoá biểu)\b", q):
            return True
        if CLASS_RE.search(q) or GRADE_RE.search(q) or PERIOD_RE.search(q):
            return True
        if re.search(r"\bbuổi\s+(?:sáng|chiều)\b", q):
            return True
        if parsed.days and re.search(r"\bhọc\b|\bmôn\b|\btiết\b", q):
            return True
        if self.context and re.search(r"\b(?:tiết|buổi|thứ|sáng|chiều|còn|nữa|vậy|thế)\b", q):
            return True
        return False

    # ------------------------- querying -------------------------

    def _candidate_lessons(self, query: Query) -> list[Lesson]:
        lessons = self.lessons
        if query.class_name:
            lessons = [x for x in lessons if x.class_name == query.class_name]
        elif query.grade:
            lessons = [x for x in lessons if x.grade == query.grade]
        if query.days:
            wanted = set(query.days)
            lessons = [x for x in lessons if x.day in wanted]
        if query.sessions:
            wanted = set(query.sessions)
            lessons = [x for x in lessons if x.session in wanted]
        if query.periods:
            wanted = set(query.periods)
            lessons = [x for x in lessons if x.period in wanted]
        return lessons

    @staticmethod
    def _dedupe_lessons(lessons: list[Lesson]) -> list[Lesson]:
        seen: set[tuple[Any, ...]] = set()
        result: list[Lesson] = []
        for lesson in lessons:
            key = (
                lesson.source, lesson.sheet, lesson.class_name, lesson.day,
                lesson.session, lesson.period, lesson.value,
            )
            if key not in seen:
                seen.add(key)
                result.append(lesson)
        return result

    @staticmethod
    def _format_subject(value: str) -> str:
        """Giữ nguyên môn + giáo viên như dữ liệu Excel, chỉ chuẩn hóa khoảng trắng.

        Ví dụ:
          HĐTN-Loan -> HĐTN-Loan
          Ngữ văn-T.Trang -> Ngữ văn-T.Trang
          A - Thầy B; C - Cô D -> A - Thầy B / C - Cô D
        """
        parts = [part.strip() for part in re.split(r"\s*;\s*", value or "") if part.strip()]
        if not parts:
            return ""
        cleaned = []
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip()
            if part == "[TRỐNG]":
                cleaned.append(part)
            else:
                part = re.sub(r"\s*\[([^\]]*)\]\s*$", r" [\1]", part).strip()
                cleaned.append(part)
        return " / ".join(cleaned)

    def answer(self, question: str) -> dict[str, Any]:
        query = self.parse_query(question, use_context=True)
        if query.mode == "timetable" and not query.class_name and not query.grade:
            return {"ok": False, "type": "timetable", "message": "Mình chưa xác định được lớp cần tra thời khóa biểu.", "query": query.__dict__}

        lessons = self._dedupe_lessons(self._candidate_lessons(query))
        lessons.sort(key=lambda x: (int(x.day), 0 if x.session == "S" else 1, x.period, x.class_name))

        if query.class_name and len(query.days) == 1 and len(query.sessions) == 1 and len(query.periods) == 1:
            matches = [x for x in self._by_key.get((query.class_name, query.days[0], query.sessions[0], query.periods[0]), []) if x.value]
            matches = self._dedupe_lessons(matches)
            if not matches:
                msg = (
                    f"Dữ liệu TKB hiện tại không có bản ghi cho lớp {query.class_name}, "
                    f"{DAY_DISPLAY[query.days[0]]}, {SESSION_DISPLAY[query.sessions[0]]}, tiết {query.periods[0]}."
                )
                return {"ok": False, "type": "timetable_lookup", "message": msg, "query": query.__dict__}
            item = matches[0]
            subject = self._format_subject(item.value)
            msg = (
                f"Lớp {item.class_name}, {DAY_DISPLAY[item.day]}, {SESSION_DISPLAY[item.session]}, "
                f"tiết {item.period}: {subject}."
            )
            if item.ambiguity:
                msg += " [Dữ liệu lấy từ ô nhóm đã gộp trong Excel.]"
            self._update_context(query)
            return {"ok": True, "type": "timetable_lookup", "message": msg, "rows": [self.lesson_to_dict(item) for item in matches], "query": query.__dict__}

        if query.mode == "timetable":
            if not lessons:
                return {"ok": False, "type": "timetable", "message": "Không tìm thấy dữ liệu thời khóa biểu phù hợp.", "query": query.__dict__}
            self._update_context(query)
            msg = self.format_table(lessons)
            if len(query.sessions) > 1:
                returned = {x.session for x in lessons}
                missing = [s for s in query.sessions if s not in returned]
                if missing:
                    names = ", ".join(SESSION_DISPLAY[s].lower() for s in missing)
                    msg += f"\n\nLưu ý: dữ liệu TKB hiện tại không có tiết {names} cho lớp {query.class_name or 'đã chọn'}."
            return {"ok": True, "type": "timetable", "message": msg, "rows": [self.lesson_to_dict(x) for x in lessons], "query": query.__dict__}

        if query.class_name and query.days and query.sessions and query.periods:
            if not lessons:
                return {"ok": False, "type": "timetable_lookup", "message": "Không tìm thấy dữ liệu thời khóa biểu phù hợp.", "query": query.__dict__}
            self._update_context(query)
            return {"ok": True, "type": "timetable", "message": self.format_table(lessons), "rows": [self.lesson_to_dict(x) for x in lessons], "query": query.__dict__}

        return {"ok": False, "type": "timetable", "message": "Mình chưa có đủ thông tin để tra thời khóa biểu.", "query": query.__dict__}

    def _update_context(self, query: Query) -> None:
        if query.class_name:
            self.context["class_name"] = query.class_name
        if query.grade:
            self.context["grade"] = query.grade
        if len(query.days) == 1:
            self.context["days"] = list(query.days)
        elif len(query.days) > 1:
            self.context.pop("days", None)
        if len(query.sessions) == 1:
            self.context["sessions"] = list(query.sessions)
        elif len(query.sessions) > 1:
            self.context.pop("sessions", None)
        if len(query.periods) == 1:
            self.context["periods"] = list(query.periods)
        elif len(query.periods) > 1:
            self.context.pop("periods", None)

    @staticmethod
    def lesson_to_dict(item: Lesson) -> dict[str, Any]:
        subject = TimetableEngine._format_subject(item.value)
        return {
            "class_name": item.class_name,
            "day": item.day,
            "day_name": DAY_DISPLAY.get(item.day, item.day),
            "session": item.session,
            "session_name": SESSION_DISPLAY.get(item.session, item.session),
            "period": item.period,
            "subject": subject,
            "value": item.value,
            "source": item.source,
            "relative_path": item.relative_path,
            "sheet": item.sheet,
            "title": item.title,
            "source_type": item.source_type,
            "ambiguity": item.ambiguity,
        }

    @staticmethod
    def format_table(lessons: list[Lesson]) -> str:
        lines = [
            "| Lớp | Thứ | Buổi | Tiết | Môn học |",
            "|---|---|---|---:|---|",
        ]
        for item in lessons:
            subject = TimetableEngine._format_subject(item.value)
            suffix = "*" if item.ambiguity else ""
            lines.append(
                f"| {item.class_name} | {DAY_DISPLAY.get(item.day, item.day)} | "
                f"{SESSION_DISPLAY.get(item.session, item.session)} | {item.period} | "
                f"{subject}{suffix} |"
            )
        if any(item.ambiguity for item in lessons):
            lines.append("\n* Một số tiết lấy từ ô nhóm nhiều lớp trong Excel theo thứ tự từ trái sang phải.")
        return "\n".join(lines)


if __name__ == "__main__":
    engine = TimetableEngine()
    print(f"Đã nạp {len(engine.lessons)} bản ghi tiết học từ: {', '.join(engine._files_loaded)}")
    print("Nhập câu hỏi TKB, 'thoat' để thoát.")
    while True:
        question = input("TKB > ").strip()
        if question.lower() == "thoat":
            break
        if not question:
            continue
        if not engine.is_timetable_query(question):
            print("[Không phải truy vấn TKB]")
            continue
        result = engine.answer(question)
        print(result["message"])
