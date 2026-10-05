// Public TiYe AI backend
const API_BASE = "https://desktop-vfrv4p2.tail03a7f6.ts.net";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

const state = {
  sessionId: localStorage.getItem("tiye_ai_session") || null,
  busy: false,
  serviceOpen: null,
};

const messages = document.getElementById("messages");
const welcomePanel = document.getElementById("welcomePanel");
const form = document.getElementById("chatForm");
const input = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const documentsBtn = document.getElementById("documentsBtn");
const documentsOverlay = document.getElementById("documentsOverlay");
const documentsClose = document.getElementById("documentsClose");
const documentSearch = document.getElementById("documentSearch");
const documentsList = document.getElementById("documentsList");
const documentCount = document.getElementById("documentCount");
let documentsCache = [];

function setStatus(ok, label) {
  statusDot.classList.toggle("online", ok);
  statusDot.classList.toggle("offline", !ok);
  statusText.textContent = label;
}


async function checkHealth() {
  try {
    const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Health check failed");

    state.serviceOpen = Boolean(data.service_open);
    if (state.serviceOpen) {
      setStatus(true, "Đang hỗ trợ");
      input.disabled = state.busy;
      sendBtn.disabled = state.busy;
    } else {
      setStatus(false, "Ngoài giờ hỗ trợ · 07:00–22:00");
      input.disabled = true;
      sendBtn.disabled = true;
    }
  } catch (_) {
    state.serviceOpen = null;
    setStatus(false, "Backend chưa sẵn sàng");
    input.disabled = true;
    sendBtn.disabled = true;
  }
}

function appendMessage(role, text, meta = "") {
  const template = document.getElementById(role === "user" ? "userTemplate" : "botTemplate");
  const node = template.content.cloneNode(true);
  const bubble = node.querySelector(".bubble");
  bubble.textContent = text;
  if (role === "bot") node.querySelector(".meta-line").textContent = meta;
  messages.appendChild(node);
  messages.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "message-row bot-row";
  row.id = "typingRow";
  row.innerHTML = `
    <div class="avatar bot-avatar">✦</div>
    <div class="message-content">
      <div class="message-label bot-label">TiYe AI</div>
      <div class="bubble bot-bubble"><span class="typing"><i></i><i></i><i></i></span></div>
    </div>`;
  messages.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function removeTyping() {
  document.getElementById("typingRow")?.remove();
}

function setBusy(value) {
  state.busy = value;
  sendBtn.disabled = value;
  input.disabled = value;
}

function hideWelcome() {
  welcomePanel.style.display = "none";
}

function showWelcome() {
  welcomePanel.style.display = "block";
}

async function sendQuestion(question) {
  const clean = question.trim();
  if (!clean || state.busy || state.serviceOpen === false) return;

  if (state.serviceOpen === null) {
    await checkHealth();
    if (state.serviceOpen !== true) return;
  }

  hideWelcome();
  appendMessage("user", clean);
  input.value = "";
  input.style.height = "auto";
  setBusy(true);
  showTyping();

  try {
    const res = await fetch(apiUrl("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: clean,
        session_id: state.sessionId,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không thể kết nối TiYe AI.");

    state.sessionId = data.session_id;
    localStorage.setItem("tiye_ai_session", state.sessionId);

    let meta = data.mode === "timetable"
      ? `Thời khóa biểu · ${data.elapsed_seconds}s`
      : `Dữ liệu nhà trường · ${data.elapsed_seconds}s`;

    if (data.sources?.length) {
      const sources = data.sources
        .map((s) => `${s.source}${s.location && s.location !== "Không rõ" ? ` · ${s.location}` : ""}`)
        .join(" | ");
      meta = `${sources} · ${data.elapsed_seconds}s`;
    }

    removeTyping();
    appendMessage("bot", data.answer, meta);
  } catch (err) {
    removeTyping();
    appendMessage("bot", `Có lỗi khi xử lý câu hỏi: ${err.message}`, "Kiểm tra kết nối máy chủ");
  } finally {
    setBusy(false);
    if (state.serviceOpen === false) {
      input.disabled = true;
      sendBtn.disabled = true;
    }
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});


document.querySelectorAll(".quick-card").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.dataset.question || ""));
});

checkHealth();
setInterval(checkHealth, 60 * 1000);
input.focus();

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function documentIcon(type) {
  if (type === "PDF") return "▤";
  if (type === "Word") return "W";
  return "▦";
}

function renderDocuments(filter = "") {
  const q = filter.trim().toLowerCase();
  const items = documentsCache.filter((item) =>
    !q || `${item.name} ${item.type} ${item.source_group}`.toLowerCase().includes(q)
  );

  documentCount.textContent = `${items.length} tài liệu`;
  if (!items.length) {
    documentsList.innerHTML = `<div class="documents-empty">Không tìm thấy tài liệu phù hợp.</div>`;
    return;
  }

  documentsList.innerHTML = items.map((item) => {
    const href = item.download_url ? new URL(item.download_url, API_BASE || window.location.href).href : "";
    const safeHref = href.replaceAll("'", "%27");
    return `
      <article class="document-item">
        <div class="document-file-icon ${item.type.toLowerCase()}">${documentIcon(item.type)}</div>
        <div class="document-info">
          <div class="document-name" title="${item.name}">${item.name}</div>
          <div class="document-meta">${item.type} · ${formatBytes(item.size_bytes)} · ${item.source_group}</div>
        </div>
        <button class="document-download" type="button" data-download-url="${safeHref}" data-download-name="${item.name}" aria-label="Tải ${item.name}" title="Tải xuống">↓</button>
      </article>`;
  }).join("");
}

async function loadDocuments() {
  documentsList.innerHTML = `<div class="documents-loading">Đang tải danh sách tài liệu...</div>`;
  try {
    const res = await fetch(apiUrl("/api/documents"), { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không thể tải danh sách tài liệu.");
    documentsCache = data.documents || [];
    renderDocuments(documentSearch.value);
  } catch (err) {
    documentCount.textContent = "";
    documentsList.innerHTML = `<div class="documents-empty">Không thể tải danh sách tài liệu.</div>`;
  }
}

function openDocuments() {
  documentsOverlay.hidden = false;
  document.body.classList.add("documents-open");
  documentSearch.value = "";
  loadDocuments();
  setTimeout(() => documentSearch.focus(), 40);
}

function closeDocuments() {
  documentsOverlay.hidden = true;
  document.body.classList.remove("documents-open");
  documentsBtn.focus();
}

documentsBtn?.addEventListener("click", openDocuments);
documentsClose?.addEventListener("click", closeDocuments);
documentsOverlay?.addEventListener("click", (event) => {
  if (event.target === documentsOverlay) closeDocuments();
});
documentSearch?.addEventListener("input", () => renderDocuments(documentSearch.value));

// Dùng delegated click thay vì phụ thuộc vào thẻ <a>. Điều này giúp nút tải
// hoạt động ổn định cả trên desktop lẫn mobile và dễ chẩn đoán hơn.
documentsList?.addEventListener("click", (event) => {
  const button = event.target.closest(".document-download");
  if (!button) return;

  const href = button.dataset.downloadUrl;
  if (!href) return;

  button.disabled = true;
  button.textContent = "…";
  window.location.assign(href);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !documentsOverlay.hidden) closeDocuments();
});
