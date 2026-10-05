```javascript
const API_BASE = "https://desktop-qb-971.tail03a7f6.ts.net";

function apiUrl(path) {
  const base = API_BASE.replace(/\/+$/, "");
  const cleanPath = String(path).replace(/^\/+/, "");
  return `${base}/${cleanPath}`;
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
    const res = await fetch(apiUrl("/api/health"), {
      cache: "no-store",
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Health check failed");
    }

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
  const template = document.getElementById(
    role === "user" ? "userTemplate" : "botTemplate"
  );

  const node = template.content.cloneNode(true);
  const bubble = node.querySelector(".bubble");

  bubble.textContent = text;

  if (role === "bot") {
    node.querySelector(".meta-line").textContent = meta;
  }

  messages.appendChild(node);

  messages.lastElementChild?.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}

function showTyping() {
  const row = document.createElement("div");

  row.className = "message-row bot-row";
  row.id = "typingRow";

  row.innerHTML = `
    <div class="avatar bot-avatar">✦</div>
    <div class="message-content">
      <div class="message-label bot-label">TiYe AI</div>
      <div class="bubble bot-bubble">
        <span class="typing"><i></i><i></i><i></i></span>
      </div>
    </div>
  `;

  messages.appendChild(row);

  row.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
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

  if (!clean || state.busy || state.serviceOpen === false) {
    return;
  }

  if (state.serviceOpen === null) {
    await checkHealth();

    if (state.serviceOpen !== true) {
      return;
    }
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
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: clean,
        session_id: state.sessionId,
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(
        data.detail || "Không thể kết nối TiYe AI."
      );
    }

    state.sessionId = data.session_id;

    localStorage.setItem(
      "tiye_ai_session",
      state.sessionId
    );

    let meta =
      data.mode === "timetable"
        ? `Thời khóa biểu · ${data.elapsed_seconds}s`
        : `Dữ liệu nhà trường · ${data.elapsed_seconds}s`;

    if (data.sources?.length) {
      const sources = data.sources
        .map((s) => {
          return `${s.source}${
            s.location && s.location !== "Không rõ"
              ? ` · ${s.location}`
              : ""
          }`;
        })
        .join(" | ");

      meta = `${sources} · ${data.elapsed_secon_
```
