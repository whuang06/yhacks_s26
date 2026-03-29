const API = "http://127.0.0.1:8765";

let sessionId = localStorage.getItem("session_id") || "";
let discoveredFiles = [];
let isExpanded = false;

const SPRITE_W = 72;
const SPRITE_H = 72;
const PANEL_W = 380;
const PANEL_H = 520;

/* ── DOM refs ─────────────────────────────────────────────── */

const collapsedView = document.getElementById("collapsed-view");
const expandedView = document.getElementById("expanded-view");
const spriteBtn = document.getElementById("sprite-btn");
const btnMinimize = document.getElementById("btn-minimize");

const tabs = document.querySelectorAll(".tab");
const tabChat = document.getElementById("tab-chat");
const tabFiles = document.getElementById("tab-files");
const tabPreview = document.getElementById("tab-preview");
const tabContents = { chat: tabChat, files: tabFiles, preview: tabPreview };

const log = document.getElementById("log");
const form = document.getElementById("chat-form");
const msg = document.getElementById("msg");
const fileList = document.getElementById("file-list");
const btnClearFiles = document.getElementById("btn-clear-files");
const previewBody = document.getElementById("preview-body");
const previewPath = document.getElementById("preview-path");
const previewPlaceholder = document.getElementById("preview-placeholder");
const btnClosePreview = document.getElementById("btn-close-preview");
const searchInput = document.getElementById("files-search");

/* ── Tauri window helpers ─────────────────────────────────── */

function tauriAvailable() {
  return !!(window.__TAURI__ && window.__TAURI__.window && window.__TAURI__.dpi);
}

async function setWinSize(w, h) {
  if (!tauriAvailable()) return;
  try {
    const win = window.__TAURI__.window.getCurrentWindow();
    await win.setSize(new window.__TAURI__.dpi.LogicalSize(w, h));
  } catch (e) {
    console.warn("setSize:", e);
  }
}

async function setWinPos(x, y) {
  if (!tauriAvailable()) return;
  try {
    const win = window.__TAURI__.window.getCurrentWindow();
    await win.setPosition(new window.__TAURI__.dpi.LogicalPosition(x, y));
  } catch (e) {
    console.warn("setPosition:", e);
  }
}

async function getWinLogicalPos() {
  if (!tauriAvailable()) return null;
  try {
    const win = window.__TAURI__.window.getCurrentWindow();
    const phys = await win.outerPosition();
    const scale = await win.scaleFactor();
    return { x: phys.x / scale, y: phys.y / scale };
  } catch (e) {
    console.warn("outerPosition:", e);
    return null;
  }
}

async function setWinResizable(on) {
  if (!tauriAvailable()) return;
  try {
    const win = window.__TAURI__.window.getCurrentWindow();
    await win.setResizable(on);
  } catch (e) {
    console.warn("setResizable:", e);
  }
}

/* ── Sprite toggle ────────────────────────────────────────── */

async function expand() {
  if (isExpanded) return;
  isExpanded = true;
  const pos = await getWinLogicalPos();
  if (pos) await setWinPos(pos.x - (PANEL_W - SPRITE_W), pos.y);
  await setWinSize(PANEL_W, PANEL_H);
  await setWinResizable(true);
  collapsedView.classList.add("hidden");
  expandedView.classList.remove("hidden");
}

async function collapse() {
  if (!isExpanded) return;
  isExpanded = false;
  const pos = await getWinLogicalPos();
  expandedView.classList.add("hidden");
  collapsedView.classList.remove("hidden");
  await setWinResizable(false);
  if (pos) await setWinPos(pos.x + (PANEL_W - SPRITE_W), pos.y);
  await setWinSize(SPRITE_W, SPRITE_H);
}

/* ── Collapsed sprite: drag to move, click to expand ──────── */

let _spriteDown = null;

spriteBtn.addEventListener("mousedown", (e) => {
  _spriteDown = { x: e.screenX, y: e.screenY, t: Date.now(), dragging: false };
});

spriteBtn.addEventListener("mousemove", async (e) => {
  if (!_spriteDown || _spriteDown.dragging) return;
  const dx = Math.abs(e.screenX - _spriteDown.x);
  const dy = Math.abs(e.screenY - _spriteDown.y);
  if (dx + dy > 4) {
    _spriteDown.dragging = true;
    if (tauriAvailable()) {
      try {
        await window.__TAURI__.window.getCurrentWindow().startDragging();
      } catch (_) {}
    }
  }
});

spriteBtn.addEventListener("mouseup", () => {
  if (_spriteDown && !_spriteDown.dragging && Date.now() - _spriteDown.t < 400) {
    expand();
  }
  _spriteDown = null;
});

spriteBtn.addEventListener("mouseleave", () => {
  _spriteDown = null;
});

/* ── Expanded panel header: drag to move ─────────────────── */

document.querySelector(".panel-header").addEventListener("mousedown", async (e) => {
  if (e.target.closest("button")) return;
  if (tauriAvailable()) {
    try {
      await window.__TAURI__.window.getCurrentWindow().startDragging();
    } catch (_) {}
  }
});

btnMinimize.addEventListener("click", collapse);

/* ── Tab switching ────────────────────────────────────────── */

function switchTab(name) {
  tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  Object.entries(tabContents).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
}

tabs.forEach((t) =>
  t.addEventListener("click", () => switchTab(t.dataset.tab))
);

/* ── Utilities ────────────────────────────────────────────── */

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/* ── Chat rendering ───────────────────────────────────────── */

function renderReplyContent(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => {
    return `<pre class="reply-code">${code.trim()}</pre>`;
  });
  html = html.replace(/`([^`]+)`/g, '<code class="reply-inline-code">$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\n/g, "<br>");
  return html;
}

function append(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg-group ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  if (role === "assistant") {
    bubble.innerHTML = renderReplyContent(text);
  } else {
    bubble.textContent = text;
  }
  wrapper.appendChild(bubble);

  log.appendChild(wrapper);
  log.scrollTop = log.scrollHeight;
}

/* ── Thinking block (collapsible, like ChatGPT reasoning) ── */

function showThinkingBlock(planText) {
  const block = document.createElement("div");
  block.className = "thinking-block";
  block.id = "current-thinking-block";

  const header = document.createElement("button");
  header.className = "thinking-header";
  header.innerHTML =
    '<span class="thinking-icon">\u2728</span>' +
    '<span class="thinking-label">Thinking\u2026</span>' +
    '<span class="thinking-chevron">\u25B6</span>';

  const body = document.createElement("div");
  body.className = "thinking-body collapsed";

  const pre = document.createElement("pre");
  pre.className = "thinking-content";
  pre.textContent = planText;
  body.appendChild(pre);

  header.addEventListener("click", () => {
    const isOpen = !body.classList.contains("collapsed");
    body.classList.toggle("collapsed", isOpen);
    header.querySelector(".thinking-chevron").textContent = isOpen
      ? "\u25B6"
      : "\u25BC";
  });

  block.appendChild(header);
  block.appendChild(body);
  log.appendChild(block);
  log.scrollTop = log.scrollHeight;
  return block;
}

function finalizeThinkingBlock(block, wasAccepted) {
  const label = block.querySelector(".thinking-label");
  if (label) {
    label.textContent = wasAccepted ? "Plan executed" : "Plan declined";
  }
  block.classList.add(wasAccepted ? "thinking-accepted" : "thinking-rejected");
}

/* ── Plan proposal: Accept / Reject UI ────────────────────── */

function showPlanProposal(replyText, onAccept, onReject) {
  const planMatch = replyText.match(
    /=== Plan preview[\s\S]*?(?=\n\n=== dry_run|$)/
  );
  const planPreview = planMatch ? planMatch[0].trim() : replyText;
  const summary = replyText.replace(planPreview, "").trim();

  const thinkingBlock = showThinkingBlock(planPreview);

  if (summary) {
    append("assistant", summary);
  }

  const actions = document.createElement("div");
  actions.className = "plan-actions";

  const btnAccept = document.createElement("button");
  btnAccept.className = "btn-plan btn-accept";
  btnAccept.textContent = "\u2714 Accept & Execute";

  const btnReject = document.createElement("button");
  btnReject.className = "btn-plan btn-reject";
  btnReject.textContent = "\u2718 Reject";

  function disableButtons() {
    btnAccept.disabled = true;
    btnReject.disabled = true;
    actions.classList.add("decided");
  }

  btnAccept.addEventListener("click", () => {
    disableButtons();
    finalizeThinkingBlock(thinkingBlock, true);
    onAccept();
  });

  btnReject.addEventListener("click", () => {
    disableButtons();
    finalizeThinkingBlock(thinkingBlock, false);
    onReject();
  });

  actions.appendChild(btnAccept);
  actions.appendChild(btnReject);
  thinkingBlock.appendChild(actions);
  log.scrollTop = log.scrollHeight;
}

/* ── Send a follow-up message (for Accept/Reject) ────────── */

async function sendFollowUp(text) {
  showThinking();
  try {
    await ensureSessionWithRetry({ attempts: 5, delayMs: 300 });
    const r = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const j = await r.json();
    removeThinking();
    if (!r.ok) {
      const detail = j.detail;
      const errText =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : JSON.stringify(detail);
      throw new Error(errText || r.statusText);
    }
    sessionId = j.session_id;
    localStorage.setItem("session_id", sessionId);
    append("assistant", j.reply);
    if (Array.isArray(j.found_files) && j.found_files.length > 0) {
      replaceDiscoveredFiles(j.found_files);
      updateFilesTabBadge(j.found_files.length);
    }
  } catch (err) {
    removeThinking();
    append("assistant", "Error: " + err.message);
  }
}

/* ── Loading dots ─────────────────────────────────────────── */

function showThinking() {
  const el = document.createElement("div");
  el.className = "thinking-dots";
  el.id = "thinking-indicator";
  el.innerHTML =
    '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function removeThinking() {
  const el = document.getElementById("thinking-indicator");
  if (el) el.remove();
}

/* ── Session / retry ──────────────────────────────────────── */

async function ensureSession() {
  const health = await fetch(`${API}/health`);
  if (!health.ok) throw new Error("API not reachable on port 8765.");
  if (!sessionId) {
    const nr = await fetch(`${API}/api/session/new`, { method: "POST" });
    if (!nr.ok) throw new Error("Could not create session");
    const j = await nr.json();
    sessionId = j.session_id;
    localStorage.setItem("session_id", sessionId);
  }
}

async function ensureSessionWithRetry({ attempts = 30, delayMs = 500 } = {}) {
  let lastError = null;
  for (let i = 0; i < attempts; i += 1) {
    try {
      await ensureSession();
      return;
    } catch (err) {
      lastError = err;
      if (i < attempts - 1) await sleep(delayMs);
    }
  }
  throw lastError || new Error("Could not initialize chat session.");
}

/* ── Chat input: Enter = newline, Cmd/Ctrl+Enter = send ──── */

msg.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.isComposing) return;
  if (e.metaKey || e.ctrlKey) {
    e.preventDefault();
    if (!msg.value.trim()) return;
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = msg.value.trim();
  if (!text) return;
  msg.value = "";
  msg.style.height = "auto";
  append("user", text);
  showThinking();
  try {
    await ensureSessionWithRetry({ attempts: 10, delayMs: 300 });
    const r = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const j = await r.json();
    removeThinking();
    if (!r.ok) {
      const detail = j.detail;
      const errText =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : JSON.stringify(detail);
      throw new Error(errText || r.statusText);
    }
    sessionId = j.session_id;
    localStorage.setItem("session_id", sessionId);

    if (j.is_plan_proposal) {
      showPlanProposal(
        j.reply,
        () => sendFollowUp("Confirmed. Execute the plan now."),
        () => sendFollowUp("Rejected. Do not execute the plan.")
      );
    } else {
      append("assistant", j.reply);
    }

    if (Array.isArray(j.found_files) && j.found_files.length > 0) {
      replaceDiscoveredFiles(j.found_files);
      updateFilesTabBadge(j.found_files.length);
    }
  } catch (err) {
    removeThinking();
    append("assistant", "Error: " + err.message);
  }
});

msg.addEventListener("input", () => {
  msg.style.height = "auto";
  msg.style.height = Math.min(msg.scrollHeight, 96) + "px";
});

/* ── Files tab badge ──────────────────────────────────────── */

function updateFilesTabBadge(count) {
  const filesTab = document.querySelector('.tab[data-tab="files"]');
  if (!filesTab) return;
  filesTab.textContent = count > 0 ? `Files (${count})` : "Files";
}

/* ── Discovered files ─────────────────────────────────────── */

function replaceDiscoveredFiles(paths) {
  discoveredFiles = paths.filter((p) => p && typeof p === "string");
  renderDiscoveredFileList();
}

function renderDiscoveredFileList() {
  fileList.innerHTML = "";
  const query = (searchInput && searchInput.value || "").trim().toLowerCase();
  const filtered = query
    ? discoveredFiles.filter((fp) => fp.toLowerCase().includes(query))
    : discoveredFiles;

  if (filtered.length === 0) {
    const li = document.createElement("li");
    li.className = "file-empty";
    li.textContent =
      discoveredFiles.length === 0
        ? "No files yet. Ask Chat to find something."
        : "No matches for filter.";
    fileList.appendChild(li);
    return;
  }

  for (const filePath of filtered) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "file-entry file-entry-file";
    const displayName = filePath.split("/").pop() || filePath;
    btn.textContent = `\u{1F4C4} ${displayName}`;
    btn.title = filePath;
    btn.addEventListener("click", () => loadPreview(filePath));
    li.appendChild(btn);
    fileList.appendChild(li);
  }
}

if (searchInput) {
  searchInput.addEventListener("input", renderDiscoveredFileList);
}

btnClearFiles.addEventListener("click", () => {
  discoveredFiles = [];
  renderDiscoveredFileList();
  updateFilesTabBadge(0);
});

/* ── Quick semantic search from Files tab ─────────────────── */

const btnQuickSearch = document.getElementById("btn-quick-search");
if (btnQuickSearch && searchInput) {
  btnQuickSearch.addEventListener("click", async () => {
    const q = searchInput.value.trim();
    if (!q) return;
    btnQuickSearch.disabled = true;
    btnQuickSearch.textContent = "Searching\u2026";
    try {
      const r = await fetch(
        `${API}/api/semantic/search?q=${encodeURIComponent(q)}`
      );
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      const paths = (j.hits || []).map((h) => h.filepath).filter(Boolean);
      replaceDiscoveredFiles(paths);
      updateFilesTabBadge(paths.length);
    } catch (err) {
      console.error("Quick search:", err);
    } finally {
      btnQuickSearch.disabled = false;
      btnQuickSearch.textContent = "\u{1F50D}";
    }
  });

  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) {
      e.preventDefault();
      btnQuickSearch.click();
    }
  });
}

/* ── Preview ──────────────────────────────────────────────── */

function clearPreview() {
  previewBody.innerHTML = "";
  previewBody.classList.add("is-hidden");
  previewPath.textContent = "";
  previewPlaceholder.classList.remove("is-hidden");
}

function showPreviewLoading(name) {
  previewPlaceholder.classList.add("is-hidden");
  previewPath.textContent = name;
  previewBody.classList.remove("is-hidden");
  previewBody.innerHTML = '<p class="preview-loading">Loading\u2026</p>';
}

async function loadPreview(filePath) {
  switchTab("preview");
  showPreviewLoading(filePath);
  const url = `${API}/api/file/preview?rel_path=${encodeURIComponent(filePath)}`;
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) {
    const detail = j.detail;
    const err =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : JSON.stringify(detail);
    previewBody.innerHTML = `<p class="preview-error">${escapeHtml(err || r.statusText)}</p>`;
    return;
  }
  previewPath.textContent = j.path || filePath;
  previewBody.innerHTML = "";
  if (j.kind === "image") {
    const img = document.createElement("img");
    img.className = "preview-image";
    img.alt = j.name || "preview";
    img.onerror = () => {
      previewBody.innerHTML = `<p class="preview-error">Image failed to render.</p>`;
    };
    img.src = `data:${j.mime};base64,${j.content}`;
    previewBody.appendChild(img);
  } else if (j.kind === "pdf") {
    const iframe = document.createElement("iframe");
    iframe.className = "preview-pdf";
    iframe.title = j.name || "PDF preview";
    iframe.src = `data:application/pdf;base64,${j.content}`;
    previewBody.appendChild(iframe);
  } else if (j.kind === "text") {
    const pre = document.createElement("pre");
    pre.className = "preview-text";
    pre.textContent = j.content;
    previewBody.appendChild(pre);
  } else {
    const p = document.createElement("p");
    p.className = "preview-binary";
    p.textContent =
      j.message ||
      `Cannot preview (${j.size != null ? j.size + " bytes" : "binary"}).`;
    previewBody.appendChild(p);
  }
}

btnClosePreview.addEventListener("click", clearPreview);

/* ── Position window to top-right on startup ──────────────── */

async function positionTopRight() {
  const margin = 8;
  const sw = window.screen.availWidth + (window.screen.availLeft || 0);
  const sy = (window.screen.availTop || 0) + 16;
  const sx = sw - SPRITE_W - margin;
  if (tauriAvailable()) {
    const cur = await getWinLogicalPos();
    if (cur && Math.abs(cur.x - sx) < 2 && Math.abs(cur.y - sy) < 2) return;
  }
  await setWinPos(sx, sy);
}

/* ── Init ─────────────────────────────────────────────────── */

window.addEventListener("load", async () => {
  await positionTopRight();
  await setWinResizable(false);
  renderDiscoveredFileList();
  ensureSessionWithRetry({ attempts: 30, delayMs: 500 }).catch((e) => {
    append("system", `Startup error: ${String(e.message)}`);
  });
});
