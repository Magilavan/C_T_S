/* ═══════════════════════════════════════════════════════════════════════════
   DrugBot — Application Logic & Security Layer
   Connects to the FastAPI backend at /api/chat, /api/documents, and /api/chat/sessions
   with strict JWT authentication, user-isolated sessions, and dynamic theming.
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── Auth Guard: Redirect to login if token is absent ──
const AUTH_TOKEN = localStorage.getItem("drugbot_token");
if (!AUTH_TOKEN) {
  window.location.href = "/frontend/login.html";
}

// ── Verify session token validity against server ──
(async function verifyAuth() {
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });
    if (!res.ok) {
      localStorage.removeItem("drugbot_token");
      localStorage.removeItem("drugbot_user_email");
      window.location.href = "/frontend/login.html";
    }
  } catch {
    // Graceful offline/network tolerance
  }
})();

// ──────── Application State ────────
const state = {
  sessions: [],            // Array of { id, title, created_at, updated_at } fetched from server
  activeSessionId: null,
  messages: {},            // Cache of { [sessionId]: [{ id, role, text, citations, confidence, scores, safety_notice }] }
  userDocuments: [],       // Array of user-owned PDFs { id, document_id, drug_name, filename, file_size, chunk_count }
  ingestedDrugs: [],       // Aggregated unique drug names
  isStreaming: false,
  themePreference: localStorage.getItem("drugbot_theme") || "system", // 'light' | 'dark' | 'system'
};

// ──────── DOM Elements ────────
const $sidebar        = document.getElementById("sidebar");
const $btnToggle      = document.getElementById("btn-toggle-sidebar");
const $btnOpenSidebar = document.getElementById("btn-open-sidebar");
const $btnNewChat     = document.getElementById("btn-new-chat");
const $chatHistory    = document.getElementById("chat-history");
const $welcome        = document.getElementById("welcome-screen");
const $thread         = document.getElementById("conversation-thread");
const $form           = document.getElementById("chat-form");
const $input          = document.getElementById("message-input");
const $sendBtn        = document.getElementById("btn-send");
const $btnMic         = document.getElementById("btn-mic");
const $btnUpload      = document.getElementById("btn-upload");
const $uploadModal    = document.getElementById("upload-modal");
const $btnCloseModal  = document.getElementById("btn-close-modal");
const $uploadForm     = document.getElementById("upload-form");
const $drugNameInput  = document.getElementById("drug-name-input");
const $pdfFileInput   = document.getElementById("pdf-file-input");
const $dropZone       = document.getElementById("drop-zone");
const $dropZonePrompt = document.getElementById("drop-zone-prompt");
const $selectedFileContainer = document.getElementById("selected-file-container");
const $selectedFileName = document.getElementById("selected-file-name");
const $selectedFileSize = document.getElementById("selected-file-size");
const $btnRemovePdf   = document.getElementById("btn-remove-pdf");
const $uploadProgress = document.getElementById("upload-progress");
const $progressFill   = document.getElementById("progress-fill");
const $uploadStatus   = document.getElementById("upload-status");
const $submitUpload   = document.getElementById("btn-submit-upload");
const $drugsList      = document.getElementById("ingested-drugs-list");

// Theme DOM elements
const $themeBtnLight  = document.getElementById("theme-btn-light");
const $themeBtnDark   = document.getElementById("theme-btn-dark");
const $themeBtnSystem = document.getElementById("theme-btn-system");

// ════════════════════════════════════════════════════════════════════════════
//  THEME ENGINE (Light, Dark, System Preference & Persistence)
// ════════════════════════════════════════════════════════════════════════════

/**
 * Apply the selected theme mode:
 * - 'light': High-contrast, clean achromatic graphite on warm paper palette.
 * - 'dark': Eye-strain reduced deep obsidian slate with luminous accents.
 * - 'system': Automatically detects OS dark/light mode and updates on system changes.
 */
function applyTheme(preference) {
  state.themePreference = preference;
  localStorage.setItem("drugbot_theme", preference);

  let effectiveTheme = preference;
  if (preference === "system") {
    effectiveTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  // Set the data-theme attribute on root <html> element
  document.documentElement.setAttribute("data-theme", effectiveTheme);

  // Update theme toggle UI states
  if ($themeBtnLight && $themeBtnDark && $themeBtnSystem) {
    $themeBtnLight.classList.toggle("active", preference === "light");
    $themeBtnLight.setAttribute("aria-checked", preference === "light" ? "true" : "false");

    $themeBtnDark.classList.toggle("active", preference === "dark");
    $themeBtnDark.setAttribute("aria-checked", preference === "dark" ? "true" : "false");

    $themeBtnSystem.classList.toggle("active", preference === "system");
    $themeBtnSystem.setAttribute("aria-checked", preference === "system" ? "true" : "false");
  }
}

// Attach theme change event listeners
if ($themeBtnLight) $themeBtnLight.addEventListener("click", () => applyTheme("light"));
if ($themeBtnDark) $themeBtnDark.addEventListener("click", () => applyTheme("dark"));
if ($themeBtnSystem) $themeBtnSystem.addEventListener("click", () => applyTheme("system"));

// React to system color scheme changes when user preference is 'system'
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  if (state.themePreference === "system") {
    const effectiveTheme = e.matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", effectiveTheme);
  }
});

// Initialize theme immediately
applyTheme(state.themePreference);

// ════════════════════════════════════════════════════════════════════════════
//  SIDEBAR TOGGLE
// ════════════════════════════════════════════════════════════════════════════
$btnToggle.addEventListener("click", () => toggleSidebar());
$btnOpenSidebar.addEventListener("click", () => toggleSidebar());

function toggleSidebar() {
  const collapsed = $sidebar.classList.toggle("collapsed");
  $btnOpenSidebar.classList.toggle("hidden", !collapsed);
}

// ════════════════════════════════════════════════════════════════════════════
//  AUTHENTICATED SERVER-SIDE SESSION & CHAT HISTORY
// ════════════════════════════════════════════════════════════════════════════

function generateId() {
  return "s-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

/**
 * Fetch all chat sessions belonging strictly to the authenticated user from the backend.
 */
async function fetchChatSessions() {
  try {
    const res = await fetch(`${API_BASE}/api/chat/sessions`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });

    if (!res.ok) {
      if (res.status === 401) {
        window.location.href = "/frontend/login.html";
      }
      return;
    }

    const data = await res.json();
    if (data && Array.isArray(data.sessions)) {
      state.sessions = data.sessions;
      renderChatHistory();
    }
  } catch (err) {
    console.warn("Error fetching chat sessions:", err);
  }
}

/**
 * Load full message history for a specific conversation session from the server.
 * Server validates session ownership; access to other users' sessions is rejected.
 */
async function loadSessionHistory(sessionId) {
  state.activeSessionId = sessionId;
  renderChatHistory();

  // If already cached in memory, render immediately
  if (state.messages[sessionId] && state.messages[sessionId].length > 0) {
    renderThread();
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });

    if (!res.ok) {
      if (res.status === 403 || res.status === 404) {
        showToast("Session access denied or not found.");
        state.sessions = state.sessions.filter(s => s.id !== sessionId);
        renderChatHistory();
        showWelcome();
        return;
      }
      throw new Error(`Failed to load history (${res.status})`);
    }

    const data = await res.json();
    if (data && Array.isArray(data.messages)) {
      state.messages[sessionId] = data.messages.map(m => ({
        id: m.id,
        role: m.role,
        text: m.text,
        citations: m.citations || [],
        confidence: m.confidence || null,
        scores: m.scores || null,
        safety_notice: m.safety_notice || null,
      }));
      renderThread();
    }
  } catch (err) {
    console.warn("Could not load message history:", err);
    renderThread();
  }
}

/**
 * Create a new local conversation draft; session is registered on server upon first message.
 */
function createNewSession(firstMessage) {
  const id = generateId();
  const title = firstMessage
    ? firstMessage.slice(0, 50) + (firstMessage.length > 50 ? "…" : "")
    : "New chat";
  const session = { id, title, created_at: new Date().toISOString() };
  state.sessions.unshift(session);
  state.messages[id] = [];
  setActiveSession(id);
  return id;
}

/**
 * Delete a chat session on the server and remove from local state.
 */
async function deleteSession(id) {
  try {
    const res = await fetch(`${API_BASE}/api/chat/sessions/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });

    if (!res.ok && res.status !== 404) {
      showToast("Failed to delete session on server.");
    }
  } catch (err) {
    console.warn("Session deletion error:", err);
  }

  state.sessions = state.sessions.filter(s => s.id !== id);
  delete state.messages[id];

  if (state.activeSessionId === id) {
    if (state.sessions.length > 0) {
      loadSessionHistory(state.sessions[0].id);
    } else {
      state.activeSessionId = null;
      showWelcome();
    }
  }
  renderChatHistory();
}

function setActiveSession(id) {
  loadSessionHistory(id);
}

function showWelcome() {
  $welcome.classList.remove("hidden");
  $thread.classList.add("hidden");
}

function showThread() {
  $welcome.classList.add("hidden");
  $thread.classList.remove("hidden");
}

// ──────── Render chat history sidebar ────────
function renderChatHistory() {
  $chatHistory.innerHTML = "";

  if (state.sessions.length === 0) {
    $chatHistory.innerHTML = `
      <div class="chat-history-empty">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span>No conversations yet</span>
      </div>`;
    return;
  }

  state.sessions.forEach(session => {
    const card = document.createElement("div");
    card.className = "conversation-card" + (session.id === state.activeSessionId ? " active" : "");
    card.innerHTML = `
      <span class="conversation-card-title">${escapeHtml(session.title || "Untitled Conversation")}</span>
      <button class="conversation-card-delete" title="Delete conversation" aria-label="Delete conversation">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 6 6 18"/><path d="m6 6 12 12"/>
        </svg>
      </button>`;

    card.addEventListener("click", (e) => {
      if (e.target.closest(".conversation-card-delete")) return;
      loadSessionHistory(session.id);
    });

    card.querySelector(".conversation-card-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session.id);
    });

    $chatHistory.appendChild(card);
  });
}

// ──────── Render conversation thread ────────
function renderThread() {
  const msgs = state.messages[state.activeSessionId];
  if (!msgs || msgs.length === 0) {
    if (!state.activeSessionId || !msgs || msgs.length === 0) {
      showWelcome();
    }
    return;
  }

  showThread();
  $thread.innerHTML = "";

  msgs.forEach(msg => {
    appendMessageToDOM(msg);
  });

  scrollToBottom();
}

function appendMessageToDOM(msg) {
  const row = document.createElement("div");
  row.className = "message-row";

  const isUser = msg.role === "user";

  // ── Citations ──
  let citationsHTML = "";
  if (msg.citations && msg.citations.length > 0) {
    const uniqueCitations = [];
    const seen = new Set();
    for (const c of msg.citations) {
      const key = `${c.document}|${c.section}|${c.page}`;
      if (!seen.has(key)) { seen.add(key); uniqueCitations.push(c); }
    }
    citationsHTML = `<div class="message-citations">
      ${uniqueCitations.map(c => {
        const sectionLabel = (c.section && c.section !== "Not available")
          ? `§${escapeHtml(c.section)}` : "Section: Not available";
        const pageLabel = (c.page && c.page !== "Not available")
          ? `p. ${c.page}` : "Page: Not available";
        const docLabel = escapeHtml(c.document || "Prescribing Information");
        return `<span class="citation-tag" title="${docLabel} — ${sectionLabel}, ${pageLabel}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></svg>
          ${sectionLabel} ${pageLabel}
        </span>`;
      }).join("")}
    </div>`;
  }

  // ── Safety notice panel ──
  let safetyNoticeHTML = "";
  if (msg.safety_notice && !isUser) {
    safetyNoticeHTML = `<div class="safety-notice-panel">
      <span class="safety-notice-label">Safety Notice</span>
      <span class="safety-notice-text">${escapeHtml(msg.safety_notice)}</span>
    </div>`;
  }

  // ── Confidence badge ──
  let confidenceHTML = "";
  if (msg.confidence && !isUser && msg.confidence !== "conversational") {
    const scores = msg.scores;
    if (msg.confidence === "high_risk") {
      confidenceHTML = `<div class="confidence-badge confidence-low">Emergency Response</div>`;
    } else if (msg.confidence === "out_of_scope") {
      confidenceHTML = `<div class="confidence-badge confidence-scope">Out-of-Scope (Non-Medical)</div>`;
    } else if (msg.confidence === "not_found") {
      confidenceHTML = `<div class="confidence-badge confidence-low">Not Found in Provided Document</div>`;
    } else if (msg.confidence === "limited_evidence" || msg.confidence === "low_groundedness") {
      confidenceHTML = `<div class="confidence-badge confidence-medium">Document Evidence: Limited</div>`;
    } else if (scores && scores.grounding_score != null) {
      const pct = Math.round(
        (0.3 * (scores.retrieval_score || 0)
        + 0.5 * (scores.grounding_score || 0)
        + 0.2 * (scores.citation_score || 0)) * 100
      );
      let label, levelClass;
      if (pct >= 80) { label = "Document Evidence: High"; levelClass = "confidence-high"; }
      else if (pct >= 55) { label = "Document Evidence: Moderate"; levelClass = "confidence-medium"; }
      else { label = "Document Evidence: Limited"; levelClass = "confidence-low"; }
      confidenceHTML = `<div class="confidence-badge ${levelClass}">${label}</div>`;
    }
  }

  // ── Speaker button for bot messages ──
  let speakerHTML = "";
  if (!isUser && msg.text) {
    speakerHTML = `
      <button class="btn-speak" title="Read response aloud" aria-label="Read response aloud" type="button">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        </svg>
      </button>`;
  }

  const footerActionsHTML = (!isUser) ? `
    <div class="message-footer-actions">
      <div>${confidenceHTML}</div>
      ${speakerHTML}
    </div>` : "";

  row.innerHTML = `
    <div class="message-content">
      <div class="message-avatar ${isUser ? "avatar-user" : "avatar-bot"}">
        ${isUser ? "Y" : "D"}
      </div>
      <div class="message-body">
        <div class="message-text">${isUser ? escapeHtml(msg.text) : formatBotText(msg.text)}</div>
        ${safetyNoticeHTML}
        ${citationsHTML}
        ${footerActionsHTML}
      </div>
    </div>`;

  $thread.appendChild(row);

  if (!isUser) {
    const speakBtn = row.querySelector(".btn-speak");
    if (speakBtn) {
      speakBtn.addEventListener("click", () => speakResponse(msg.text, speakBtn));
    }
  }
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row";
  row.id = "typing-indicator";
  row.innerHTML = `
    <div class="message-content">
      <div class="message-avatar avatar-bot">D</div>
      <div class="message-body">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  $thread.appendChild(row);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    $thread.scrollTop = $thread.scrollHeight;
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  CHAT FORM & INFERENCE
// ════════════════════════════════════════════════════════════════════════════

// Auto-resize textarea
$input.addEventListener("input", () => {
  $input.style.height = "auto";
  $input.style.height = Math.min($input.scrollHeight, 180) + "px";
  $sendBtn.disabled = $input.value.trim().length === 0;
});

// Enter to send (Shift+Enter for newline)
$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!$sendBtn.disabled && !state.isStreaming) {
      $form.dispatchEvent(new Event("submit"));
    }
  }
});

$form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $input.value.trim();
  if (!text || state.isStreaming) return;

  // Stop any ongoing speech or recognition
  stopSpeech();
  if (isListening && recognition) {
    recognition.stop();
  }

  // Create session if needed
  let sessionId = state.activeSessionId;
  if (!sessionId || !state.messages[sessionId]) {
    sessionId = createNewSession(text);
  }

  // Add user message
  const userMsg = { role: "user", text };
  state.messages[sessionId].push(userMsg);

  // Update session title if first message
  if (state.messages[sessionId].length === 1) {
    const session = state.sessions.find(s => s.id === sessionId);
    if (session) {
      session.title = text.slice(0, 50) + (text.length > 50 ? "…" : "");
    }
    renderChatHistory();
  }

  showThread();
  $thread.innerHTML = "";
  state.messages[sessionId].forEach(msg => appendMessageToDOM(msg));
  scrollToBottom();

  // Clear input
  $input.value = "";
  $input.style.height = "auto";
  $sendBtn.disabled = true;

  // Show typing
  state.isStreaming = true;
  appendTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${AUTH_TOKEN}`,
      },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        drug_name_hint: null,
      }),
    });

    if (!res.ok) {
      if (res.status === 401) {
        localStorage.removeItem("drugbot_token");
        window.location.href = "/frontend/login.html";
        return;
      }
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();

    const botMsg = {
      role: "bot",
      text: data.answer,
      citations: data.citations || [],
      confidence: data.confidence || null,
      scores: data.scores || null,
      safety_notice: data.safety_notice || null,
      question_category: data.question_category || null,
      active_drug: data.active_drug || null,
    };

    state.messages[sessionId].push(botMsg);
    removeTypingIndicator();
    appendMessageToDOM(botMsg);
    scrollToBottom();
  } catch (err) {
    removeTypingIndicator();
    const errorMsg = {
      role: "bot",
      text: `⚠ Unable to reach the server: ${err.message}`,
      citations: [],
      confidence: null,
    };
    state.messages[sessionId].push(errorMsg);
    appendMessageToDOM(errorMsg);
    scrollToBottom();
  } finally {
    state.isStreaming = false;
  }
});

// ── New Chat ──
$btnNewChat.addEventListener("click", () => {
  stopSpeech();
  if (isListening && recognition) {
    recognition.stop();
  }

  state.activeSessionId = null;
  $thread.innerHTML = "";
  showWelcome();
  renderChatHistory();
  $input.value = "";
  $input.style.height = "auto";
  $sendBtn.disabled = true;
  $input.focus();
});

// ── Suggestion Chips ──
document.querySelectorAll(".suggestion-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const text = chip.getAttribute("data-suggestion");
    $input.value = text;
    $sendBtn.disabled = false;
    $form.dispatchEvent(new Event("submit"));
  });
});

// ════════════════════════════════════════════════════════════════════════════
//  UPLOAD MODAL & USER-ISOLATED PDF INGESTION
// ════════════════════════════════════════════════════════════════════════════
function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function displaySelectedFile(file) {
  if (!file) return;
  if ($selectedFileName) $selectedFileName.textContent = file.name;
  if ($selectedFileSize) $selectedFileSize.textContent = formatFileSize(file.size);
  if ($selectedFileContainer) $selectedFileContainer.classList.remove("hidden");
  if ($dropZonePrompt) $dropZonePrompt.classList.add("hidden");
  if ($dropZone) $dropZone.classList.add("has-file");
  validateUploadForm();
}

function clearSelectedFile() {
  if ($pdfFileInput) $pdfFileInput.value = "";
  if ($selectedFileName) $selectedFileName.textContent = "";
  if ($selectedFileSize) $selectedFileSize.textContent = "";
  if ($selectedFileContainer) $selectedFileContainer.classList.add("hidden");
  if ($dropZonePrompt) $dropZonePrompt.classList.remove("hidden");
  if ($dropZone) $dropZone.classList.remove("has-file");
  validateUploadForm();
}

function resetUploadModal() {
  if ($uploadForm) $uploadForm.reset();
  clearSelectedFile();
  if ($uploadProgress) $uploadProgress.classList.add("hidden");
  if ($progressFill) {
    $progressFill.style.width = "0%";
    $progressFill.classList.remove("indeterminate");
  }
  if ($uploadStatus) $uploadStatus.textContent = "";
  if ($submitUpload) $submitUpload.disabled = true;
}

function closeModal() {
  $uploadModal.classList.add("hidden");
  resetUploadModal();
}

$btnUpload.addEventListener("click", () => {
  resetUploadModal();
  $uploadModal.classList.remove("hidden");
  $drugNameInput.focus();
});

$btnCloseModal.addEventListener("click", closeModal);
$uploadModal.addEventListener("click", (e) => {
  if (e.target === $uploadModal) closeModal();
});

if ($btnRemovePdf) {
  $btnRemovePdf.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearSelectedFile();
  });
}

// Drag and drop handlers
$dropZone.addEventListener("click", (e) => {
  if (e.target.closest("#btn-remove-pdf")) return;
  $pdfFileInput.click();
});

$dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  $dropZone.classList.add("drag-over");
});

$dropZone.addEventListener("dragleave", () => {
  $dropZone.classList.remove("drag-over");
});

$dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  $dropZone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].name.toLowerCase().endsWith(".pdf")) {
    $pdfFileInput.files = files;
    displaySelectedFile(files[0]);
  }
});

$pdfFileInput.addEventListener("change", () => {
  if ($pdfFileInput.files.length > 0) {
    displaySelectedFile($pdfFileInput.files[0]);
  } else {
    clearSelectedFile();
  }
});

$drugNameInput.addEventListener("input", validateUploadForm);

function validateUploadForm() {
  $submitUpload.disabled = !(
    $drugNameInput.value.trim().length > 0 &&
    $pdfFileInput.files &&
    $pdfFileInput.files.length > 0
  );
}

$uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const drugName = $drugNameInput.value.trim();
  const file = $pdfFileInput.files ? $pdfFileInput.files[0] : null;
  if (!drugName || !file) return;

  $submitUpload.disabled = true;
  $uploadProgress.classList.remove("hidden");
  $progressFill.classList.add("indeterminate");
  $uploadStatus.textContent = "Validating medical domain & ingesting document…";

  const formData = new FormData();
  formData.append("drug_name", drugName);
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/documents/upload`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${AUTH_TOKEN}` },
      body: formData,
    });

    if (!res.ok) {
      let errorMsg = `Upload failed (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) errorMsg = errJson.detail;
      } catch {}
      throw new Error(errorMsg);
    }

    const data = await res.json();

    $progressFill.classList.remove("indeterminate");
    $progressFill.style.width = "100%";
    $uploadStatus.textContent = `✓ Ingested "${drugName}" — ${data.chunk_count || data.chunks_stored || "?"} chunks stored.`;

    // Refresh indexed drugs from authenticated endpoint
    await fetchIndexedDrugs();

    setTimeout(closeModal, 1800);
  } catch (err) {
    $progressFill.classList.remove("indeterminate");
    $progressFill.style.width = "0%";
    $uploadStatus.textContent = `✗ ${err.message}`;
    $submitUpload.disabled = false;
  }
});

// ──────── Fetch and sync user-isolated documents with backend ────────
async function fetchIndexedDrugs() {
  try {
    const res = await fetch(`${API_BASE}/api/documents`, {
      headers: { "Authorization": `Bearer ${AUTH_TOKEN}` },
    });
    if (res.ok) {
      const data = await res.json();
      if (data) {
        state.userDocuments = data.user_documents || [];
        if (Array.isArray(data.documents)) {
          state.ingestedDrugs = data.documents.map(d => d.drug_name.toUpperCase());
        }
        renderIngestedDrugs();
      }
    }
  } catch (err) {
    console.warn("Could not fetch indexed documents from backend:", err);
  }
}

/**
 * Download a PDF strictly isolated to the user with server-side token authorization.
 */
async function downloadUserDocument(docId, filename) {
  try {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(docId)}/download`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });

    if (!res.ok) {
      if (res.status === 403) {
        showToast("Access denied: You can only download your own uploaded PDFs.");
      } else {
        showToast(`Download failed (${res.status})`);
      }
      return;
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "document.pdf";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    showToast(`✓ Downloaded ${filename}`);
  } catch (err) {
    showToast(`✗ Download error: ${err.message}`);
  }
}

// ──────── Delete an indexed drug document ────────
async function deleteDrug(drugName, docId) {
  const confirmed = window.confirm(
    `Are you sure you want to delete "${drugName}"?\n\nThis will remove your stored chunks, embeddings, and PDF file.`
  );
  if (!confirmed) return;

  const targetIdentifier = docId || drugName;

  try {
    const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(targetIdentifier)}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${AUTH_TOKEN}` },
    });

    if (!res.ok) {
      let errDetail = `Failed to delete document (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) errDetail = errJson.detail;
      } catch {}
      throw new Error(errDetail);
    }

    const data = await res.json();
    await fetchIndexedDrugs();
    showToast(`✓ Deleted "${drugName}" (${data.chunks_deleted || 0} chunks removed)`);
  } catch (err) {
    showToast(`✗ ${err.message}`);
  }
}

// ──────── Render user-isolated ingested drugs in sidebar ────────
function renderIngestedDrugs() {
  $drugsList.innerHTML = "";

  // Prefer rendering user_documents if available
  if (state.userDocuments && state.userDocuments.length > 0) {
    state.userDocuments.forEach(doc => {
      const badge = document.createElement("div");
      badge.className = "drug-badge";
      badge.innerHTML = `
        <div class="drug-badge-left">
          <span class="drug-badge-dot"></span>
          <span class="drug-badge-name" title="${escapeHtml(doc.drug_name)} — ${escapeHtml(doc.filename)}">${escapeHtml(doc.drug_name)}</span>
        </div>
        <div class="drug-badge-actions">
          <button class="btn-drug-action btn-download-drug" title="Download PDF (${escapeHtml(doc.filename)})" aria-label="Download ${escapeHtml(doc.drug_name)} PDF">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          </button>
          <button class="btn-drug-action btn-delete-drug" title="Delete ${escapeHtml(doc.drug_name)} document" aria-label="Delete ${escapeHtml(doc.drug_name)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
            </svg>
          </button>
        </div>
      `;

      const downloadBtn = badge.querySelector(".btn-download-drug");
      if (downloadBtn) {
        downloadBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          downloadUserDocument(doc.document_id, doc.filename);
        });
      }

      const delBtn = badge.querySelector(".btn-delete-drug");
      if (delBtn) {
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteDrug(doc.drug_name, doc.document_id);
        });
      }

      $drugsList.appendChild(badge);
    });
    return;
  }

  // Fallback to aggregated drug names if userDocuments is empty
  if (!state.ingestedDrugs || state.ingestedDrugs.length === 0) {
    const emptyNotice = document.createElement("div");
    emptyNotice.className = "drug-badge-empty";
    emptyNotice.textContent = "No documents indexed yet";
    $drugsList.appendChild(emptyNotice);
    return;
  }

  state.ingestedDrugs.forEach(name => {
    const badge = document.createElement("div");
    badge.className = "drug-badge";
    badge.innerHTML = `
      <div class="drug-badge-left">
        <span class="drug-badge-dot"></span>
        <span class="drug-badge-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
      </div>
      <div class="drug-badge-actions">
        <button class="btn-drug-action btn-delete-drug" title="Delete ${escapeHtml(name)} document" aria-label="Delete ${escapeHtml(name)}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
          </svg>
        </button>
      </div>
    `;

    const delBtn = badge.querySelector(".btn-delete-drug");
    if (delBtn) {
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteDrug(name);
      });
    }

    $drugsList.appendChild(badge);
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  UTILITIES & FORMATTING
// ════════════════════════════════════════════════════════════════════════════
function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatBotText(text) {
  let html = escapeHtml(text);

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Inline code: `text`
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Inline source citations: [§Section, p.Page] or [Section X, p.Y]
  html = html.replace(/\[(§[^\]]+)\]/g, '<span class="inline-citation-tag">[$1]</span>');

  // Bullet lists: lines starting with "- " or "• "
  html = html.replace(/^[•\-]\s+(.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

  // Numbered lists: lines starting with "1. ", "2. " etc.
  html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");

  // Paragraphs (double newlines)
  html = html.replace(/\n{2,}/g, "</p><p>");
  html = `<p>${html}</p>`;

  // Single newlines become <br> (but not inside lists)
  html = html.replace(/(?<!<\/li>)\n(?!<)/g, "<br>");

  // Clean up empty <p> tags
  html = html.replace(/<p>\s*<\/p>/g, "");

  return html;
}

function showToast(msg) {
  let toast = document.getElementById("voice-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "voice-toast";
    toast.className = "voice-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

// ════════════════════════════════════════════════════════════════════════════
//  VOICE INTERACTION (STT & TTS)
// ════════════════════════════════════════════════════════════════════════════
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognition && $btnMic) {
  try {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isListening = true;
      $btnMic.classList.add("listening");
      $btnMic.setAttribute("title", "Stop voice input");
      $btnMic.setAttribute("aria-label", "Stop voice input");
      $input.placeholder = "Listening… speak your question";
    };

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      if (transcript) {
        $input.value = transcript;
        $input.style.height = "auto";
        $input.style.height = Math.min($input.scrollHeight, 180) + "px";
        $sendBtn.disabled = $input.value.trim().length === 0;
      }
    };

    recognition.onerror = (event) => {
      isListening = false;
      stopListeningUI();
      if (event.error === "no-speech") {
        showToast("No speech detected. Please try speaking again.");
      } else if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        showToast("Microphone access denied. Please check your browser permissions.");
      } else if (event.error !== "aborted") {
        showToast(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      isListening = false;
      stopListeningUI();
    };
  } catch (e) {
    console.warn("SpeechRecognition initialization failed", e);
  }
}

function stopListeningUI() {
  if ($btnMic) {
    $btnMic.classList.remove("listening");
    $btnMic.setAttribute("title", "Start voice input");
    $btnMic.setAttribute("aria-label", "Start voice input");
  }
  if ($input) {
    $input.placeholder = "Ask about medications, dosages, contraindications, drug interactions, or prescribing guidelines…";
  }
}

if ($btnMic) {
  $btnMic.addEventListener("click", () => {
    if (!SpeechRecognition || !recognition) {
      showToast("Speech recognition is not supported in this browser.");
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (e) {
        recognition.stop();
      }
    }
  });
}

// ── Text-to-Speech (TTS) ──
let currentlySpeakingBtn = null;

function cleanTextForSpeech(rawText) {
  if (!rawText) return "";
  let text = rawText;
  text = text.replace(/\[(?:chunk_\d+|§[^\]]+)[^\]]*\]/gi, "");
  text = text.replace(/[*_#`~>]/g, "");
  text = text.replace(/<[^>]*>/g, "");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

function stopSpeech() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  resetSpeakingUI();
}

function resetSpeakingUI() {
  if (currentlySpeakingBtn) {
    currentlySpeakingBtn.classList.remove("speaking");
    currentlySpeakingBtn.setAttribute("title", "Read response aloud");
    currentlySpeakingBtn.setAttribute("aria-label", "Read response aloud");
    currentlySpeakingBtn = null;
  }
}

function speakResponse(rawText, btnEl) {
  if (!("speechSynthesis" in window)) {
    showToast("Text-to-speech is unavailable in this browser.");
    return;
  }

  if (currentlySpeakingBtn === btnEl && window.speechSynthesis.speaking) {
    stopSpeech();
    return;
  }

  stopSpeech();

  const spokenText = cleanTextForSpeech(rawText);
  if (!spokenText) return;

  const utterance = new SpeechSynthesisUtterance(spokenText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.lang = "en-US";

  currentlySpeakingBtn = btnEl;
  btnEl.classList.add("speaking");
  btnEl.setAttribute("title", "Stop reading");
  btnEl.setAttribute("aria-label", "Stop reading");

  utterance.onend = () => {
    resetSpeakingUI();
  };

  utterance.onerror = (e) => {
    if (e.error !== "interrupted" && e.error !== "canceled") {
      showToast("Speech synthesis error occurred.");
    }
    resetSpeakingUI();
  };

  window.speechSynthesis.speak(utterance);
}

// ════════════════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ════════════════════════════════════════════════════════════════════════════
async function init() {
  // Sync chat sessions and indexed documents from backend
  await fetchChatSessions();
  await fetchIndexedDrugs();

  // Show active session if available, else welcome
  if (state.sessions.length > 0) {
    loadSessionHistory(state.sessions[0].id);
  } else {
    showWelcome();
  }

  $input.focus();
}

// ════════════════════════════════════════════════════════════════════════════
//  LOGOUT
// ════════════════════════════════════════════════════════════════════════════
document.getElementById("btn-logout").addEventListener("click", () => {
  localStorage.removeItem("drugbot_token");
  localStorage.removeItem("drugbot_user_email");
  window.location.href = "/frontend/login.html";
});

init();
