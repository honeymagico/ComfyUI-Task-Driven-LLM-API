/* ---------------------------------------------------------------------------
 * Task-Driven LLM API -- browser extension
 *
 * Two jobs:
 *
 *  1. Node affordances: a "提示詞庫" button, an on-node preview of the last
 *     LLM answer, and socket labels that show which picture each IMAGE slot
 *     is meant to carry for the selected TASK.
 *
 *  2. The prompt library editor: a two-pane panel that reads and writes the
 *     same JSON the node executes, with a live preview of the exact message
 *     list the model will receive.
 *
 * The editor never reimplements the payload rules -- the preview comes from
 * ``/task_llm/preview``, i.e. from the very code that builds the request at
 * execution time.  What the panel shows is what gets sent.
 * ------------------------------------------------------------------------ */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* --------------------------------------------------------------------------
 * Constants
 * ----------------------------------------------------------------------- */

const NODE_CLASS = "TaskDrivenLLM";
const API = "/task_llm";
const IMAGE_SLOTS = ["IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4", "IMAGE5", "IMAGE6"];
const DELIVERY_MODES = ["逐張標註", "集中單一訊息"];
const THINK_CHOICES = ["關閉", "自動", "開啟"];
const PROVIDER_IDS = ["ollama", "openai_compatible"];

/** Names of the widgets declared by nodes.py. */
const W = {
  task: "task",
  provider: "provider",
  model: "model",
  modelCustom: "model_custom",
  userPrompt: "user_prompt",
  systemOverride: "system_override",
  delivery: "image_delivery",
  think: "think",
  temperature: "temperature",
  numPredict: "num_predict",
  seed: "seed",
  imageMaxSide: "image_max_side",
  imageFormat: "image_format",
  keepAlive: "keep_alive",
  timeout: "timeout",
  debug: "debug",
};

const RESULT_WIDGET = "LLM 輸出";
const LIBRARY_BUTTON = "提示詞庫";

const STYLE_ID = "tdl-styles";
const CSS_URL = new URL("./task_llm.css", import.meta.url).href;

/* --------------------------------------------------------------------------
 * Icons -- authored SVG, one stroke weight, no glyph stand-ins
 * ----------------------------------------------------------------------- */

const ICONS = {
  library:
    '<path d="M4 5.5V19"/><path d="M9 5.5V19"/><path d="M14 5.5v13.5"/><path d="M14 5.9l5 1.2V20l-5-1.2"/>',
  plus: '<path d="M12 5v14"/><path d="M5 12h14"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 5H6a2 2 0 0 0-2 2v9"/>',
  trash: '<path d="M4 7h16"/><path d="M10 7V4.5h4V7"/><path d="M6.5 7l1 12.5h9L17.5 7"/>',
  up: '<path d="M12 19V5"/><path d="M6 11l6-6 6 6"/>',
  down: '<path d="M12 5v14"/><path d="M6 13l6 6 6-6"/>',
  save: '<path d="M5 4h11l3 3v13H5z"/><path d="M9 4v6h6V4"/><path d="M8 20v-6h8v6"/>',
  folder: '<path d="M3.5 6.5h5l2 2.2h10V19a1 1 0 0 1-1 1h-15a1 1 0 0 1-1-1z"/>',
  download: '<path d="M12 4v10"/><path d="M7.5 10l4.5 4.5L16.5 10"/><path d="M5 19.5h14"/>',
  upload: '<path d="M12 19.5V9.5"/><path d="M7.5 13.5L12 9l4.5 4.5"/><path d="M5 4.5h14"/>',
  refresh: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.8 4.2v4.2h-4.2"/>',
  plug: '<path d="M9 3.5V9"/><path d="M15 3.5V9"/><path d="M6 9h12v2.6a6 6 0 0 1-12 0z"/><path d="M12 17.6v3"/>',
  search: '<circle cx="11" cy="11" r="6"/><path d="M15.6 15.6L20 20"/>',
  close: '<path d="M6.5 6.5l11 11"/><path d="M17.5 6.5l-11 11"/>',
  warn: '<path d="M12 4.5l8.5 15h-17z"/><path d="M12 10v4.2"/><path d="M12 16.9v.1"/>',
  layers: '<path d="M12 3.5l8 4.3-8 4.3-8-4.3z"/><path d="M4 12.4l8 4.3 8-4.3"/>',
  sliders: '<path d="M4 7.5h7"/><path d="M15 7.5h5"/><path d="M4 16.5h4"/><path d="M12 16.5h8"/><path d="M13.5 7.5a2 2 0 1 0-4 0 2 2 0 0 0 4 0z"/><path d="M10.5 16.5a2 2 0 1 0-4 0 2 2 0 0 0 4 0z"/>',
  globe: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17"/><path d="M12 3.5c2.4 2.5 2.4 14.5 0 17"/><path d="M12 3.5c-2.4 2.5-2.4 14.5 0 17"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  chevron: '<path d="M9 6l6 6-6 6"/>',
  empty: '<path d="M4 6.5h6l1.6 2h8.4V18a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1z"/><path d="M8 12.5h8"/><path d="M8 15.5h5"/>',
  bolt: '<path d="M13.5 3L6 13.5h5L10.5 21 18 10.5h-5z"/>',
};

function icon(name, size = 16) {
  const body = ICONS[name] || ICONS.warn;
  return (
    `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" ` +
    `stroke="currentColor" stroke-width="1.6" stroke-linecap="round" ` +
    `stroke-linejoin="round" aria-hidden="true">${body}</svg>`
  );
}

/* --------------------------------------------------------------------------
 * Tiny hyperscript helper (keeps a file this size readable)
 * ----------------------------------------------------------------------- */

const PROP_KEYS = new Set(["value", "checked", "disabled", "indeterminate", "selectedIndex", "textContent"]);

function appendChildren(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === true) continue;
    parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function h(tag, props = null, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "html") el.innerHTML = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style") Object.assign(el.style, value);
    else if (PROP_KEYS.has(key)) el[key] = value;
    else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  appendChildren(el, children);
  return el;
}

function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Highlight the 【IMAGEn】 markers so the tagging scheme is visible at a glance. */
function highlightTags(text) {
  return escapeHtml(text).replace(/【IMAGE\d+(?:（[^）]*）)?】/g, "<b>$&</b>");
}

/* --------------------------------------------------------------------------
 * HTTP helpers
 * ----------------------------------------------------------------------- */

async function requestJson(path, options = {}) {
  const response = await api.fetchApi(path, { cache: "no-store", ...options });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(body?.error || `${path} 回應 HTTP ${response.status}`);
  }
  return body;
}

const apiGet = (path) => requestJson(path);
const apiPost = (path, payload) =>
  requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });

/* --------------------------------------------------------------------------
 * Library cache
 * ----------------------------------------------------------------------- */

const libraryCache = {
  data: null,
  promise: null,
};

async function getLibrary(force = false) {
  if (!force && libraryCache.data) return libraryCache.data;
  if (!force && libraryCache.promise) return libraryCache.promise;
  libraryCache.promise = apiGet(`${API}/library`)
    .then((data) => {
      libraryCache.data = data;
      return data;
    })
    .finally(() => {
      libraryCache.promise = null;
    });
  return libraryCache.promise;
}

/** Find a task by the exact dropdown label, then by id. */
function findTask(taskLabel, library = libraryCache.data?.library) {
  if (!taskLabel || !library?.tasks) return null;
  return (
    library.tasks.find((task) => labelOf(task) === taskLabel) ||
    library.tasks.find((task) => task.id === taskLabel) ||
    null
  );
}

function labelOf(task) {
  return task.category ? `${task.category} · ${task.title}` : task.title;
}

function nodeWidget(node, name) {
  return node?.widgets?.find((widget) => widget.name === name);
}

function widgetValue(node, name, fallback = "") {
  const widget = nodeWidget(node, name);
  return widget ? widget.value : fallback;
}

function connectedSlots(node) {
  const slots = [];
  for (const input of node?.inputs || []) {
    if (!IMAGE_SLOTS.includes(input.name)) continue;
    if (input.link !== null && input.link !== undefined) slots.push(input.name);
  }
  return slots;
}

/* --------------------------------------------------------------------------
 * Node-side behaviour
 * ----------------------------------------------------------------------- */

/** Show each socket's role for the selected TASK, e.g. "IMAGE1 · 景". */
function applySlotLabels(node) {
  const task = findTask(widgetValue(node, W.task));
  let changed = false;
  for (const input of node.inputs || []) {
    if (!IMAGE_SLOTS.includes(input.name)) continue;
    const role = task?.image_roles?.[input.name];
    const label = role ? `${input.name} · ${role}` : input.name;
    if (input.localized_name !== label) {
      // localized_name only affects drawing and is stripped before the prompt
      // is serialised, so it can never reach the backend payload.
      input.localized_name = label;
      changed = true;
    }
  }
  if (changed) {
    node.setSize?.(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
  }
}

function setOutput(node, state, text = "") {
  node.__tdlOutput = { state, text, at: Date.now() };
  app.graph.setDirtyCanvas(true, true);
}

const RESULT_LINES = 5;
const RESULT_LINE_H = 16;
// Header line + N output lines, plus padding. Shared by computeSize and draw so
// the reserved space and the painted box always agree.
const RESULT_H = RESULT_LINE_H * (RESULT_LINES + 1) + 16;

function drawResultWidget(node, ctx, width, y, height) {
  const out = node.__tdlOutput;
  const pad = 10;
  const inner = width - pad * 2;
  const boxHeight = Math.max(height, RESULT_H) - 6;
  const state = out?.state || "idle";

  ctx.save();
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(pad, y + 3, inner, boxHeight, 6);
  else ctx.rect(pad, y + 3, inner, boxHeight);
  ctx.fillStyle = "rgba(127,127,127,0.14)";
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";

  const clip = (text, max) => {
    if (ctx.measureText(text).width <= max) return text;
    let cut = text;
    while (cut.length > 1 && ctx.measureText(cut + "…").width > max) cut = cut.slice(0, -1);
    return cut + "…";
  };

  const wrap = (text, max) => {
    const lines = [];
    let line = "";
    for (const char of text) {
      if (char === "\n") {
        lines.push(line);
        line = "";
      } else if (line && ctx.measureText(line + char).width > max) {
        lines.push(line);
        line = char;
      } else {
        line += char;
      }
      if (lines.length > RESULT_LINES) break;
    }
    if (line && lines.length <= RESULT_LINES) lines.push(line);
    return lines.slice(0, RESULT_LINES);
  };

  let cursor = y + 9;
  if (state === "idle") {
    ctx.fillStyle = "rgba(140,140,140,0.95)";
    ctx.fillText(clip("執行後，LLM 的回覆會顯示在這裡。", inner - 16), pad + 8, cursor);
  } else if (state === "error") {
    ctx.fillStyle = "#ef7f78";
    ctx.fillText(clip("執行失敗", inner - 16), pad + 8, cursor);
    cursor += RESULT_LINE_H;
    ctx.fillStyle = "#e8a5a0";
    for (const line of wrap(String(out.text || "").replace(/\s+/g, " "), inner - 16)) {
      ctx.fillText(line, pad + 8, cursor);
      cursor += RESULT_LINE_H;
    }
  } else {
    ctx.fillStyle = "rgba(140,140,140,0.95)";
    ctx.fillText(clip(out.header || "", inner - 16), pad + 8, cursor);
    cursor += RESULT_LINE_H;
    ctx.fillStyle = "rgba(226,226,226,0.95)";
    for (const line of wrap(String(out.text || "").replace(/\n{2,}/g, "\n"), inner - 16)) {
      ctx.fillText(line, pad + 8, cursor);
      cursor += RESULT_LINE_H;
    }
  }
  ctx.restore();
}

const formatSeconds = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

function showExecutionResult(node, output) {
  const text = Array.isArray(output?.text) ? output.text[0] : output?.text;
  const rawMeta = Array.isArray(output?.meta) ? output.meta[0] : output?.meta;
  let meta = null;
  if (typeof rawMeta === "string") {
    try {
      meta = JSON.parse(rawMeta);
    } catch {
      meta = null;
    }
  }
  const label = meta?.task?.title
    ? meta.task.category
      ? `${meta.task.category} · ${meta.task.title}`
      : meta.task.title
    : widgetValue(node, W.task);
  const model = meta?.model || widgetValue(node, W.model);
  const images = meta?.images?.length ?? connectedSlots(node).length;
  const parts = [
    label,
    meta?.provider ? `${meta.provider}/${model}` : model,
    images ? `${images} 張圖` : "無圖",
  ];
  if (meta?.duration_ms) parts.push(formatSeconds(meta.duration_ms));

  node.__tdlOutput = {
    state: "ok",
    text: text ?? "",
    header: parts.filter(Boolean).join("  ·  "),
    at: Date.now(),
  };
  app.graph.setDirtyCanvas(true, true);
}

function addNodeWidgets(node) {
  if (nodeWidget(node, LIBRARY_BUTTON)) return;

  const button = node.addWidget("button", LIBRARY_BUTTON, null, () => {
    openLibraryEditor(node);
  });
  if (button) {
    button.serialize = false;
    button.tooltip = "新增／編輯提示詞庫裡的 TASK，並設定各圖片接口的角色。";
  }

  // A "custom" widget whose type is unknown resolves to a LegacyWidget, which
  // keeps our own draw/computeSize/serialize fields intact.
  const result = node.addCustomWidget({
    type: "tdl_output_preview",
    name: RESULT_WIDGET,
    value: "",
    serialize: false,
    options: { serialize: false },
    computedHeight: RESULT_H,
    computeSize: (width) => [width, RESULT_H],
    draw: (ctx, widgetNode, width, y, height) => drawResultWidget(widgetNode, ctx, width, y, height),
    tooltip: "上一次執行時 LLM 回覆的內容。",
  });
  if (result && !result.tooltip) result.tooltip = "上一次執行時 LLM 回覆的內容。";

  node.__tdlOutput = { state: "idle", text: "" };
  node.setSize?.(node.computeSize());
}

/* --------------------------------------------------------------------------
 * Editor panel
 * ----------------------------------------------------------------------- */

let openEditor = null;

function openLibraryEditor(node) {
  if (openEditor) {
    openEditor.focus();
    return;
  }
  openEditor = new LibraryEditor(node);
  openEditor.open().catch((error) => {
    console.error("[Task-Driven LLM] 無法開啟提示詞庫編輯器", error);
    alert(`無法開啟提示詞庫編輯器：\n${error.message || error}`);
    openEditor = null;
  });
}

class LibraryEditor {
  constructor(node) {
    this.node = node;
    this.mode = "tasks";
    this.state = {
      library: null,
      savedLibrary: "",
      config: null,
      savedConfig: "",
      locate: null,
      warnings: [],
      selectedId: null,
      query: "",
      status: { kind: "idle", text: "" },
      busy: false,
      confirmingClose: false,
      preview: { loading: false, error: "", transcript: "", notes: [], usedSlots: [] },
    };
    this.previewTimer = null;
  }

  /* -- lifecycle --------------------------------------------------------- */

  async open() {
    this.build();
    document.body.appendChild(this.overlay);
    // Bubble phase on the panel itself: handling keys here and then stopping
    // propagation keeps ComfyUI's window-level shortcuts (prompt weight edit,
    // node delete, ...) from acting on the graph while a field has focus.
    this.overlay.addEventListener("keydown", this.onKeyDown);
    await this.reload();
    this.focus();
  }

  focus() {
    this.overlay?.querySelector("input, select, textarea, button")?.focus();
  }

  close(force = false) {
    if (!force && (this.isLibraryDirty() || this.isConfigDirty())) {
      this.state.confirmingClose = true;
      this.renderHeader();
      this.renderFooter();
      return;
    }
    clearTimeout(this.previewTimer);
    this.overlay.remove();
    openEditor = null;
  }

  onKeyDown = (event) => {
    try {
      if (event.key === "Escape") {
        event.preventDefault();
        if (this.state.confirmingClose) {
          this.state.confirmingClose = false;
          this.renderFooter();
        } else {
          this.close();
        }
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        this.save();
        return;
      }
      if (event.altKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
        event.preventDefault();
        this.moveTask(event.key === "ArrowUp" ? -1 : 1);
        return;
      }
      if (event.key === "Tab") this.trapFocus(event);
    } finally {
      event.stopPropagation();
    }
  };

  trapFocus(event) {
    const focusable = [
      ...this.overlay.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ),
    ].filter((el) => el.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /* -- data -------------------------------------------------------------- */

  async reload(force = false) {
    this.setStatus("busy", "載入提示詞庫…");
    const [library, config] = await Promise.all([getLibrary(force), apiGet(`${API}/config`)]);
    this.state.library = JSON.parse(JSON.stringify(library.library));
    this.state.savedLibrary = JSON.stringify(this.state.library);
    this.state.locate = library.locate;
    this.state.warnings = library.warnings || [];
    this.state.config = config.config;
    this.state.savedConfig = JSON.stringify(this.editableConfig());
    if (!this.currentTask() || !this.state.library.tasks.some((t) => t.id === this.state.selectedId)) {
      this.state.selectedId = this.state.library.tasks[0]?.id ?? null;
    }
    this.renderAll();
    this.schedulePreview(0);
    this.setStatus(
      this.state.warnings.length ? "warn" : "idle",
      this.state.warnings.length ? this.state.warnings[0] : `共 ${this.state.library.tasks.length} 個 TASK`
    );
  }

  editableConfig() {
    const config = this.state.config;
    if (!config) return {};
    const providers = {};
    for (const id of Object.keys(config.providers || {})) {
      const entry = config.providers[id];
      providers[id] = {
        base_url: entry.base_url ?? "",
        default_model: entry.default_model ?? "",
        keep_alive: entry.keep_alive ?? "",
        timeout: entry.timeout ?? 300,
      };
    }
    return { image_quality: config.image_quality ?? 90, providers };
  }

  isLibraryDirty() {
    return JSON.stringify(this.state.library) !== this.state.savedLibrary;
  }

  isConfigDirty() {
    return JSON.stringify(this.editableConfig()) !== this.state.savedConfig;
  }

  currentTask() {
    const tasks = this.state.library?.tasks || [];
    return tasks.find((task) => task.id === this.state.selectedId) || null;
  }

  setStatus(kind, text) {
    this.state.status = { kind, text };
    this.renderHeader();
    this.renderFooter();
  }

  setBusy(busy, label) {
    this.state.busy = busy;
    this.renderFooter();
    if (busy) this.setStatus("busy", label || "處理中…");
  }

  /* -- DOM skeleton ------------------------------------------------------ */

  build() {
    this.overlay = h("div", {
      class: "tdl-root tdl-overlay",
      role: "dialog",
      "aria-modal": "true",
      "aria-label": "提示詞庫",
      onmousedown: (event) => {
        if (event.target === this.overlay) this.close();
      },
    });

    this.mark = h("span", { class: "tdl-head__mark", html: icon("library", 17) });
    this.titleEl = h("h2", { class: "tdl-head__title", textContent: "提示詞庫" });
    this.pathEl = h("span", { class: "tdl-head__path" });
    this.statusEl = h("span", { class: "tdl-head__status" });
    this.segEl = h("div", { class: "tdl-seg" });

    this.panel = h(
      "div",
      { class: "tdl-panel", role: "document" },
      h(
        "header",
        { class: "tdl-head" },
        this.mark,
        h("div", { class: "tdl-head__titles" }, this.titleEl, this.pathEl),
        h("div", { class: "tdl-head__spacer" }),
        this.statusEl,
        this.segEl,
        h("button", {
          class: "tdl-iconbtn",
          title: "關閉（Esc）",
          "aria-label": "關閉",
          html: icon("close", 16),
          onclick: () => this.close(),
        })
      ),
      (this.bodyEl = h("div", { class: "tdl-body" })),
      (this.footEl = h("footer", { class: "tdl-foot" }))
    );

    this.overlay.appendChild(this.panel);
    this.overlay.addEventListener("mousedown", (event) => event.stopPropagation());
    this.renderHeader();
  }

  renderAll() {
    this.renderHeader();
    this.renderBody();
    this.renderFooter();
  }

  /* -- header ------------------------------------------------------------ */

  renderHeader() {
    const locate = this.state.locate;
    this.pathEl.textContent = locate?.tasks_path || "";
    this.pathEl.title = locate?.tasks_path || "";

    clear(this.segEl);
    const tabs = [
      { id: "tasks", label: "TASK 編輯器", icon: "sliders" },
      { id: "config", label: "連線設定", icon: "globe" },
    ];
    for (const tab of tabs) {
      this.segEl.appendChild(
        h(
          "button",
          {
            class: `tdl-seg__btn${this.mode === tab.id ? " is-active" : ""}`,
            type: "button",
            "aria-pressed": this.mode === tab.id,
            onclick: () => this.setMode(tab.id),
          },
          h("span", { html: icon(tab.icon, 14) }),
          tab.label
        )
      );
    }

    clear(this.statusEl);
    const dirty = this.isLibraryDirty() || this.isConfigDirty();
    if (dirty) this.statusEl.appendChild(h("span", { class: "tdl-dirty", textContent: "未儲存的變更" }));
    const { kind, text } = this.state.status;
    if (text) {
      if (dirty) this.statusEl.appendChild(h("span", { textContent: "·" }));
      this.statusEl.dataset.kind = kind;
      this.statusEl.appendChild(h("span", { textContent: text }));
    } else {
      delete this.statusEl.dataset.kind;
    }
  }

  setMode(mode) {
    if (this.mode === mode) return;
    this.mode = mode;
    this.state.confirmingClose = false;
    this.renderAll();
    if (mode === "tasks") this.schedulePreview(0);
  }

  /* -- body -------------------------------------------------------------- */

  renderBody() {
    clear(this.bodyEl);
    if (!this.state.library) return;
    if (this.mode === "config") {
      this.bodyEl.appendChild(this.buildConfigPane());
      return;
    }
    if (!this.state.library.tasks.length) {
      // One empty state for an empty library, not two competing ones: with
      // nothing to list there is no point showing an empty list *and* an empty
      // editor side by side.
      this.bodyEl.appendChild(this.buildEmptyLibraryPane());
      return;
    }
    this.bodyEl.appendChild(this.buildListPane());
    this.bodyEl.appendChild(this.buildTaskPane());
  }

  buildEmptyLibraryPane() {
    return h(
      "section",
      { class: "tdl-pane" },
      this.buildEmptyState({
        title: "提示詞庫是空的",
        body: "新增一個 TASK，或還原內建範例（多圖參考合成、生活化抓拍照片…）開始使用。",
        primary: { label: "新增 TASK", icon: "plus", onClick: () => this.addTask() },
        secondary: { label: "還原內建範例", icon: "refresh", onClick: () => this.restoreDefaults() },
      })
    );
  }

  /** Replace only the task list.
   *
   * In-form edits (title, category, image roles, ...) need the list to follow
   * them, but rebuilding the whole body would destroy the very input the user
   * is typing into -- losing focus, caret position and scroll.  Swapping just
   * the list subtree keeps the form untouched.
   */
  renderList() {
    if (!this.state.library || this.mode !== "tasks") return;
    const scroll = this.bodyEl.querySelector(".tdl-list__scroll");
    const scrollTop = scroll?.scrollTop ?? 0;
    const current = this.bodyEl.querySelector(".tdl-list");
    const next = this.buildListPane();
    if (current) current.replaceWith(next);
    else this.bodyEl.prepend(next);
    const replaced = this.bodyEl.querySelector(".tdl-list__scroll");
    if (replaced) replaced.scrollTop = scrollTop;
  }

  /* -- task list --------------------------------------------------------- */

  buildListPane() {
    const tasks = this.state.library.tasks || [];
    const query = this.state.query.trim().toLowerCase();
    const matches = (task) =>
      !query ||
      `${labelOf(task)} ${task.description || ""} ${task.prompt || ""}`.toLowerCase().includes(query);

    const groups = new Map();
    for (const task of tasks) {
      if (!matches(task)) continue;
      const key = task.category || "未分類";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(task);
    }

    const scroll = h("div", { class: "tdl-list__scroll" });
    if (!tasks.length) {
      scroll.appendChild(
        this.buildEmptyState({
          title: "提示詞庫是空的",
          body: "新增一個 TASK，或還原內建範例開始使用。",
          primary: { label: "新增 TASK", icon: "plus", onClick: () => this.addTask() },
          secondary: { label: "還原內建範例", icon: "refresh", onClick: () => this.restoreDefaults() },
        })
      );
    } else if (!groups.size) {
      scroll.appendChild(
        this.buildEmptyState({
          title: "沒有符合的 TASK",
          body: `找不到包含「${this.state.query}」的任務，換個關鍵字試試。`,
          primary: { label: "清除搜尋", icon: "close", onClick: () => this.setQuery("") },
        })
      );
    } else {
      for (const [category, items] of groups) {
        scroll.appendChild(h("div", { class: "tdl-group__label", textContent: category }));
        for (const task of items) {
          const slots = Object.keys(task.image_roles || {}).length;
          scroll.appendChild(
            h(
              "button",
              {
                class: `tdl-item${task.id === this.state.selectedId ? " is-active" : ""}${
                  task.enabled === false ? " is-off" : ""
                }`,
                type: "button",
                "aria-current": task.id === this.state.selectedId,
                onclick: () => this.selectTask(task.id),
                ondblclick: () => this.focusField("title"),
              },
              h("input", {
                type: "checkbox",
                checked: task.enabled !== false,
                title: task.enabled === false ? "已停用（不會出現在下拉選單）" : "啟用中",
                "aria-label": "啟用這個 TASK",
                onclick: (event) => {
                  event.stopPropagation();
                  task.enabled = event.target.checked;
                  this.touch();
                  this.renderList();
                },
              }),
              h("span", { class: "tdl-item__title", textContent: task.title || "（未命名）" }),
              h("span", { class: "tdl-item__meta", textContent: slots ? `${slots} 圖` : "無圖" })
            )
          );
        }
      }
    }

    const search = h("input", {
      type: "search",
      placeholder: "搜尋 TASK…",
      value: this.state.query,
      "aria-label": "搜尋 TASK",
      oninput: (event) => this.setQuery(event.target.value),
    });

    return h(
      "aside",
      { class: "tdl-list" },
      h("div", { class: "tdl-list__search" }, h("span", { html: icon("search", 14) }), search),
      scroll,
      h(
        "div",
        { class: "tdl-list__foot" },
        h("button", {
          class: "tdl-btn tdl-btn--block",
          type: "button",
          onclick: () => this.addTask(),
        }, h("span", { html: icon("plus", 15) }), "新增 TASK")
      )
    );
  }

  buildEmptyState({ title, body, primary, secondary }) {
    return h(
      "div",
      { class: "tdl-empty" },
      h("span", { html: icon("empty", 34) }),
      h("h3", { textContent: title }),
      h("p", { textContent: body }),
      h(
        "div",
        { class: "tdl-empty__actions" },
        primary &&
          h("button", { class: "tdl-btn tdl-btn--primary", type: "button", onclick: primary.onClick },
            h("span", { html: icon(primary.icon, 15) }), primary.label),
        secondary &&
          h("button", { class: "tdl-btn", type: "button", onclick: secondary.onClick },
            h("span", { html: icon(secondary.icon, 15) }), secondary.label)
      )
    );
  }

  setQuery(value) {
    this.state.query = value;
    this.renderList();
    const input = this.bodyEl.querySelector(".tdl-list__search input");
    if (input) {
      input.focus();
      input.selectionStart = input.value.length;
      input.selectionEnd = input.value.length;
    }
  }

  selectTask(id) {
    this.state.selectedId = id;
    this.state.preview = { loading: false, error: "", transcript: "", notes: [], usedSlots: [] };
    this.renderBody();
    this.renderFooter();
    this.schedulePreview(0);
  }

  focusField(key) {
    this.bodyEl.querySelector(`[data-field="${key}"]`)?.focus();
  }

  /* -- task form --------------------------------------------------------- */

  buildTaskPane() {
    const task = this.currentTask();
    if (!task) {
      return h(
        "section",
        { class: "tdl-pane" },
        this.buildEmptyState({
          title: "尚未選擇 TASK",
          body: "從左邊挑一個 TASK 來編輯，或新增一個。",
          primary: { label: "新增 TASK", icon: "plus", onClick: () => this.addTask() },
        })
      );
    }

    const pane = h("section", { class: "tdl-pane" });
    pane.appendChild(this.buildBasics(task));
    pane.appendChild(this.buildPrompts(task));
    pane.appendChild(this.buildImageRoles(task));
    pane.appendChild(this.buildParameters(task));
    pane.appendChild(this.buildPreview(task));
    return pane;
  }

  section(title, ...children) {
    return h(
      "div",
      { class: "tdl-section" },
      h("h3", { class: "tdl-section__head" }, title),
      ...children
    );
  }

  field(label, control, { hint, optional, error, key } = {}) {
    return h(
      "div",
      { class: "tdl-field" },
      h("label", { for: key, dataset: key ? { field: key } : null },
        label, optional ? h("span", { class: "tdl-field__opt", textContent: optional }) : null),
      control,
      hint ? h("p", { class: "tdl-hint", html: hint }) : null,
      error ? h("span", { class: "tdl-field__error", textContent: error }) : null
    );
  }

  textInput(value, onChange, extra = {}) {
    return h("input", {
      type: "text",
      value: value ?? "",
      ...extra,
      oninput: (event) => {
        onChange(event.target.value);
      },
    });
  }

  textArea(value, onChange, extra = {}) {
    return h("textarea", {
      value: value ?? "",
      spellcheck: false,
      ...extra,
      oninput: (event) => onChange(event.target.value),
    });
  }

  buildBasics(task) {
    const titleError = task.title?.trim() ? null : "標題不能是空的，下拉選單要靠它辨識。";
    return this.section(
      "基本資料",
      h(
        "div",
        { class: "tdl-grid tdl-grid--3" },
        this.field(
          "標題",
          this.textInput(task.title, (value) => {
            task.title = value;
            this.touch({ list: true, title: true });
          }, { id: "tdl-title", "data-field": "title", "aria-invalid": titleError ? "true" : null }),
          { key: "title", error: titleError }
        ),
        this.field(
          "類別",
          this.textInput(task.category, (value) => {
            task.category = value;
            this.touch({ list: true });
          }, { placeholder: "例如：提示詞生成" }),
          { optional: "可留空", key: "category" }
        ),
        this.field(
          "啟用",
          h(
            "label",
            { class: "tdl-check", style: { height: "32px" } },
            h("input", {
              type: "checkbox",
              checked: task.enabled !== false,
              onchange: (event) => {
                task.enabled = event.target.checked;
                this.touch({ list: true });
              },
            }),
            task.enabled === false ? "已停用" : "出現在節點下拉選單"
          ),
          { key: "enabled" }
        )
      ),
      this.field(
        "說明",
        this.textInput(task.description, (value) => {
          task.description = value;
        }, { placeholder: "一句話說明這個 TASK 做什麼（只顯示在編輯器裡）" }),
        { optional: "可留空", key: "description" }
      )
    );
  }

  buildPrompts(task) {
    return this.section(
      "提示詞",
      h(
        "div",
        { class: "tdl-field" },
        h(
          "label",
          { for: "tdl-system" },
          "系統提示詞",
          h("span", { class: "tdl-field__opt", textContent: "留空＝使用全域預設（含圖片對照約定）" })
        ),
        this.textArea(task.system, (value) => {
          task.system = value;
        }, { id: "tdl-system", class: "tdl-mono", rows: 5, placeholder: "留空即可；填寫則整段取代全域預設。" })
      ),
      this.field(
        "任務提示詞",
        this.textArea(task.prompt, (value) => {
          task.prompt = value;
          this.schedulePreview();
        }, {
          id: "tdl-prompt",
          class: "tdl-mono",
          rows: 10,
          "aria-invalid": task.prompt?.trim() ? null : "true",
          placeholder: "直接引用接口名稱，例如：\n請參考 IMAGE1 的景、IMAGE2 的人、IMAGE3 的色調，合成一條生圖提示詞。",
        }),
        {
          key: "prompt",
          error: task.prompt?.trim() ? null : "沒有內容的話，執行時只會送出圖片對照表。",
          hint:
            "接口名稱直接寫 <code>IMAGE1</code>…<code>IMAGE6</code>，不必用變數。可用 <code>{{user_prompt}}</code> " +
            "指定節點上臨時指示要插入的位置（不用也可以，屆時會附加在最後）。",
        }
      ),
      h(
        "label",
        { class: "tdl-check", style: { marginTop: "10px" } },
        h("input", {
          type: "checkbox",
          checked: task.append_user_prompt !== false,
          onchange: (event) => {
            task.append_user_prompt = event.target.checked;
            this.touch();
            this.schedulePreview();
          },
        }),
        "把節點上的使用者提示詞附加到最後"
      )
    );
  }

  buildImageRoles(task) {
    const wired = connectedSlots(this.node);
    const rows = IMAGE_SLOTS.map((slot) =>
      h(
        "div",
        { class: "tdl-role" },
        h("span", {
          class: "tdl-role__slot",
          dataset: { connected: wired.includes(slot) ? "true" : "false" },
          title: wired.includes(slot) ? "節點上已連接圖片" : "節點上未連接",
          textContent: slot,
        }),
        this.textInput(task.image_roles?.[slot] ?? "", (value) => {
          if (!task.image_roles) task.image_roles = {};
          if (value.trim()) task.image_roles[slot] = value;
          else delete task.image_roles[slot];
          if (!Object.keys(task.image_roles).length) delete task.image_roles;
          this.touch({ list: true, preview: true });
        }, { placeholder: "例如：景 / 人 / 色調", maxlength: 24 })
      )
    );
    return this.section(
      "圖片接口角色",
      h("div", { class: "tdl-grid tdl-grid--3" }, rows),
      h("p", {
        class: "tdl-hint",
        html:
          "這裡填的角色會顯示在節點插槽名稱上（例如 <code>IMAGE1 · 景</code>），也會出現在送給 LLM 的圖片對照表與 " +
          "<code>【IMAGEn】</code> 標記裡。只填短名詞最好讀。",
      })
    );
  }

  buildParameters(task) {
    const numeric = (key, { min, max, step, placeholder }) =>
      this.takeOrClear(task, key, (value) => {
        task[key] = value;
        this.touch();
        this.schedulePreview();
      }, min, max, step, placeholder);

    const providerOptions = [
      { value: "", label: "（跟隨節點）" },
      ...PROVIDER_IDS.map((id) => ({
        value: id,
        label: this.state.config?.provider_info?.[id]?.label || id,
      })),
    ];

    return this.section(
      "生成參數",
      h(
        "div",
        { class: "tdl-grid tdl-grid--3" },
        this.field("provider", this.select(providerOptions, task.provider ?? "", (value) => {
          if (value) task.provider = value;
          else delete task.provider;
          this.touch();
          this.schedulePreview();
        }), { optional: "可留空" }),
        this.field("model", this.textInput(task.model ?? "", (value) => {
          if (value.trim()) task.model = value;
          else delete task.model;
          this.touch();
          this.schedulePreview();
        }, { placeholder: "留空＝用節點上的模型", class: "tdl-mono" }), { optional: "可留空" }),
        this.field("think", this.select(
          [
            { value: "", label: "（沿用節點）" },
            { value: "false", label: "關閉" },
            { value: "true", label: "開啟" },
          ],
          task.think === undefined || task.think === null ? "" : String(task.think),
          (value) => {
            if (value === "") delete task.think;
            else task.think = value === "true";
            this.touch();
          }
        ), { optional: "可留空" })
      ),
      h(
        "div",
        { class: "tdl-grid tdl-grid--3", style: { marginTop: "12px" } },
        this.field("temperature", numeric("temperature", { min: 0, max: 2, step: 0.05 }), { optional: "0–2" }),
        this.field("num_predict", numeric("num_predict", { min: 0, max: 32768, step: 64 }), { optional: "0＝不限" }),
        this.field("keep_alive", this.textInput(task.keep_alive ?? "", (value) => {
          if (value.trim()) task.keep_alive = value;
          else delete task.keep_alive;
          this.touch();
        }, { placeholder: "例如 5m", class: "tdl-mono" }), { optional: "Ollama 專用" })
      ),
      h("p", {
        class: "tdl-hint",
        html: "留空的欄位會沿用節點上的設定；有填的話以這裡為準（TASK 是預設集，優先權較高）。",
      })
    );
  }

  takeOrClear(task, key, assign, min, max, step, placeholder) {
    return h("input", {
      type: "number",
      value: task[key] === undefined || task[key] === null ? "" : task[key],
      min,
      max,
      step,
      placeholder: placeholder ?? "（沿用節點）",
      oninput: (event) => {
        const raw = event.target.value.trim();
        if (raw === "") delete task[key];
        else {
          const parsed = Number(raw);
          if (!Number.isNaN(parsed)) task[key] = parsed;
        }
        assign(task[key]);
      },
    });
  }

  select(options, value, onChange) {
    const el = h("select", {
      onchange: (event) => {
        onChange(event.target.value);
      },
    });
    for (const option of options) {
      el.appendChild(
        h("option", { value: option.value, selected: option.value === value, textContent: option.label })
      );
    }
    return el;
  }

  buildPreview(task) {
    const wired = connectedSlots(this.node);
    const chips = h(
      "div",
      { class: "tdl-preview__wiring" },
      ...IMAGE_SLOTS.map((slot) => {
        const on = wired.includes(slot);
        return h(
          "span",
          {
            class: "tdl-chip",
            dataset: { on: on ? "true" : "false" },
            title: on ? "節點上已連接" : "節點上未連接",
          },
          h("span", { class: "tdl-chip__dot" }),
          slot
        );
      })
    );

    const pre = h("pre", { class: "tdl-transcript" });
    if (this.state.preview.loading) pre.textContent = "計算中…";
    else if (this.state.preview.error) pre.textContent = this.state.preview.error;
    else if (this.state.preview.transcript) pre.innerHTML = highlightTags(this.state.preview.transcript);
    else pre.textContent = "尚無內容。";

    const notes = h("div", { class: "tdl-preview__notes" }, ...this.previewNotes());

    return this.section(
      "送出內容預覽",
      h(
        "div",
        { class: "tdl-preview" },
        h("p", {
          class: "tdl-hint",
          style: { marginTop: "0" },
          textContent:
            "以下是這個 TASK 搭配節點目前實際連接的圖片後，會送給 LLM 的完整內容（圖片本身不顯示）。",
        }),
        chips,
        pre,
        notes
      ),
      h(
        "div",
        { style: { marginTop: "10px" } },
        h("button", {
          class: "tdl-btn",
          type: "button",
          onclick: () => this.schedulePreview(0),
        }, h("span", { html: icon("refresh", 15) }), "重新整理預覽")
      )
    );
  }

  previewNotes() {
    const notes = [];
    for (const note of this.state.preview.notes || []) {
      notes.push(
        h("div", { class: "tdl-note" }, h("span", { html: icon("warn", 15) }), h("span", { textContent: note }))
      );
    }
    return notes;
  }

  /** Refresh only the preview block, leaving the rest of the form (and any
   * focused input) alone. */
  applyPreviewToDom() {
    const pre = this.bodyEl?.querySelector(".tdl-transcript");
    if (!pre) return;
    const { loading, error, transcript } = this.state.preview;
    if (loading) pre.textContent = "計算中…";
    else if (error) pre.textContent = error;
    else if (transcript) pre.innerHTML = highlightTags(transcript);
    else pre.textContent = "尚無內容。";

    const host = this.bodyEl.querySelector(".tdl-preview__notes");
    if (host) {
      clear(host);
      appendChildren(host, this.previewNotes());
    }
  }

  /* -- preview ----------------------------------------------------------- */

  schedulePreview(delay = 420) {
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => this.runPreview(), delay);
  }

  async runPreview() {
    const task = this.currentTask();
    if (!task || !this.bodyEl.querySelector(".tdl-transcript")) return;
    if (!(task.prompt || "").trim() && !(task.system || "").trim()) {
      this.state.preview = {
        loading: false,
        error: "這個 TASK 還沒有提示詞內容。",
        transcript: "",
        notes: [],
      };
      this.applyPreviewToDom();
      return;
    }

    this.state.preview.loading = true;
    this.applyPreviewToDom();
    try {
      const result = await apiPost(`${API}/preview`, {
        task_body: task,
        default_system: this.state.library.default_system,
        connected: connectedSlots(this.node),
        user_prompt: widgetValue(this.node, W.userPrompt, ""),
        system_override: widgetValue(this.node, W.systemOverride, ""),
        image_delivery: widgetValue(this.node, W.delivery, DELIVERY_MODES[0]),
      });
      this.state.preview = {
        loading: false,
        error: "",
        transcript: result.transcript || "",
        notes: result.notes || [],
        usedSlots: result.used_slots || [],
      };
    } catch (error) {
      this.state.preview = { loading: false, error: error.message || String(error), transcript: "", notes: [] };
    }
    // Update in place: a full re-render here would blow away the caret of
    // whatever field the user is typing in while the preview refreshes.
    this.applyPreviewToDom();
  }

  /* -- mutations --------------------------------------------------------- */

  touch({ list = false, title = false, preview = false } = {}) {
    this.renderHeader();
    this.renderFooter();
    if (list) this.renderList();
    else if (title) {
      const active = this.bodyEl.querySelector(".tdl-item.is-active .tdl-item__title");
      if (active) active.textContent = this.currentTask()?.title || "（未命名）";
    }
    if (preview) this.schedulePreview();
  }

  addTask() {
    const base = "新 TASK";
    let index = 1;
    const titles = new Set((this.state.library.tasks || []).map((task) => task.title));
    while (titles.has(index === 1 ? base : `${base} ${index}`)) index += 1;
    const task = {
      id: `task-${Date.now().toString(36)}`,
      title: index === 1 ? base : `${base} ${index}`,
      category: "",
      description: "",
      system: "",
      prompt: "",
      append_user_prompt: true,
      enabled: true,
      image_roles: {},
    };
    this.state.library.tasks.push(task);
    this.state.selectedId = task.id;
    this.state.query = "";
    this.renderBody();
    this.touch();
    this.schedulePreview(600);
    requestAnimationFrame(() => {
      const input = this.bodyEl.querySelector('[data-field="title"]');
      input?.focus();
      input?.select?.();
    });
  }

  duplicateTask() {
    const task = this.currentTask();
    if (!task) return;
    const copy = JSON.parse(JSON.stringify(task));
    copy.id = `${task.id}-copy-${Date.now().toString(36)}`;
    copy.title = `${task.title} 複本`;
    const index = this.state.library.tasks.indexOf(task);
    this.state.library.tasks.splice(index + 1, 0, copy);
    this.state.selectedId = copy.id;
    this.renderBody();
    this.touch();
  }

  deleteTask() {
    const task = this.currentTask();
    if (!task) return;
    const index = this.state.library.tasks.indexOf(task);
    this.state.library.tasks.splice(index, 1);
    const next = this.state.library.tasks[Math.min(index, this.state.library.tasks.length - 1)];
    this.state.selectedId = next?.id ?? null;
    this.renderBody();
    this.touch();
  }

  moveTask(delta) {
    const task = this.currentTask();
    if (!task) return;
    const tasks = this.state.library.tasks;
    const index = tasks.indexOf(task);
    const target = index + delta;
    if (target < 0 || target >= tasks.length) return;
    tasks.splice(index, 1);
    tasks.splice(target, 0, task);
    this.renderBody();
    this.touch();
  }

  /* -- config pane ------------------------------------------------------- */

  buildConfigPane() {
    const config = this.state.config;
    const pane = h("section", { class: "tdl-pane" });

    for (const [id, info] of Object.entries(config.provider_info || {})) {
      const entry = config.providers?.[id] || {};
      const overrides = entry.env_override || [];

      const card = h("div", { class: "tdl-card" });
      card.appendChild(
        h(
          "div",
          { class: "tdl-card__head" },
          h(
            "div",
            { style: { flex: "1 1 auto", minWidth: "0" } },
            h("h4", { class: "tdl-card__title", textContent: info.label || id }),
            h("p", { class: "tdl-card__summary", textContent: info.summary || "" })
          ),
          h("span", {
            class: "tdl-badge",
            dataset: { on: overrides.length ? "true" : "false" },
            textContent: overrides.length ? "環境變數覆寫中" : id,
            title: overrides.length ? `以下欄位被環境變數覆寫：${overrides.join(", ")}` : id,
          })
        )
      );

      const fields = h("div", { class: "tdl-grid tdl-grid--2" });
      fields.appendChild(
        this.field(
          "Base URL",
          this.textInput(entry.base_url ?? "", (value) => {
            entry.base_url = value;
            this.configTouched();
          }, { class: "tdl-mono", placeholder: info.base_url_hint || "" }),
          { optional: info.base_url_hint }
        )
      );
      fields.appendChild(
        this.field(
          "API Key",
          h("input", {
            type: "password",
            autocomplete: "off",
            placeholder: entry.api_key_set ? `已設定（${entry.api_key_hint}）— 留空＝不變更` : info.key_hint || "",
            oninput: (event) => {
              // Only send the key when the user actually typed something, so a
              // stored secret is never overwritten by an empty field.
              if (event.target.value === "") delete entry.api_key;
              else entry.api_key = event.target.value;
              this.configTouched();
            },
          }),
          { optional: info.needs_key ? "必填" : "可留空" }
        )
      );
      fields.appendChild(
        this.field(
          "預設模型",
          this.textInput(entry.default_model ?? "", (value) => {
            entry.default_model = value;
            this.configTouched();
          }, { class: "tdl-mono", placeholder: "節點下拉選單的預設值" }),
          { optional: "可留空" }
        )
      );
      fields.appendChild(
        this.field(
          "逾時（秒）",
          h("input", {
            type: "number",
            min: 5,
            max: 3600,
            value: entry.timeout ?? 300,
            oninput: (event) => {
              entry.timeout = Number(event.target.value) || 300;
              this.configTouched();
            },
          })
        )
      );
      if (info.supports_keep_alive) {
        fields.appendChild(
          this.field(
            "keep_alive",
            this.textInput(entry.keep_alive ?? "", (value) => {
              entry.keep_alive = value;
              this.configTouched();
            }, { class: "tdl-mono", placeholder: "例如 5m" })
          )
        );
      }

      const result = h("div", { style: { marginTop: "12px" } });
      const actions = h(
        "div",
        { class: "tdl-card__actions" },
        h("button", {
          class: "tdl-btn",
          type: "button",
          onclick: (event) => this.probeProvider(id, event.currentTarget, result),
        }, h("span", { html: icon("plug", 15) }), "測試連線"),
        h("button", {
          class: "tdl-btn",
          type: "button",
          onclick: (event) => this.fetchModels(id, event.currentTarget, result),
        }, h("span", { html: icon("refresh", 15) }), "重新整理模型清單")
      );

      card.appendChild(fields);
      if (config.base_url_presets?.some((preset) => preset.provider === id)) {
        const presets = h(
          "div",
          { class: "tdl-card__actions" },
          ...config.base_url_presets
            .filter((preset) => preset.provider === id)
            .map((preset) =>
              h("button", {
                class: "tdl-btn tdl-btn--ghost",
                type: "button",
                textContent: preset.label,
                title: preset.base_url,
                onclick: () => {
                  entry.base_url = preset.base_url;
                  this.configTouched();
                  this.renderBody();
                  this.setStatus("idle", `已套用 ${preset.label} 的 Base URL，記得儲存。`);
                },
              })
            )
        );
        card.appendChild(h("div", { class: "tdl-section__head", style: { marginTop: "14px" } }, "快速套用"));
        card.appendChild(presets);
      }
      card.appendChild(actions);
      card.appendChild(result);
      pane.appendChild(card);
    }

    const images = h("div", { class: "tdl-card" });
    images.appendChild(
      h(
        "div",
        { class: "tdl-card__head" },
        h(
          "div",
          {},
          h("h4", { class: "tdl-card__title", textContent: "圖片編碼" }),
          h("p", {
            class: "tdl-card__summary",
            textContent: "送給 LLM 前會先把圖片的長邊縮小；長邊與格式在節點上調整，這裡設定 JPEG 品質。",
          })
        )
      )
    );
    images.appendChild(
      h(
        "div",
        { class: "tdl-grid tdl-grid--3" },
        this.field(
          "JPEG 品質",
          h("input", {
            type: "number",
            min: 1,
            max: 100,
            value: config.image_quality ?? 90,
            oninput: (event) => {
              config.image_quality = Number(event.target.value) || 90;
              this.configTouched();
            },
          }),
          { optional: "1–100" }
        )
      )
    );
    pane.appendChild(images);

    const paths = h("div", { class: "tdl-card" });
    paths.appendChild(
      h(
        "div",
        { class: "tdl-card__head" },
        h(
          "div",
          {},
          h("h4", { class: "tdl-card__title", textContent: "檔案位置" }),
          h("p", {
            class: "tdl-card__summary",
            textContent: this.state.locate?.using_fallback
              ? "節點目錄不可寫，資料改存在 ComfyUI 的使用者目錄下。"
              : "資料就放在節點目錄裡，可以直接用編輯器開啟。",
          })
        )
      )
    );
    const locate = this.state.locate || {};
    for (const [label, path] of [
      ["提示詞庫", locate.tasks_path],
      ["內建範例", locate.default_tasks_path],
      ["連線設定", locate.config_path],
      ["節點目錄", locate.node_dir],
    ]) {
      if (!path) continue;
      paths.appendChild(
        h(
          "div",
          { class: "tdl-pathrow" },
          h("em", { textContent: label }),
          h("span", { textContent: path, title: path }),
          h("button", {
            class: "tdl-iconbtn",
            type: "button",
            title: "複製路徑",
            "aria-label": "複製路徑",
            html: icon("copy", 14),
            onclick: () => this.copyText(path, "路徑已複製"),
          })
        )
      );
    }
    paths.appendChild(
      h(
        "div",
        { class: "tdl-card__actions" },
        h("button", {
          class: "tdl-btn",
          type: "button",
          onclick: () => this.openFolder("prompts"),
        }, h("span", { html: icon("folder", 15) }), "開啟庫的目錄")
      )
    );
    pane.appendChild(paths);
    return pane;
  }

  configTouched() {
    this.renderHeader();
    this.renderFooter();
  }

  async probeProvider(id, button, container) {
    button.disabled = true;
    clear(container);
    container.appendChild(h("p", { class: "tdl-hint", textContent: "測試中…" }));
    try {
      const result = await apiPost(`${API}/probe`, { provider: id, settings: this.providerOverrides(id) });
      clear(container);
      container.appendChild(
        h("div", { class: "tdl-note", dataset: { kind: result.ok ? "ok" : "error" } },
          h("span", { html: icon(result.ok ? "check" : "warn", 15) }),
          h("span", { textContent: result.ok ? result.message || "連線正常。" : result.error || "連線失敗。" }))
      );
      if (result.ok && result.models?.length) {
        container.appendChild(
          h("p", { class: "tdl-hint", textContent: `可用模型：${result.models.slice(0, 12).join("、")}${result.models.length > 12 ? " …" : ""}` })
        );
      }
    } catch (error) {
      clear(container);
      container.appendChild(
        h("div", { class: "tdl-note", dataset: { kind: "error" } },
          h("span", { html: icon("warn", 15) }), h("span", { textContent: error.message }))
      );
    } finally {
      button.disabled = false;
    }
  }

  async fetchModels(id, button, container) {
    button.disabled = true;
    clear(container);
    container.appendChild(h("p", { class: "tdl-hint", textContent: "讀取模型清單…" }));
    try {
      const result = await apiPost(`${API}/models`, { provider: id, settings: this.providerOverrides(id) });
      clear(container);
      if (!result.ok) {
        container.appendChild(
          h("div", { class: "tdl-note", dataset: { kind: "error" } },
            h("span", { html: icon("warn", 15) }), h("span", { textContent: result.error }))
        );
        if (result.hint) container.appendChild(h("p", { class: "tdl-hint", textContent: result.hint }));
        if (result.models?.length) {
          container.appendChild(h("p", { class: "tdl-hint", textContent: `上次成功的清單：${result.models.join("、")}` }));
        }
        return;
      }
      if (!result.models.length) {
        container.appendChild(h("p", { class: "tdl-hint", textContent: "伺服器沒有回報任何模型。" }));
        return;
      }
      container.appendChild(
        h(
          "div",
          { class: "tdl-preview__wiring" },
          ...result.models.map((name) =>
            h("button", {
              class: "tdl-chip",
              type: "button",
              textContent: name,
              title: `設為預設模型：${name}`,
              onclick: () => {
                this.state.config.providers[id].default_model = name;
                this.configTouched();
                this.renderBody();
              },
            })
          )
        )
      );
      container.appendChild(h("p", { class: "tdl-hint", textContent: "點一下模型名稱即可設為預設。" }));
      await this.refreshCombosOnly();
    } catch (error) {
      clear(container);
      container.appendChild(
        h("div", { class: "tdl-note", dataset: { kind: "error" } },
          h("span", { html: icon("warn", 15) }), h("span", { textContent: error.message }))
      );
    } finally {
      button.disabled = false;
    }
  }

  providerOverrides(id) {
    const entry = this.state.config?.providers?.[id];
    if (!entry) return {};
    const overrides = { base_url: entry.base_url, timeout: entry.timeout };
    if (entry.api_key) overrides.api_key = entry.api_key;
    return overrides;
  }

  /* -- footer ------------------------------------------------------------ */

  renderFooter() {
    clear(this.footEl);
    const dirty = this.isLibraryDirty() || this.isConfigDirty();

    if (this.state.confirmingClose) {
      this.footEl.appendChild(
        h(
          "div",
          { class: "tdl-confirm" },
          h("span", { textContent: "有未儲存的變更，確定要關閉嗎？" }),
          h("button", {
            class: "tdl-btn",
            type: "button",
            textContent: "繼續編輯",
            onclick: () => {
              this.state.confirmingClose = false;
              this.renderFooter();
            },
          }),
          h("button", {
            class: "tdl-btn tdl-btn--primary",
            type: "button",
            textContent: "放棄變更並關閉",
            onclick: () => this.close(true),
          })
        )
      );
      return;
    }

    const note = h("div", { class: "tdl-foot__note" });
    const { kind, text } = this.state.status;
    note.dataset.kind = kind;
    if (text) note.textContent = text;
    else if (dirty) note.textContent = "變更尚未寫入檔案。按「儲存」後節點的下拉選單會立即更新。";
    else note.textContent = "所有變更都已儲存。";
    this.footEl.appendChild(note);

    const actions = h("div", { class: "tdl-foot__actions" });

    if (this.mode === "tasks") {
      actions.appendChild(
        h("button", {
          class: "tdl-btn",
          type: "button",
          disabled: !this.currentTask(),
          onclick: () => this.duplicateTask(),
        }, h("span", { html: icon("copy", 15) }), "複製")
      );
      actions.appendChild(
        h("button", {
          class: "tdl-btn",
          type: "button",
          disabled: !this.currentTask(),
          title: "上移（Alt+↑）",
          "aria-label": "上移",
          onclick: () => this.moveTask(-1),
        }, h("span", { html: icon("up", 15) }))
      );
      actions.appendChild(
        h("button", {
          class: "tdl-btn",
          type: "button",
          disabled: !this.currentTask(),
          title: "下移（Alt+↓）",
          "aria-label": "下移",
          onclick: () => this.moveTask(1),
        }, h("span", { html: icon("down", 15) }))
      );
      actions.appendChild(
        h("button", {
          class: "tdl-btn tdl-btn--danger",
          type: "button",
          disabled: !this.currentTask(),
          onclick: () => this.deleteTask(),
        }, h("span", { html: icon("trash", 15) }), "刪除")
      );
      actions.appendChild(h("span", { style: { width: "8px" } }));
    }

    actions.appendChild(
      h("button", {
        class: "tdl-btn",
        type: "button",
        disabled: this.state.busy,
        title: "載入 JSON 檔覆蓋整個提示詞庫",
        onclick: () => this.importLibrary(),
      }, h("span", { html: icon("upload", 15) }), "匯入")
    );
    actions.appendChild(
      h("button", {
        class: "tdl-btn",
        type: "button",
        disabled: this.state.busy,
        onclick: () => this.exportLibrary(),
      }, h("span", { html: icon("download", 15) }), "匯出")
    );
    actions.appendChild(
      h("button", {
        class: "tdl-btn",
        type: "button",
        disabled: this.state.busy,
        onclick: () => this.openFolder("prompts"),
      }, h("span", { html: icon("folder", 15) }), "開啟目錄")
    );
    actions.appendChild(
      h("button", {
        class: "tdl-btn tdl-btn--primary",
        type: "button",
        disabled: !dirty || this.state.busy,
        onclick: () => this.save(),
      }, h("span", { html: icon(dirty ? "save" : "check", 15) }), dirty ? "儲存" : "已儲存")
    );

    this.footEl.appendChild(actions);
  }

  /* -- file actions ------------------------------------------------------ */

  async save() {
    if (this.state.busy) return;
    const task = this.currentTask();
    if (task && !task.title?.trim()) {
      this.setStatus("error", "標題不能是空白的。");
      this.focusField("title");
      return;
    }

    this.setBusy(true, "儲存中…");
    try {
      let response = null;
      if (this.isLibraryDirty()) {
        response = await apiPost(`${API}/library`, { library: this.state.library });
        this.state.warnings = response.warnings || [];
        this.state.savedLibrary = JSON.stringify(this.state.library);
      }
      if (this.isConfigDirty()) {
        const payload = this.editableConfig();
        for (const [id, entry] of Object.entries(this.state.config.providers || {})) {
          if (entry.api_key !== undefined) payload.providers[id].api_key = entry.api_key;
        }
        const saved = await apiPost(`${API}/config`, { config: payload });
        this.state.config = saved.config;
        this.state.savedConfig = JSON.stringify(this.editableConfig());
      }
      libraryCache.data = null;
      await this.refreshCombosOnly();
      await getLibrary(true);

      this.setBusy(false);
      if (this.state.warnings.length) {
        this.setStatus("warn", `已儲存，但有提醒：${this.state.warnings[0]}`);
      } else {
        this.setStatus("ok", `已儲存（${this.state.library.tasks.length} 個 TASK）`);
      }
      this.renderAll();
    } catch (error) {
      this.setBusy(false);
      this.setStatus("error", `儲存失敗：${error.message}`);
    }
  }

  async refreshCombosOnly() {
    try {
      await app.refreshComboInNodes?.();
    } catch (error) {
      console.warn("[Task-Driven LLM] 重新整理下拉選單失敗", error);
      return;
    }
    for (const node of app.graph?._nodes || []) {
      if (node.comfyClass !== NODE_CLASS) continue;
      const widget = nodeWidget(node, W.task);
      if (widget) {
        const raw = widget.options?.values;
        const values = typeof raw === "function" ? raw() : raw;
        if (Array.isArray(values) && values.length && !values.includes(widget.value)) {
          widget.value = values[0];
        }
      }
      applySlotLabels(node);
    }
    app.graph?.setDirtyCanvas(true, true);
  }

  async restoreDefaults() {
    if (
      this.state.library.tasks.length &&
      !confirm("這會用內建範例覆蓋目前的提示詞庫（還未儲存的修改會遺失），要繼續嗎？")
    ) {
      return;
    }
    this.setBusy(true, "還原內建範例…");
    try {
      const result = await apiPost(`${API}/library/restore`, {});
      this.state.library = result.library;
      this.state.savedLibrary = JSON.stringify(this.state.library);
      this.state.selectedId = this.state.library.tasks[0]?.id ?? null;
      libraryCache.data = null;
      await this.refreshCombosOnly();
      this.setBusy(false);
      this.setStatus("ok", `已還原 ${this.state.library.tasks.length} 個內建 TASK。`);
      this.renderAll();
    } catch (error) {
      this.setBusy(false);
      this.setStatus("error", `還原失敗：${error.message}`);
    }
  }

  async importLibrary() {
    const input = h("input", { type: "file", accept: "application/json,.json", style: { display: "none" } });
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) return;
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        this.setBusy(true, "匯入中…");
        const result = await apiPost(`${API}/import`, { library: parsed });
        this.state.library = result.library;
        this.state.savedLibrary = JSON.stringify(this.state.library);
        this.state.selectedId = this.state.library.tasks[0]?.id ?? null;
        libraryCache.data = null;
        await this.refreshCombosOnly();
        this.setBusy(false);
        this.setStatus("ok", `已匯入 ${this.state.library.tasks.length} 個 TASK。`);
        this.renderAll();
      } catch (error) {
        this.setBusy(false);
        this.setStatus("error", `匯入失敗：${error.message}`);
      }
    });
    input.click();
  }

  async exportLibrary() {
    try {
      const response = await api.fetchApi(`${API}/export`, { cache: "no-store" });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = h("a", { href: url, download: "tasks.json" });
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      this.setStatus("ok", "已匯出 tasks.json。");
    } catch (error) {
      this.setStatus("error", `匯出失敗：${error.message}`);
    }
  }

  async openFolder(target) {
    try {
      const result = await apiPost(`${API}/open_folder`, { target });
      await this.copyText(result.path, `已送出開啟指令：${result.path}`, false);
      this.setStatus("idle", `庫的目錄：${result.path}`);
    } catch (error) {
      this.setStatus("error", `無法開啟目錄：${error.message}`);
    }
  }

  async copyText(text, message, setStatus = true) {
    try {
      await navigator.clipboard.writeText(text);
      if (setStatus) this.setStatus("ok", message);
    } catch {
      this.setStatus("warn", text);
    }
  }
}

/* --------------------------------------------------------------------------
 * Extension registration
 * ----------------------------------------------------------------------- */

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const link = document.createElement("link");
  link.id = STYLE_ID;
  link.rel = "stylesheet";
  link.href = CSS_URL;
  document.head.appendChild(link);
}

app.registerExtension({
  name: "taskllm.library",

  async setup() {
    injectStyles();

    // The node's on-node preview is filled from the execution events.
    api.addEventListener("executed", (event) => {
      const detail = event.detail;
      if (!detail?.node) return;
      const node = typeof detail.node === "object" ? detail.node : app.graph?.getNodeById?.(detail.node);
      if (!node || node.comfyClass !== NODE_CLASS) return;
      try {
        showExecutionResult(node, detail.output);
      } catch (error) {
        console.warn("[Task-Driven LLM] 無法顯示執行結果", error);
      }
    });

    api.addEventListener("execution_error", (event) => {
      const detail = event.detail || {};
      if (detail.node_type && detail.node_type !== NODE_CLASS) return;
      const node =
        (detail.node_id && app.graph?.getNodeById?.(detail.node_id)) ||
        app.graph?._nodes?.find((candidate) => candidate.comfyClass === NODE_CLASS && candidate.has_errors);
      if (!node) return;
      setOutput(node, "error", detail.exception_message || detail.exception_type || "執行失敗");
    });

    // Keep socket labels meaningful without needing the editor.
    getLibrary()
      .then(() => {
        for (const node of app.graph?._nodes || []) {
          if (node.comfyClass === NODE_CLASS) applySlotLabels(node);
        }
      })
      .catch(() => {});
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);
      addNodeWidgets(this);
      const taskWidget = nodeWidget(this, W.task);
      if (taskWidget) {
        const original = taskWidget.callback;
        taskWidget.callback = (value, ...rest) => {
          const returned = original?.call(taskWidget, value, ...rest);
          applySlotLabels(this);
          return returned;
        };
      }
      applySlotLabels(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      if (this.__tdlOutput === undefined) {
        this.__tdlOutput = { state: "idle", text: "" };
      }
      const widget = nodeWidget(this, RESULT_WIDGET);
      if (widget && (widget.value === null || widget.value === undefined)) widget.value = "";
      requestAnimationFrame(() => applySlotLabels(this));
      return result;
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_canvas, options) {
      const result = getExtraMenuOptions?.apply(this, arguments);
      options.push(
        null,
        {
          content: "編輯提示詞庫",
          callback: () => openLibraryEditor(this),
        },
        {
          content: "開啟庫的目錄",
          callback: async () => {
            try {
              const response = await apiPost(`${API}/open_folder`, { target: "prompts" });
              alert(`提示詞庫位置：\n${response.path}`);
            } catch (error) {
              alert(`無法取得目錄：${error.message}`);
            }
          },
        }
      );
      return result;
    };

    // Alt+↑ / Alt+↓ reorder the selected TASK (handled by the panel itself
    // while it is open; see LibraryEditor.onKeyDown).
  },
});
