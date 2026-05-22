const state = {
  config: null,
  sizes: null,
  running: false,
  scaleMode: false,
  currentResults: null,
  logOffset: 0,
  selectedRun: "",
  inputFiles: [],
  inputPageSize: 5,
  resultPage: 1,
  resultPageSize: 25,
  scaleMeasurements: new Map(),
  imageBundles: new Map(),
  currentImageBundle: null,
  currentImageIndex: 0,
  statusTimer: null,
  imageZoom: 1,
  imagePanX: 0,
  imagePanY: 0,
  imageDragging: false,
  imageDragStart: null,
  confirmResolver: null,
};

const $ = (id) => document.getElementById(id);

const heroIcon = (paths, width = "1.5") => (
  `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${width}" aria-hidden="true">` +
  paths.map((d) => `<path stroke-linecap="round" stroke-linejoin="round" d="${d}" />`).join("") +
  `</svg>`
);

const solidIcon = (paths) => (
  `<svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">` +
  paths.map((d) => `<path d="${d}" />`).join("") +
  `</svg>`
);

const icons = {
  play: heroIcon(["M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"]),
  trash: heroIcon(["m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"]),
  image: heroIcon(["m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"]),
  folder: heroIcon(["M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 0 0-1.883 2.542l.857 6a2.25 2.25 0 0 0 2.227 1.932H19.05a2.25 2.25 0 0 0 2.227-1.932l.857-6a2.25 2.25 0 0 0-1.883-2.542m-16.5 0V6A2.25 2.25 0 0 1 6 3.75h3.879a1.5 1.5 0 0 1 1.06.44l2.122 2.12a1.5 1.5 0 0 0 1.06.44H18A2.25 2.25 0 0 1 20.25 9v.776"]),
  plus: heroIcon(["M12 4.5v15m7.5-7.5h-15"]),
  expand: heroIcon(["M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15"]),
  collapse: heroIcon(["M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25"]),
  check: heroIcon(["m4.5 12.75 6 6 9-13.5"], "1.8"),
  fileSolid: solidIcon(["M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V7.875L13.875 1.5h-8.25ZM13.5 3.31v3.94c0 .621.504 1.125 1.125 1.125h3.94L13.5 3.31Z"]),
  warning: heroIcon(["M12 9v4.5m0 3h.01M10.34 3.96 2.77 17.1A1.95 1.95 0 0 0 4.46 20h15.08a1.95 1.95 0 0 0 1.69-2.9L13.66 3.96a1.91 1.91 0 0 0-3.32 0Z"], "1.8"),
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function showModal(id) {
  const node = $(id);
  if (!node) return;
  node.hidden = false;
  node.classList.add("show");
  document.body.classList.add("modal-open");
  window.setTimeout(() => node.querySelector("[data-modal-close]")?.focus(), 0);
}

function hideModal(id) {
  const node = $(id);
  if (!node) return;
  node.classList.remove("show");
  node.hidden = true;
  if (!document.querySelector(".modal.show")) {
    document.body.classList.remove("modal-open");
  }
  node.dispatchEvent(new CustomEvent("app-modal-hidden"));
}

function setHidden(node, hidden) {
  if (!node) return;
  node.classList.toggle("hidden", hidden);
}

function resolveConfirm(confirmed) {
  const resolver = state.confirmResolver;
  state.confirmResolver = null;
  hideModal("confirmModal");
  if (resolver) resolver(Boolean(confirmed));
}

function fillConfirmDetails(details) {
  const box = $("confirmDetails");
  box.innerHTML = "";
  const items = Array.isArray(details) ? details.filter(Boolean) : [details].filter(Boolean);
  if (!items.length) {
    box.hidden = true;
    return;
  }

  const list = document.createElement("ul");
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
  box.appendChild(list);
  box.hidden = false;
}

function requestConfirm({
  title = "Xác nhận thao tác",
  message = "Bạn có chắc muốn tiếp tục?",
  details = [],
  confirmLabel = "Xác nhận",
  danger = true,
} = {}) {
  const requiredNodes = ["confirmModal", "confirmTitle", "confirmMessage", "confirmDetails", "confirmAcceptBtn", "confirmCancelBtn"]
    .map((id) => $(id));
  if (requiredNodes.some((node) => !node)) {
    const detailText = (Array.isArray(details) ? details : [details]).filter(Boolean).join("\n");
    return Promise.resolve(window.confirm(`${title}\n\n${message}${detailText ? `\n\n${detailText}` : ""}`));
  }

  if (state.confirmResolver) {
    const resolver = state.confirmResolver;
    state.confirmResolver = null;
    resolver(false);
  }

  $("confirmTitle").textContent = title;
  $("confirmMessage").textContent = message;
  $("confirmAcceptBtn").textContent = confirmLabel;
  $("confirmAcceptBtn").classList.toggle("btn-danger", danger);
  $("confirmAcceptBtn").classList.toggle("btn-primary", !danger);
  fillConfirmDetails(details);

  return new Promise((resolve) => {
    state.confirmResolver = resolve;
    showModal("confirmModal");
    window.setTimeout(() => $("confirmCancelBtn")?.focus(), 0);
  });
}

function flashSaved(button, label = "Đã lưu") {
  if (!button) return;
  const previousHtml = button.dataset.previousHtml || button.innerHTML;
  button.dataset.previousHtml = previousHtml;
  window.clearTimeout(button.saveFeedbackTimer);
  button.classList.add("btn-saved");
  button.innerHTML = `${icons.check} ${label}`;
  button.saveFeedbackTimer = window.setTimeout(() => {
    button.classList.remove("btn-saved");
    button.innerHTML = button.dataset.previousHtml || previousHtml;
    delete button.dataset.previousHtml;
  }, 1800);
}

function toast(message, type = "ok", options = {}) {
  const box = $("toast");
  const body = $("toastBody") || box;
  const variants = {
    ok: {title: "Thông báo", icon: "i"},
    success: {title: "Thành công", icon: "✓"},
    warning: {title: "Cảnh báo", icon: "!"},
    error: {title: "Lỗi", icon: "!"},
  };
  const variant = variants[type] || variants.ok;
  body.innerHTML = "";
  const marker = document.createElement("span");
  marker.className = "toast-mark";
  marker.textContent = options.icon || variant.icon;
  const copy = document.createElement("span");
  copy.className = "toast-copy";
  const title = document.createElement("strong");
  title.textContent = options.title || variant.title;
  const text = document.createElement("span");
  text.textContent = message;
  copy.append(title, text);
  body.append(marker, copy);
  box.className = `toast ${type}`;
  box.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => box.classList.remove("show"), options.duration || (type === "success" ? 4200 : 3000));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatTime(seconds) {
  if (!seconds) return "--";
  return new Date(seconds * 1000).toLocaleString("vi-VN");
}

function clampPage(page, totalPages) {
  return Math.min(Math.max(1, Number(page) || 1), Math.max(1, totalPages || 1));
}

function pageWindow(total, page, pageSize) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const currentPage = clampPage(page, totalPages);
  const start = total ? (currentPage - 1) * pageSize : 0;
  const end = total ? Math.min(start + pageSize, total) : 0;
  return {totalPages, currentPage, start, end};
}

function imageFileName(url) {
  const clean = String(url || "").split(/[?#]/)[0];
  const fileName = clean.split("/").filter(Boolean).at(-1);
  if (!fileName) return "";
  try {
    return decodeURIComponent(fileName);
  } catch (_error) {
    return fileName;
  }
}

function updateConfidenceLabels() {
  $("confDetValue").textContent = `${$("cfg_CONF_DET").value}%`;
  $("confSegValue").textContent = `${$("cfg_CONF_SEG").value}%`;
}

function setRunning(running) {
  state.running = running;
  document.body.classList.toggle("is-running", running);

  const runBtn = $("runBtn");
  runBtn.disabled = running;
  runBtn.classList.toggle("run-active", running);
  runBtn.innerHTML = running
    ? `<span class="spinner-border" aria-hidden="true"></span> Đang chạy`
    : `${icons.play} Chạy pipeline`;

  ["scaleModeBtn", "applyScaleBtn", "cancelScaleBtn"].forEach((id) => {
    const button = $(id);
    if (button) button.disabled = running;
  });

  $("fileInput").disabled = running;
  $("folderInput").disabled = running;
  ["chooseFileBtn", "chooseFolderBtn"].forEach((id) => {
    const button = $(id);
    if (button) button.disabled = running;
  });
  $("dropZone").classList.toggle("disabled", running);
  document.querySelectorAll("#configForm input, #configForm button, #sizeForm input, #sizeForm button")
    .forEach((node) => {
      node.disabled = running;
    });

  const status = $("pipelineStatus");
  status.classList.toggle("running", running);
  status.classList.remove("error");
  status.textContent = running ? "Đang chạy" : "Sẵn sàng";
}

function updateStatusView(status) {
  const running = Boolean(status.running);
  setRunning(running);
  const statusEl = $("pipelineStatus");
  if (!running && status.returncode !== null && status.returncode !== 0) {
    statusEl.classList.add("error");
    statusEl.textContent = `Lỗi ${status.returncode}`;
  }
  if (running) {
    $("runHint").textContent = `Bắt đầu: ${formatTime(status.started_at)}`;
  } else if (status.ended_at) {
    $("runHint").textContent = `Hoàn tất: ${formatTime(status.ended_at)}`;
  } else {
    $("runHint").textContent = "Log sẽ đọc từ output/pipeline.log khi chạy.";
  }
}

async function refreshStatus() {
  try {
    const status = await requestJson("/api/pipeline/status");
    const wasRunning = state.running;
    updateStatusView(status);
    if (status.running) {
      await pollLog();
    }
    if (wasRunning && !status.running) {
      await pollLog();
      await loadRuns({preferLatest: true});
      await loadResults();
      toast(status.returncode === 0 ? "Pipeline đã chạy xong" : "Pipeline dừng với lỗi", status.returncode === 0 ? "success" : "error");
    }
    if (status.running) {
      startStatusPolling();
    } else {
      stopStatusPolling();
    }
  } catch (error) {
    stopStatusPolling();
    toast(error.message, "error");
  }
}

function startStatusPolling() {
  if (state.statusTimer) return;
  state.statusTimer = window.setTimeout(async () => {
    state.statusTimer = null;
    await refreshStatus();
  }, 1500);
}

function stopStatusPolling() {
  if (!state.statusTimer) return;
  window.clearTimeout(state.statusTimer);
  state.statusTimer = null;
}

function appendLog(content) {
  if (!content) return;
  const logBody = $("logBody");
  const lines = content.split(/\r?\n/);
  lines.forEach((line, index) => {
    if (!line && index === lines.length - 1) return;
    const div = document.createElement("div");
    div.className = "log-line";
    if (line.includes("[ERROR]") || line.toLowerCase().includes("error")) {
      div.classList.add("error");
    } else if (line.includes("[WARNING]") || line.toLowerCase().includes("warning")) {
      div.classList.add("warning");
    }
    div.textContent = line;
    logBody.appendChild(div);
  });
  logBody.scrollTop = logBody.scrollHeight;
}

async function pollLog() {
  const payload = await requestJson(`/api/pipeline/log?offset=${state.logOffset}`);
  if (payload.offset < state.logOffset) {
    $("logBody").textContent = "";
  }
  state.logOffset = payload.offset;
  appendLog(payload.content);
}

function showSettingsErrors(payload) {
  const errors = Array.isArray(payload?._settings_errors) ? payload._settings_errors : [];
  if (!errors.length) return;
  toast(errors.join(" | "), "warning", {
    title: "Lỗi đọc settings.json",
    duration: 7000,
  });
}

async function loadConfig() {
  const payload = await requestJson("/api/config");
  showSettingsErrors(payload);
  const {_settings_errors, ...config} = payload;
  state.config = config;
  Object.entries(state.config).forEach(([key, value]) => {
    const input = $(`cfg_${key}`);
    if (!input) return;
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
    } else if (key === "CONF_DET" || key === "CONF_SEG") {
      input.value = Math.round(Number(value) * 100);
    } else {
      input.value = value;
    }
  });
  updateConfidenceLabels();
  $("scaleChip").textContent = `SCALE: ${Number(state.config.SCALE).toFixed(4)} mm/px`;
}

function collectConfig() {
  const keys = [
    "INPUT_DIR",
    "OUTPUT_DIR",
    "SCALE",
    "CONF_DET",
    "CONF_SEG",
    "BBOX_PAD",
    "TOUCH_THRESHOLD",
    "TARGET_FPS",
    "SAVE",
    "CLEAR_INPUT",
    "CLEAR_OUTPUT",
    "CHUNK_MODE",
    "CONVEYOR_VERTICAL",
  ];
  const payload = {};
  keys.forEach((key) => {
    const input = $(`cfg_${key}`);
    if (input.type === "checkbox") {
      payload[key] = input.checked;
    } else if (key === "CONF_DET" || key === "CONF_SEG") {
      payload[key] = Number(input.value) / 100;
    } else if (input.type === "number") {
      payload[key] = Number(input.value);
    } else {
      payload[key] = input.value.trim();
    }
  });
  return payload;
}

async function saveConfig(event) {
  event.preventDefault();
  const submitter = event.submitter || event.currentTarget.querySelector("[type='submit']");
  try {
    state.config = await requestJson("/api/config", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectConfig()),
    });
    await loadConfig();
    await loadInputFiles();
    flashSaved(submitter, "Đã lưu config");
    toast("Cấu hình đã được lưu chính thức vào settings.json", "success", {title: "Đã lưu cấu hình"});
  } catch (error) {
    toast(error.message, "error");
  }
}

async function pickConfigPath(button) {
  const key = button.dataset.configKey;
  const mode = button.dataset.pickMode;
  if (!key || !mode || state.running) return;

  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  try {
    const result = await requestJson("/api/config/pick-path", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({key, mode}),
    });
    if (result.cancelled || !result.path) return;

    const input = $(`cfg_${key}`);
    if (input) {
      input.value = result.path;
      input.focus();
    }
    if (key === "INPUT_DIR") {
      state.config = await requestJson("/api/config", {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(collectConfig()),
      });
      await loadConfig();
      await loadInputFiles();
      toast("Đã chọn INPUT_DIR và cập nhật danh sách file input", "success");
      return;
    }
    toast(`Đã chọn ${key}`, "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.removeAttribute("aria-busy");
    button.disabled = state.running;
  }
}

async function loadSizes() {
  const payload = await requestJson("/api/config/sizes");
  showSettingsErrors(payload);
  const {_settings_errors, ...sizes} = payload;
  state.sizes = sizes;
  $("size_undersize").value = state.sizes.undersize_label;
  $("size_oversize").value = state.sizes.oversize_label;
  $("size_fallback").value = state.sizes.fallback_label;
  renderSizeRows();
}

function renderSizeRows() {
  const rows = $("sizeRows");
  rows.innerHTML = "";
  Object.entries(state.sizes.ranges).forEach(([label, bounds]) => {
    rows.appendChild(createSizeRow(label, bounds[0], bounds[1]));
  });
}

function createSizeRow(label = "", lo = "", hi = "") {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="form-control size-name-input" data-size-label type="text" value="${escapeHtml(label)}" placeholder="tên cỡ"></td>
    <td><input class="form-control" data-size-lo type="number" min="0" step="any" inputmode="decimal" value="${escapeHtml(lo)}"></td>
    <td><input class="form-control" data-size-hi type="number" min="0" step="any" inputmode="decimal" value="${escapeHtml(hi)}"></td>
    <td>
      <button class="btn btn-secondary size-remove-btn" type="button" data-remove-size aria-label="Xóa cỡ ${escapeHtml(label || "mới")}">${icons.trash} Xóa</button>
    </td>
  `;
  return tr;
}

function suggestSizeName() {
  const used = new Set(
    [...document.querySelectorAll("[data-size-label]")]
      .map((input) => input.value.trim())
      .filter(Boolean),
  );
  let index = document.querySelectorAll("#sizeRows tr").length + 1;
  let label = `cỡ ${index}`;
  while (used.has(label)) {
    index += 1;
    label = `cỡ ${index}`;
  }
  return label;
}

function addSizeRow() {
  const rows = $("sizeRows");
  const previousRow = rows.querySelector("tr:last-child");
  const previousHi = previousRow ? Number(previousRow.querySelector("[data-size-hi]").value) : 0;
  const lo = Number.isFinite(previousHi) ? previousHi : 0;
  const hi = lo + 10;
  const row = createSizeRow(suggestSizeName(), lo, hi);
  rows.appendChild(row);
  row.querySelector("[data-size-label]").focus();
}

function collectSizes() {
  const ranges = {};
  const labels = new Set();
  document.querySelectorAll("#sizeRows tr").forEach((row) => {
    const label = row.querySelector("[data-size-label]").value.trim();
    const lo = Number(row.querySelector("[data-size-lo]").value);
    const hi = Number(row.querySelector("[data-size-hi]").value);
    if (!label) {
      throw new Error("Tên cỡ không được để trống");
    }
    if (labels.has(label)) {
      throw new Error(`Tên cỡ ${label} bị trùng`);
    }
    labels.add(label);
    ranges[label] = [lo, hi];
  });
  return {
    ranges,
    undersize_label: $("size_undersize").value.trim(),
    oversize_label: $("size_oversize").value.trim(),
    fallback_label: $("size_fallback").value.trim(),
  };
}

async function saveSizes(event) {
  event.preventDefault();
  const submitter = event.submitter || event.currentTarget.querySelector("[type='submit']");
  try {
    state.sizes = await requestJson("/api/config/sizes", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectSizes()),
    });
    await loadSizes();
    flashSaved(submitter, "Đã lưu phân loại");
    toast("Bảng phân loại kích cỡ đã được lưu chính thức vào settings.json", "success", {title: "Đã lưu phân loại"});
  } catch (error) {
    toast(error.message, "error");
  }
}

async function uploadFiles(files) {
  if (!files.length || state.running) return;
  const form = new FormData();
  Array.from(files).forEach((file) => form.append("files", file));
  try {
    const result = await requestJson("/api/files/upload", {method: "POST", body: form});
    await loadInputFiles();
    const saved = result.saved.length;
    const rejected = result.rejected.length;
    toast(rejected ? `Đã nạp ${saved} file, từ chối ${rejected} file` : `Đã nạp ${saved} file`);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("fileInput").value = "";
    $("folderInput").value = "";
  }
}

async function loadInputFiles() {
  const payload = await requestJson("/api/files/input");
  state.inputFiles = payload.files || [];
  state.inputPageSize = Number($("inputPageSize")?.value) || state.inputPageSize;
  renderInputFiles();
}

function renderInputFiles() {
  const list = $("fileList");
  const pager = $("inputPager");
  const total = state.inputFiles.length;
  const visibleCount = Number(state.inputPageSize) || 5;
  list.innerHTML = "";
  list.style.maxHeight = "";
  list.classList.remove("scrollable");

  if (!total) {
    list.innerHTML = `<div class="empty-state compact">Input đang trống</div>`;
    if (pager) pager.hidden = false;
    $("inputPageInfo").textContent = "0 file";
    return;
  }

  state.inputFiles.forEach((file) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <div class="file-main" title="${escapeHtml(file.name)} ${escapeHtml(file.suffix || "file")} · ${formatBytes(file.size)}">
        <span class="file-name">${escapeHtml(file.name)}</span>
        <span class="file-meta">${escapeHtml(file.suffix || "file")} · ${formatBytes(file.size)}</span>
      </div>
      <button class="btn btn-secondary" type="button" data-delete-file="${escapeHtml(file.name)}" aria-label="Xóa file ${escapeHtml(file.name)}">${icons.trash} Xóa</button>
    `;
    list.appendChild(row);
  });

  list.style.maxHeight = `${visibleCount * 50 + Math.max(0, visibleCount - 1) * 8}px`;
  list.classList.toggle("scrollable", total > visibleCount);
  if (pager) pager.hidden = false;
  $("inputPageInfo").textContent = `${total} file`;
}

async function deleteInputFile(name) {
  try {
    await requestJson(`/api/files/input/${encodeURIComponent(name)}`, {method: "DELETE"});
    await loadInputFiles();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function confirmPipelineDataDeletion() {
  const formConfig = collectConfig();
  const config = state.config || formConfig;
  const details = [];
  if (config.CLEAR_OUTPUT) {
    details.push(`CLEAR_OUTPUT đang bật: kết quả cũ trong "${config.OUTPUT_DIR}" sẽ bị xóa trước khi chạy.`);
  }
  if (config.CLEAR_INPUT) {
    details.push(`CLEAR_INPUT đang bật: file trong "${config.INPUT_DIR}" sẽ bị xóa sau khi pipeline ghi JSON thành công.`);
  }
  if (
    details.length
    &&
    state.config
    && (Boolean(formConfig.CLEAR_OUTPUT) !== Boolean(state.config.CLEAR_OUTPUT)
      || Boolean(formConfig.CLEAR_INPUT) !== Boolean(state.config.CLEAR_INPUT))
  ) {
    details.push("Pipeline dùng cấu hình đã lưu chính thức trong settings.json. Nếu vừa đổi CLEAR_INPUT/CLEAR_OUTPUT trên form, hãy lưu config trước khi chạy.");
  }
  if (!details.length) return true;

  return requestConfirm({
    title: "Xác nhận xóa dữ liệu",
    message: "Pipeline sắp chạy với cấu hình có bật tự động xóa dữ liệu.",
    details,
    confirmLabel: "Vẫn chạy pipeline",
  });
}

async function runPipeline() {
  if (!(await confirmPipelineDataDeletion())) return;

  try {
    const runBtn = $("runBtn");
    runBtn.classList.add("run-clicked");
    window.setTimeout(() => runBtn.classList.remove("run-clicked"), 260);
    $("logBody").textContent = "";
    state.logOffset = 0;
    const payload = await requestJson("/api/pipeline/run", {method: "POST"});
    updateStatusView(payload.status);
    await pollLog();
    if (payload.status?.running) {
      startStatusPolling();
    }
    toast("Pipeline đã bắt đầu chạy", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadRuns({preferLatest = false} = {}) {
  const payload = await requestJson("/api/results/runs");
  const select = $("runSelect");
  const previous = state.selectedRun || select.value;
  select.innerHTML = "";
  if (!payload.runs.length) {
    select.innerHTML = `<option value="">Chưa có run</option>`;
    state.selectedRun = "";
    return;
  }
  payload.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = `${run.name} · ${run.shrimp_count} tôm`;
    select.appendChild(option);
  });
  state.selectedRun = !preferLatest && payload.runs.some((run) => run.name === previous) ? previous : payload.runs[0].name;
  select.value = state.selectedRun;
}

function updateSizeFilter(sources) {
  const select = $("sizeFilter");
  const previous = select.value;
  const sizes = new Set();
  sources.forEach((source) => (source.shrimps || []).forEach((shrimp) => sizes.add(String(shrimp.size))));
  select.innerHTML = `<option value="">Tất cả size</option>`;
  [...sizes].sort().forEach((size) => {
    const option = document.createElement("option");
    option.value = size;
    option.textContent = size;
    select.appendChild(option);
  });
  select.value = sizes.has(previous) ? previous : "";
}

function flattenImages(images) {
  const list = [];
  Object.entries(images || {}).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach((url, index) => {
        const fileName = imageFileName(url);
        list.push({label: fileName ? `${key}_${index + 1} · ${fileName}` : `${key}_${index + 1}`, url});
      });
    } else if (value) {
      const fileName = imageFileName(value);
      list.push({label: fileName ? `${key} · ${fileName}` : key, url: value});
    }
  });
  return list;
}

async function loadResults() {
  const run = $("runSelect").value;
  state.selectedRun = run;
  state.resultPage = 1;
  state.resultPageSize = Number($("resultPageSize")?.value) || state.resultPageSize;
  const data = await requestJson(`/api/results${run ? `?run=${encodeURIComponent(run)}` : ""}`);
  state.currentResults = data;
  updateSizeFilter(data.sources || []);
  renderResults(data);
  $("exportCsv").href = `/api/results/export-csv${data.run ? `?run=${encodeURIComponent(data.run)}` : ""}`;
}

function renderResults(data) {
  return renderPaginatedResults(data);
  const area = $("resultsArea");
  const filter = $("sizeFilter").value;
  state.imageBundles.clear();
  const sources = data.sources || [];
  let shrimpCount = 0;

  area.innerHTML = "";
  if (!data.run || !sources.length) {
    area.innerHTML = `<div class="empty-state">Chưa có JSON kết quả trong output.</div>`;
    $("resultSummary").innerHTML = `<span>0 nguồn</span><span>0 tôm</span><span>Chưa có kết quả</span>`;
    return;
  }

  sources.forEach((source, sourceIndex) => {
    const allShrimps = source.shrimps || [];
    const shrimps = allShrimps.filter((shrimp) => !filter || String(shrimp.size) === filter);
    shrimpCount += shrimps.length;
    const block = document.createElement("details");
    block.className = "source-block";
    block.open = sourceIndex === 0;
    const rows = shrimps.map((shrimp, shrimpIndex) => {
      const images = flattenImages(shrimp.images);
      const imageKey = `${source.source_stem}_${shrimp.track_id}_${shrimpIndex}`;
      if (images.length) {
        state.imageBundles.set(imageKey, {
          title: `${source.source_file} · ID ${shrimp.track_id} · ${shrimp.real_length_mm} mm`,
          images,
        });
      }
      const mmCell = state.scaleMode
        ? `<input class="form-control scale-mm-input" data-scale-mm data-source-stem="${escapeHtml(source.source_stem)}" data-track-id="${escapeHtml(shrimp.track_id)}" type="number" min="0" step="any" inputmode="decimal" placeholder="Nhập mm">`
        : escapeHtml(shrimp.real_length_mm);
      return `
        <tr>
          <td>${escapeHtml(shrimp.track_id)}</td>
          <td>${escapeHtml(shrimp.frame_idx)}</td>
          <td>${escapeHtml(shrimp.pixel_length)}</td>
          <td>${mmCell}</td>
          <td><span class="result-size-text">${escapeHtml(shrimp.size)}</span></td>
          <td>
            <button class="btn btn-secondary" type="button" data-image-key="${escapeHtml(imageKey)}" aria-label="Xem ảnh debug ID ${escapeHtml(shrimp.track_id)}" ${images.length ? "" : "disabled"}>${icons.image} Ảnh</button>
          </td>
        </tr>
      `;
    }).join("");
    block.innerHTML = `
      <summary>
        <span class="source-title">${escapeHtml(source.source_file)}</span>
        <span class="source-meta">${shrimps.length}/${allShrimps.length} tôm · scale ${escapeHtml(source.scale_mm_per_px ?? "--")}</span>
      </summary>
      <div class="table-responsive">
        <table class="table table-sm">
          <thead>
            <tr>
              <th style="width:80px">ID</th>
              <th style="width:110px">Frame</th>
              <th>Pixel</th>
              <th>${state.scaleMode ? "mm thực tế" : "mm"}</th>
              <th style="width:120px">Size</th>
              <th style="width:110px">Ảnh</th>
            </tr>
          </thead>
          <tbody>${rows || `<tr><td colspan="6">Không có dòng phù hợp bộ lọc.</td></tr>`}</tbody>
        </table>
      </div>
    `;
    area.appendChild(block);
  });

  $("resultSummary").innerHTML = `
    <span>${sources.length} nguồn</span>
    <span>${shrimpCount} tôm</span>
    <span>Run: ${escapeHtml(data.run)}</span>
  `;
}

function renderPaginatedResults(data) {
  const area = $("resultsArea");
  const pager = $("resultPager");
  const filter = $("sizeFilter").value;
  const sources = data.sources || [];
  const rows = [];
  state.imageBundles.clear();
  area.innerHTML = "";

  if (!data.run || !sources.length) {
    area.innerHTML = `<div class="empty-state">Chưa có JSON kết quả trong output.</div>`;
    $("resultSummary").innerHTML = `<span>0 nguồn</span><span>0 tôm</span><span>Chưa có kết quả</span>`;
    if (pager) pager.hidden = true;
    return;
  }

  sources.forEach((source, sourceIndex) => {
    const allShrimps = source.shrimps || [];
    allShrimps.forEach((shrimp, shrimpIndex) => {
      if (!filter || String(shrimp.size) === filter) {
        rows.push({source, sourceIndex, shrimp, shrimpIndex, allCount: allShrimps.length});
      }
    });
  });

  if (!rows.length) {
    area.innerHTML = `<div class="empty-state">Không có dòng phù hợp bộ lọc.</div>`;
    $("resultSummary").innerHTML = `
      <span>${sources.length} nguồn</span>
      <span>0 tôm</span>
      <span>Run: ${escapeHtml(data.run)}</span>
    `;
    if (pager) pager.hidden = true;
    return;
  }

  const sourceFilteredCounts = new Map();
  rows.forEach((row) => {
    sourceFilteredCounts.set(row.sourceIndex, (sourceFilteredCounts.get(row.sourceIndex) || 0) + 1);
  });

  const windowInfo = pageWindow(rows.length, state.resultPage, state.resultPageSize);
  state.resultPage = windowInfo.currentPage;
  const pageRows = rows.slice(windowInfo.start, windowInfo.end);
  const groups = new Map();
  pageRows.forEach((row) => {
    if (!groups.has(row.sourceIndex)) {
      groups.set(row.sourceIndex, {source: row.source, allCount: row.allCount, rows: []});
    }
    groups.get(row.sourceIndex).rows.push(row);
  });

  [...groups.values()].forEach((group, groupIndex) => {
    const block = document.createElement("details");
    block.className = "source-block";
    block.open = groupIndex === 0;
    const sourceIndex = sources.indexOf(group.source);
    const rowsHtml = group.rows.map(({source, sourceIndex: rowSourceIndex, shrimp, shrimpIndex}) => {
      const images = flattenImages(shrimp.images);
      const imageKey = `${rowSourceIndex}_${source.source_stem}_${shrimp.track_id}_${shrimpIndex}`;
      if (images.length) {
        state.imageBundles.set(imageKey, {
          title: `${source.source_file} · ID ${shrimp.track_id} · ${shrimp.real_length_mm} mm`,
          images,
        });
      }
      const measureKey = `${source.source_stem}::${shrimp.track_id}`;
      const savedMm = state.scaleMeasurements.get(measureKey) ?? "";
      const mmCell = state.scaleMode
        ? `<input class="form-control scale-mm-input" data-scale-mm data-measure-key="${escapeHtml(measureKey)}" data-source-stem="${escapeHtml(source.source_stem)}" data-track-id="${escapeHtml(shrimp.track_id)}" type="number" min="0" step="any" inputmode="decimal" placeholder="Nhập mm" value="${escapeHtml(savedMm)}">`
        : escapeHtml(shrimp.real_length_mm);
      return `
        <tr>
          <td>${escapeHtml(shrimp.track_id)}</td>
          <td>${escapeHtml(shrimp.frame_idx)}</td>
          <td>${escapeHtml(shrimp.pixel_length)}</td>
          <td>${mmCell}</td>
          <td><span class="result-size-text">${escapeHtml(shrimp.size)}</span></td>
          <td>
            <button class="btn btn-secondary" type="button" data-image-key="${escapeHtml(imageKey)}" aria-label="Xem ảnh debug ID ${escapeHtml(shrimp.track_id)}" ${images.length ? "" : "disabled"}>${icons.image} Ảnh</button>
          </td>
        </tr>
      `;
    }).join("");
    const filteredCount = sourceFilteredCounts.get(sourceIndex) || group.rows.length;
    block.innerHTML = `
      <summary>
        <span class="source-title">${escapeHtml(group.source.source_file)}</span>
        <span class="source-meta">${group.rows.length}/${filteredCount} tôm trên trang · tổng ${group.allCount} · scale ${escapeHtml(group.source.scale_mm_per_px ?? "--")}</span>
      </summary>
      <div class="table-responsive">
        <table class="table table-sm">
          <thead>
            <tr>
              <th style="width:80px">ID</th>
              <th style="width:110px">Frame</th>
              <th>Pixel</th>
              <th>${state.scaleMode ? "mm thực tế" : "mm"}</th>
              <th style="width:120px">Size</th>
              <th style="width:110px">Ảnh</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
    `;
    area.appendChild(block);
  });

  $("resultSummary").innerHTML = `
    <span>${sources.length} nguồn</span>
    <span>${rows.length} tôm</span>
    <span>Run: ${escapeHtml(data.run)}</span>
  `;

  if (pager) pager.hidden = false;
  $("resultPageInfo").textContent = `${windowInfo.start + 1}-${windowInfo.end} / ${rows.length} · Trang ${windowInfo.currentPage}/${windowInfo.totalPages}`;
  $("resultPrevPage").disabled = windowInfo.currentPage <= 1;
  $("resultNextPage").disabled = windowInfo.currentPage >= windowInfo.totalPages;
}

function updateScaleControls() {
  document.querySelector(".result-panel")?.classList.toggle("scale-mode", state.scaleMode);
  setHidden($("scaleModeBtn"), state.scaleMode);
  setHidden($("applyScaleBtn"), !state.scaleMode);
  setHidden($("cancelScaleBtn"), !state.scaleMode);
  setHidden($("scaleHelp"), !state.scaleMode);
}

function enterScaleMode() {
  if (!state.currentResults?.run || !state.currentResults.sources.length) {
    toast("Chưa có kết quả để tính scale", "error");
    return;
  }
  state.scaleMode = true;
  state.scaleMeasurements.clear();
  updateScaleControls();
  renderResults(state.currentResults);
  toast("Đã bật chế độ tính scale");
}

function cancelScaleMode() {
  state.scaleMode = false;
  state.scaleMeasurements.clear();
  updateScaleControls();
  if (state.currentResults) {
    renderResults(state.currentResults);
  }
}

function syncScaleMeasurements() {
  document.querySelectorAll("[data-scale-mm]").forEach((input) => {
    const value = input.value.trim();
    if (value) {
      state.scaleMeasurements.set(input.dataset.measureKey, value);
    } else {
      state.scaleMeasurements.delete(input.dataset.measureKey);
    }
  });
}

function collectScaleMeasurements() {
  syncScaleMeasurements();
  return [...state.scaleMeasurements.entries()]
    .map(([key, value]) => {
      const [sourceStem, trackId] = key.split("::");
      return {
        source_stem: sourceStem,
        track_id: trackId,
        real_length_mm: Number(value),
      };
    })
    .filter((item) => Number.isFinite(item.real_length_mm) && item.real_length_mm > 0);
  return [...document.querySelectorAll("[data-scale-mm]")]
    .map((input) => ({
      source_stem: input.dataset.sourceStem,
      track_id: input.dataset.trackId,
      real_length_mm: Number(input.value),
    }))
    .filter((item) => Number.isFinite(item.real_length_mm) && item.real_length_mm > 0);
}

async function applyScale() {
  const measurements = collectScaleMeasurements();
  if (!measurements.length) {
    toast("Nhập ít nhất một giá trị mm thực tế", "error");
    return;
  }
  try {
    const result = await requestJson("/api/calibrate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        run: state.currentResults.run,
        measurements,
      }),
    });
    state.scaleMode = false;
    state.scaleMeasurements.clear();
    updateScaleControls();
    await loadConfig();
    if (state.currentResults) {
      renderResults(state.currentResults);
    }
    flashSaved($("applyScaleBtn"), "Đã lưu scale");
    toast(`SCALE = ${Number(result.scale).toFixed(6)}, b = ${Number(result.intercept_mm).toFixed(6)} mm theo y = m x + b từ ${result.count} mẫu`, "success", {title: "Đã lưu scale"});
  } catch (error) {
    toast(error.message, "error");
  }
}

function applyImageTransform() {
  const image = $("modalImage");
  if (!image) return;
  image.style.transform = `translate(${state.imagePanX}px, ${state.imagePanY}px) scale(${state.imageZoom})`;
  $("imageZoomReset").textContent = `${state.imageZoom.toFixed(1)}x`;
}

function setImageZoom(nextZoom) {
  state.imageZoom = Math.min(4, Math.max(0.5, nextZoom));
  applyImageTransform();
}

function resetImagePan() {
  state.imagePanX = 0;
  state.imagePanY = 0;
  state.imageDragging = false;
  state.imageDragStart = null;
  document.querySelector(".image-view")?.classList.remove("panning");
}

function resetImageZoom() {
  state.imageZoom = 1;
  resetImagePan();
  applyImageTransform();
}

function updateImageNav() {
  const count = state.currentImageBundle?.images?.length || 0;
  ["imagePrev", "imageNext"].forEach((id) => {
    const button = $(id);
    if (!button) return;
    button.hidden = count <= 1;
    button.disabled = count <= 1;
  });
}

function showModalImage(index) {
  const bundle = state.currentImageBundle;
  if (!bundle?.images?.length) return;
  const count = bundle.images.length;
  state.currentImageIndex = ((index % count) + count) % count;
  const item = bundle.images[state.currentImageIndex];
  const image = $("modalImage");
  const caption = $("imageCaption");
  image.src = item.url;
  image.alt = item.label;
  caption.textContent = item.label;
  caption.hidden = false;
  resetImageZoom();
  $("imageTabs").querySelectorAll("[data-image-index]").forEach((node) => {
    node.classList.toggle("active", Number(node.dataset.imageIndex) === state.currentImageIndex);
  });
  updateImageNav();
}

function stepImage(delta) {
  if (!state.currentImageBundle?.images?.length) return;
  showModalImage(state.currentImageIndex + delta);
}

function beginImagePan(event) {
  if (event.button !== 0 || event.target.closest("button, .image-caption")) return;
  const image = $("modalImage");
  if (!image?.src || image.hidden) return;
  state.imageDragging = true;
  state.imageDragStart = {
    x: event.clientX,
    y: event.clientY,
    panX: state.imagePanX,
    panY: state.imagePanY,
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  event.currentTarget.classList.add("panning");
  event.preventDefault();
}

function moveImagePan(event) {
  if (!state.imageDragging || !state.imageDragStart) return;
  state.imagePanX = state.imageDragStart.panX + event.clientX - state.imageDragStart.x;
  state.imagePanY = state.imageDragStart.panY + event.clientY - state.imageDragStart.y;
  applyImageTransform();
}

function endImagePan(event) {
  if (!state.imageDragging) return;
  state.imageDragging = false;
  state.imageDragStart = null;
  event.currentTarget.releasePointerCapture?.(event.pointerId);
  event.currentTarget.classList.remove("panning");
}

function openImageModal(bundle) {
  state.currentImageBundle = bundle;
  state.currentImageIndex = 0;
  $("modalTitle").textContent = bundle.title;
  const tabs = $("imageTabs");
  const image = $("modalImage");
  const empty = $("imageEmpty");
  const caption = $("imageCaption");
  tabs.innerHTML = "";
  resetImageZoom();
  if (!bundle.images.length) {
    image.hidden = true;
    empty.hidden = false;
    caption.hidden = true;
    updateImageNav();
  } else {
    image.hidden = false;
    empty.hidden = true;
  }
  bundle.images.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "image-thumb";
    button.dataset.imageIndex = String(index);
    button.setAttribute("aria-label", item.label);
    button.title = item.label;
    button.innerHTML = `<img src="${escapeHtml(item.url)}" alt="">`;
    button.addEventListener("click", () => showModalImage(index));
    tabs.appendChild(button);
  });
  if (bundle.images.length) showModalImage(0);
  showModal("imageModal");
}

function bindModalEvents() {
  document.querySelectorAll("[data-modal-close]").forEach((button) => {
    button.addEventListener("click", () => hideModal(button.dataset.modalClose || button.closest(".modal")?.id));
  });
  document.querySelectorAll(".modal").forEach((modal) => {
    modal.addEventListener("click", (event) => {
      if (event.target === modal) hideModal(modal.id);
    });
  });
  document.addEventListener("keydown", (event) => {
    const openModal = [...document.querySelectorAll(".modal.show")].at(-1);
    if (!openModal) return;
    if (event.key === "Escape") {
      hideModal(openModal.id);
    } else if (openModal.id === "imageModal" && event.key === "ArrowLeft") {
      event.preventDefault();
      stepImage(-1);
    } else if (openModal.id === "imageModal" && event.key === "ArrowRight") {
      event.preventDefault();
      stepImage(1);
    }
  });
}

function bindEvents() {
  bindModalEvents();
  const confirmCancelBtn = $("confirmCancelBtn");
  const confirmAcceptBtn = $("confirmAcceptBtn");
  const confirmModal = $("confirmModal");
  if (confirmCancelBtn && confirmAcceptBtn && confirmModal) {
    confirmCancelBtn.addEventListener("click", () => resolveConfirm(false));
    confirmAcceptBtn.addEventListener("click", () => resolveConfirm(true));
    confirmModal.addEventListener("app-modal-hidden", () => {
      if (!state.confirmResolver) return;
      const resolver = state.confirmResolver;
      state.confirmResolver = null;
      resolver(false);
    });
  }
  $("configForm").addEventListener("submit", saveConfig);
  $("configForm").addEventListener("click", (event) => {
    const button = event.target.closest("[data-pick-path]");
    if (button) pickConfigPath(button);
  });
  $("sizeForm").addEventListener("submit", saveSizes);
  $("addSizeRow").addEventListener("click", addSizeRow);
  $("sizeRows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-size]");
    if (!button) return;
    const rows = $("sizeRows");
    if (rows.querySelectorAll("tr").length <= 1) {
      toast("Bảng phân loại cần ít nhất một cỡ", "error");
      return;
    }
    button.closest("tr")?.remove();
  });
  $("cfg_CONF_DET").addEventListener("input", updateConfidenceLabels);
  $("cfg_CONF_SEG").addEventListener("input", updateConfidenceLabels);
  $("runBtn").addEventListener("click", runPipeline);
  $("scaleModeBtn").addEventListener("click", enterScaleMode);
  $("applyScaleBtn").addEventListener("click", applyScale);
  $("cancelScaleBtn").addEventListener("click", cancelScaleMode);
  $("fileInput").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("folderInput").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("chooseFileBtn").addEventListener("click", () => {
    if (!state.running) $("fileInput").click();
  });
  $("chooseFolderBtn").addEventListener("click", () => {
    if (!state.running) $("folderInput").click();
  });
  $("inputPageSize").addEventListener("change", () => {
    state.inputPageSize = Number($("inputPageSize").value) || 5;
    renderInputFiles();
  });
  $("refreshResults").addEventListener("click", async () => {
    await loadRuns();
    await loadResults();
  });
  $("runSelect").addEventListener("change", loadResults);
  $("sizeFilter").addEventListener("change", async () => {
    state.resultPage = 1;
    const data = await requestJson(`/api/results${state.selectedRun ? `?run=${encodeURIComponent(state.selectedRun)}` : ""}`);
    state.currentResults = data;
    renderResults(data);
  });
  $("resultPageSize").addEventListener("change", () => {
    if (state.scaleMode) syncScaleMeasurements();
    state.resultPageSize = Number($("resultPageSize").value) || 25;
    state.resultPage = 1;
    if (state.currentResults) renderResults(state.currentResults);
  });
  $("resultPrevPage").addEventListener("click", () => {
    if (state.scaleMode) syncScaleMeasurements();
    state.resultPage -= 1;
    if (state.currentResults) renderResults(state.currentResults);
  });
  $("resultNextPage").addEventListener("click", () => {
    if (state.scaleMode) syncScaleMeasurements();
    state.resultPage += 1;
    if (state.currentResults) renderResults(state.currentResults);
  });
  $("resultsArea").addEventListener("input", (event) => {
    const input = event.target.closest("[data-scale-mm]");
    if (!input) return;
    const value = input.value.trim();
    if (value) {
      state.scaleMeasurements.set(input.dataset.measureKey, value);
    } else {
      state.scaleMeasurements.delete(input.dataset.measureKey);
    }
  });
  $("expandLog").addEventListener("click", () => {
    const consoleEl = document.querySelector(".log-console");
    const expanded = consoleEl.classList.toggle("expanded");
    $("expandLog").innerHTML = `${expanded ? icons.collapse : icons.expand} ${expanded ? "Thu gọn" : "Mở rộng"}`;
  });

  $("imageZoomOut").addEventListener("click", () => setImageZoom(state.imageZoom - 0.25));
  $("imageZoomIn").addEventListener("click", () => setImageZoom(state.imageZoom + 0.25));
  $("imageZoomReset").addEventListener("click", resetImageZoom);
  $("imagePrev").addEventListener("click", () => stepImage(-1));
  $("imageNext").addEventListener("click", () => stepImage(1));
  const imageView = document.querySelector(".image-view");
  imageView.addEventListener("pointerdown", beginImagePan);
  imageView.addEventListener("pointermove", moveImagePan);
  imageView.addEventListener("pointerup", endImagePan);
  imageView.addEventListener("pointercancel", endImagePan);
  imageView.addEventListener("pointerleave", endImagePan);
  $("imageModal").addEventListener("app-modal-hidden", () => {
    $("modalImage").removeAttribute("src");
    $("imageCaption").hidden = true;
    resetImagePan();
    state.currentImageBundle = null;
    state.currentImageIndex = 0;
    resetImageZoom();
  });

  $("fileList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-file]");
    if (button) deleteInputFile(button.dataset.deleteFile);
  });
  $("resultsArea").addEventListener("click", (event) => {
    const button = event.target.closest("[data-image-key]");
    if (!button || button.disabled) return;
    const bundle = state.imageBundles.get(button.dataset.imageKey);
    if (bundle) openImageModal(bundle);
  });

  const dropZone = $("dropZone");
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!state.running) dropZone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
}

async function init() {
  bindEvents();
  setRunning(false);
  updateScaleControls();
  try {
    await Promise.all([loadConfig(), loadSizes(), loadInputFiles(), loadRuns()]);
    await loadResults();
    await pollLog();
    await refreshStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

document.addEventListener("DOMContentLoaded", init);
