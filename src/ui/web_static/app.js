const state = {
  bootstrap: null,
  health: null,
  tab: "chat",
  loading: false,
  workspaceId: null,
  workspace: null,
  workspacePollTimer: null,
  chatProgressTimer: null,
  history: [],
  suggestions: [
    "总结这份实验的核心结论。",
    "指出最能支持结论的部分。",
    "比较文中的关键实验结果。",
  ],
};

const APP_STATE_KEY = "mllmproject_ui_state_v1";

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTextBlock(value) {
  return escapeHtml(value ?? "").replace(/\n/g, "<br>");
}

function persistState() {
  const payload = {
    workspaceId: state.workspaceId,
    workspace: state.workspace,
    history: state.history.filter((item) => item.role === "user" || item.role === "assistant"),
  };
  window.sessionStorage.setItem(APP_STATE_KEY, JSON.stringify(payload));
}

function restoreState() {
  try {
    const raw = window.sessionStorage.getItem(APP_STATE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.workspaceId = saved.workspaceId || null;
    state.workspace = saved.workspace || null;
    state.history = Array.isArray(saved.history) ? saved.history : [];
  } catch (_error) {
    window.sessionStorage.removeItem(APP_STATE_KEY);
  }
}

function buildHistoryPayload(limit = 8) {
  return state.history
    .filter((item) => item.role === "user" || item.role === "assistant")
    .slice(-limit)
    .map((item) => ({
      role: item.role,
      content: String(item.content || "").trim(),
    }))
    .filter((item) => item.content);
}

function humanNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return new Intl.NumberFormat("zh-CN").format(Number(value));
}

function showBanner(message, tone = "loading") {
  const banner = $("app-banner");
  banner.textContent = message;
  banner.className = `banner is-${tone}`;
}

function hideBanner() {
  const banner = $("app-banner");
  banner.textContent = "";
  banner.className = "banner banner-hidden";
}

function setActiveTab(tab) {
  state.tab = tab;
  document.querySelectorAll("[data-tab-target]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tabTarget === tab);
  });
  document.querySelectorAll(".tab-section").forEach((section) => {
    section.classList.toggle("is-active", section.id === `tab-${tab}`);
  });
}

function setLoading(loading) {
  state.loading = loading;
  const submit = $("submit-btn");
  submit.disabled = loading;
  submit.querySelector(".button-label").textContent = loading ? "处理中..." : "发送";
}

function formatScopeNote(scope) {
  if (scope === "workspace-first") {
    return "优先使用工作区内容，再补充全局检索结果。";
  }
  if (scope === "context-only") {
    return "仅使用工作区和补充上下文。";
  }
  return "优先使用全局语料。";
}

function scopeDisplayLabel(scope) {
  if (scope === "workspace-first") return "工作区优先";
  if (scope === "context-only") return "仅上下文";
  return "全局语料";
}

function collectContext() {
  const context = [];
  const scope = $("scope-select").value;
  const pastedContext = $("context-input").value.trim();
  context.push(`[ui_scope] ${formatScopeNote(scope)}`);
  if (pastedContext) {
    context.push(`[workspace_note]\n${pastedContext}`);
  }
  return context;
}

function renderSuggestions() {
  $("suggestions").innerHTML = state.suggestions
    .map(
      (prompt) =>
        `<button class="suggestion-btn" type="button" data-suggestion="${escapeHtml(prompt)}">${escapeHtml(prompt)}</button>`,
    )
    .join("");

  document.querySelectorAll("[data-suggestion]").forEach((button) => {
    button.addEventListener("click", () => {
      $("question-input").value = button.dataset.suggestion || "";
      $("question-input").focus();
    });
  });
}

function renderWorkspaceStatus() {
  const container = $("workspace-status");
  const workspace = state.workspace;
  if (!workspace) {
    container.className = "empty-state";
    container.textContent = "暂无工作区，请先上传文件。";
    return;
  }

  const progress = Math.max(0, Math.min(100, Math.round(Number(workspace.progress || 0) * 100)));
  const counts = workspace.counts || {};
  const errorBlock = workspace.last_error
    ? `<div style="margin-top:12px;color:#a44b3f;">${escapeHtml(workspace.last_error)}</div>`
    : "";
  container.className = "workspace-status-card";
  container.innerHTML = `
    <div class="section-label">工作区</div>
    <div class="workspace-status-id">${escapeHtml(workspace.workspace_id || "")}</div>
    <div class="workspace-status-copy">${escapeHtml(workspace.progress_label || workspace.stage || "idle")}</div>
    <div class="progress-track"><div class="progress-fill" style="width:${progress}%;"></div></div>
    <div class="workspace-status-metrics">
      状态：${escapeHtml(workspace.status || "idle")} | 文档：${escapeHtml(humanNumber(counts.documents))} | 分块：${escapeHtml(humanNumber(counts.chunks))} | 视觉：${escapeHtml(humanNumber(counts.visual_items))}
    </div>
    ${errorBlock}
  `;
}

function renderWorkspaceAssets() {
  const container = $("workspace-assets");
  const assets = state.workspace?.assets || [];
  const items = assets.map((asset) => `
    <article class="asset-chip asset-row">
      <div class="asset-row-copy">
        <strong>${escapeHtml(asset.name || asset.stored_name || "文件")}</strong>
        <small>${escapeHtml(asset.status || "uploaded")} | ${escapeHtml(asset.parser || "pending")} | ${escapeHtml(asset.type || "file")}</small>
      </div>
      <button class="asset-delete-btn" type="button" data-delete-asset="${escapeHtml(asset.asset_id || "")}" ${["queued", "processing"].includes(state.workspace?.status) ? "disabled" : ""}>删除</button>
    </article>
  `);

  if (!items.length) {
    container.className = "asset-list empty-state";
    container.textContent = "暂无文件。";
    return;
  }

  container.className = "asset-list";
  container.innerHTML = [
    `<article class="asset-chip asset-row"><div class="asset-row-copy"><strong>模式</strong><small>${escapeHtml(scopeDisplayLabel($("scope-select").value))}</small></div></article>`,
    ...items,
  ].join("");

  container.querySelectorAll("[data-delete-asset]").forEach((button) => {
    button.addEventListener("click", async () => {
      const assetId = button.getAttribute("data-delete-asset");
      if (!assetId || !state.workspaceId) return;
      try {
        showBanner("正在删除文件并重建索引...", "loading");
        const response = await fetch(`/api/v1/workspaces/${state.workspaceId}/assets/${assetId}`, { method: "DELETE" });
        if (!response.ok) {
          const error = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(error.detail || "删除失败");
        }
        state.workspace = await response.json();
        persistState();
        renderWorkspaceStatus();
        renderWorkspaceAssets();
        if (["queued", "processing"].includes(state.workspace.status)) {
          ensureWorkspacePolling();
        }
      } catch (error) {
        showBanner(`删除失败：${error.message}`, "error");
      }
    });
  });
}

function renderConversation() {
  const container = $("conversation");
  if (!state.history.length) {
    container.className = "conversation empty-state";
    container.textContent = "请输入问题。";
    return;
  }

  container.className = "conversation";
  container.innerHTML = state.history
    .map((item) => {
      if (item.role === "user") {
        return `
          <article class="message-card user">
            <div class="message-meta">
              <span class="message-role">用户</span>
            </div>
            <div class="message-body">${renderTextBlock(item.content)}</div>
          </article>
        `;
      }

      if (item.role === "progress") {
        return `
          <article class="message-card assistant">
            <div class="message-meta">
              <span class="message-role">助手</span>
              <span>处理中</span>
            </div>
            <div class="message-body">${renderTextBlock(item.content)}</div>
          </article>
        `;
      }

      const citations = (item.citations || [])
        .map((citation, index) => {
          const isWorkspace = String(citation.citation_kind || "").startsWith("workspace") || String(citation.source || "").startsWith("workspace_");
          const fileLabel = isWorkspace
            ? String(citation.source || "")
                .replace(/^workspace_file:/, "")
                .replace(/^workspace_note:/, "")
                .replace(/^workspace_context:/, "")
            : String(citation.source || "未知来源");
          const sectionTitle = citation.section_title || (isWorkspace ? "工作区内容" : "");
          const locator = isWorkspace ? "工作区" : (citation.page != null ? `第 ${escapeHtml(citation.page)} 页` : "页码未知");
          const tag = isWorkspace ? "工作区检索" : "语料检索";
          return `
            <article class="citation-card">
              <header>
                <span>#${index + 1} ${escapeHtml(fileLabel)}</span>
                <span>${locator}</span>
              </header>
              <div class="message-role" style="margin-bottom:8px;">${escapeHtml(tag)}</div>
              ${sectionTitle ? `<div style="margin-bottom:8px;font-weight:600;">${escapeHtml(sectionTitle)}</div>` : ""}
              <div>${escapeHtml(citation.snippet || "")}</div>
            </article>
          `;
        })
        .join("");

      return `
        <article class="message-card assistant">
          <div class="message-meta">
            <span class="message-role">助手</span>
            <span>${escapeHtml(item.model || "")}</span>
          </div>
          <div class="message-body">${renderTextBlock(item.content)}</div>
          ${
            citations
              ? `<div class="citation-list">${citations}</div>`
              : `<div class="empty-state" style="margin-top:16px;">本次回答未返回引用。</div>`
          }
        </article>
      `;
    })
    .join("");

  container.scrollTop = container.scrollHeight;
}

function renderBackendConsole() {
  const services = state.bootstrap?.services || {};
  const models = state.bootstrap?.models || {};
  const retrieval = state.bootstrap?.retrieval || {};

  $("service-console").innerHTML = [
    {
      name: "健康检查",
      detail: `${services.health?.method || "GET"} ${services.health?.path || "/health"}`,
    },
    {
      name: "对话",
      detail: `${services.chat?.method || "POST"} ${services.chat?.path || ""}`,
    },
    {
      name: "工作区",
      detail: "POST /api/v1/workspaces | upload | poll | retrieve",
    },
    {
      name: "VLM",
      detail: models.vlm?.model || "-",
    },
    {
      name: "文本向量",
      detail: models.text_embedding?.model || "-",
    },
    {
      name: "视觉向量",
      detail: models.vision_embedding?.model || "-",
    },
  ]
    .map(
      (item) => `
        <article class="stack-item">
          <strong>${escapeHtml(item.name)}</strong>
          <small><code>${escapeHtml(item.detail)}</code></small>
        </article>
      `,
    )
    .join("");

  $("retrieval-console").innerHTML = [
    `文本 Top-K: ${retrieval.top_k_text ?? "-"}`,
    `重排: ${retrieval.rerank ? "开" : "关"}`,
    `类型感知重排: ${retrieval.query_type_aware_rerank ? "开" : "关"}`,
    `视觉融合: ${retrieval.visual_fusion ? "开" : "关"}`,
    `稠密视觉融合: ${retrieval.visual_dense_fusion ? "开" : "关"}`,
    `图像感知重排: ${retrieval.query_image_aware_rerank ? "开" : "关"}`,
    `视觉生成辅助: ${retrieval.generation_visual_assist ? "开" : "关"}`,
    `默认温度: ${retrieval.default_temperature ?? "-"}`,
    `默认最大输出: ${retrieval.default_max_tokens ?? "-"}`,
  ]
    .map(
      (line) => `
        <article class="stack-item">
          <strong>${escapeHtml(line.split(":")[0])}</strong>
          <small>${escapeHtml(line.split(":").slice(1).join(":").trim())}</small>
        </article>
      `,
    )
    .join("");
}

function renderServiceCards() {
  const services = state.bootstrap?.services || {};
  const models = state.bootstrap?.models || {};
  const cards = [
    {
      title: "对话接口",
      rows: {
        方法: services.chat?.method || "POST",
        路径: services.chat?.path || "-",
        模型: models.vlm?.model || "-",
        提供方: models.vlm?.provider || "-",
        地址: models.vlm?.base_url || "-",
      },
    },
    {
      title: "工作区入库",
      rows: {
        创建: "POST /api/v1/workspaces",
        上传: "POST /api/v1/workspaces/{id}/assets",
        查询: "GET /api/v1/workspaces/{id}",
        模式: "工作区优先 / 仅上下文 / 全局语料",
      },
    },
    {
      title: "文本向量接口",
      rows: {
        方法: services.embed_text?.method || "POST",
        路径: services.embed_text?.path || "-",
        模型: models.text_embedding?.model || "-",
        提供方: models.text_embedding?.provider || "-",
      },
    },
    {
      title: "视觉向量接口",
      rows: {
        方法: services.embed_vision?.method || "POST",
        路径: services.embed_vision?.path || "-",
        模型: models.vision_embedding?.model || "-",
        提供方: models.vision_embedding?.provider || "-",
      },
    },
  ];

  $("service-cards").innerHTML = cards
    .map(
      (card) => `
        <article class="service-card">
          <h4>${escapeHtml(card.title)}</h4>
          <dl>
            ${Object.entries(card.rows)
              .map(
                ([key, value]) => `
                  <div>
                    <dt>${escapeHtml(key)}</dt>
                    <dd>${escapeHtml(value)}</dd>
                  </div>
                `,
              )
              .join("")}
          </dl>
        </article>
      `,
    )
    .join("");
}

function buildTable(rows, limit = 6) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0]);
  return `
    <table>
      <thead>
        <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .slice(0, limit)
          .map(
            (row) => `
              <tr>
                ${columns.map((column) => `<td>${escapeHtml(row[column] ?? "-")}</td>`).join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderEvaluation() {
  const evaluation = state.bootstrap?.evaluation || {};
  const summary = evaluation.manifest_summary || {};
  const repair = evaluation.repair_stats || {};
  const errors = evaluation.error_summary || {};

  $("evaluation-summary").innerHTML = [
    { label: "总行数", value: summary.num_rows },
    { label: "文档页", value: summary.unique_doc_pages },
    { label: "原始重复", value: repair.original_duplicate },
    { label: "命中", value: errors.category_counts?.clean_hit },
    { label: "生成问题", value: errors.category_counts?.generation_issue },
    { label: "真实漏检", value: errors.category_counts?.true_miss },
  ]
    .map(
      (item) => `
        <article class="summary-card">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(humanNumber(item.value))}</strong>
        </article>
      `,
    )
    .join("");

  const mainResultsMarkup = buildTable(evaluation.main_results || [], 8);
  $("main-results-table").className = mainResultsMarkup ? "table-wrap" : "table-wrap empty-state";
  $("main-results-table").innerHTML = mainResultsMarkup || "未找到结果表。";

  const errorMarkup = buildTable(evaluation.error_breakdown || [], 9);
  $("error-breakdown-table").className = errorMarkup ? "table-wrap" : "table-wrap empty-state";
  $("error-breakdown-table").innerHTML = errorMarkup || "未找到错误表。";

  const figures = evaluation.figure_urls || [];
  const gallery = $("figure-gallery");
  if (!figures.length) {
    gallery.className = "figure-gallery empty-state";
    gallery.textContent = "未找到图表。";
    return;
  }

  gallery.className = "figure-gallery";
  gallery.innerHTML = figures
    .slice(0, 6)
    .map((url) => {
      const name = url.split("/").pop() || "figure";
      return `
        <article class="figure-card panel">
          <h4>${escapeHtml(name)}</h4>
          <img src="${escapeHtml(url)}" alt="${escapeHtml(name)}">
          <p>来自本地评测产物。</p>
        </article>
      `;
    })
    .join("");
}

function renderBootstrap() {
  const bootstrap = state.bootstrap;
  if (!bootstrap) return;

  $("metric-docs").textContent = humanNumber(bootstrap.corpus?.documents);
  $("metric-chunks").textContent = humanNumber(bootstrap.corpus?.chunks);
  $("metric-visual").textContent = humanNumber(bootstrap.corpus?.visual_descriptors);
  $("metric-version").textContent = bootstrap.app?.version || "-";

  $("pill-model").textContent = `模型：${bootstrap.models?.vlm?.model || "-"}`;
  $("pill-rerank").textContent = `重排：${bootstrap.retrieval?.rerank ? "开" : "关"}`;
  $("pill-vision").textContent = `视觉辅助：${bootstrap.retrieval?.generation_visual_assist ? "开" : "关"}`;

  renderBackendConsole();
  renderServiceCards();
  renderEvaluation();
}

function updateHealth(health, failed = false) {
  const dot = $("health-dot");
  const text = $("health-text");

  dot.classList.remove("is-ok", "is-error");

  if (failed) {
    dot.classList.add("is-error");
    text.textContent = "连接失败";
    return;
  }

  if (health?.status === "ok") {
    dot.classList.add("is-ok");
    text.textContent = `正常 | v${health.version || "-"}`;
    return;
  }

  text.textContent = "未知";
}

async function createWorkspace() {
  const response = await fetch("/api/v1/workspaces", { method: "POST" });
  if (!response.ok) {
    throw new Error(`创建工作区失败：${response.status}`);
  }
  const data = await response.json();
  state.workspaceId = data.workspace_id;
  state.workspace = data;
  persistState();
  renderWorkspaceStatus();
  renderWorkspaceAssets();
  return data;
}

async function refreshWorkspace() {
  if (!state.workspaceId) return null;
  const response = await fetch(`/api/v1/workspaces/${state.workspaceId}`);
  if (!response.ok) {
    throw new Error(`工作区查询失败：${response.status}`);
  }
  const data = await response.json();
  state.workspace = data;
  persistState();
  renderWorkspaceStatus();
  renderWorkspaceAssets();
  return data;
}

function stopWorkspacePolling() {
  if (state.workspacePollTimer) {
    window.clearInterval(state.workspacePollTimer);
    state.workspacePollTimer = null;
  }
}

function ensureWorkspacePolling() {
  if (state.workspacePollTimer || !state.workspaceId) return;
  state.workspacePollTimer = window.setInterval(async () => {
    try {
      const workspace = await refreshWorkspace();
      if (!workspace) return;
      if (!["processing", "queued"].includes(workspace.status)) {
        stopWorkspacePolling();
      }
    } catch (error) {
      stopWorkspacePolling();
      showBanner(`工作区查询失败：${error.message}`, "error");
    }
  }, 2500);
}

async function uploadWorkspaceFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (!state.workspaceId) {
    await createWorkspace();
  }
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  showBanner("正在上传并处理文件...", "loading");
  const response = await fetch(`/api/v1/workspaces/${state.workspaceId}/assets`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "上传失败");
  }
  const data = await response.json();
  state.workspace = data;
  persistState();
  renderWorkspaceStatus();
  renderWorkspaceAssets();
  ensureWorkspacePolling();
}

function startChatProgress() {
  const messages = [
    "正在准备请求...",
    "正在检索内容...",
    "正在生成回答...",
  ];
  state.history.push({ role: "progress", content: messages[0] });
  renderConversation();
  let index = 1;
  state.chatProgressTimer = window.setInterval(() => {
    const progressItem = state.history.find((item) => item.role === "progress");
    if (!progressItem) return;
    progressItem.content = messages[Math.min(index, messages.length - 1)];
    index += 1;
    renderConversation();
  }, 1200);
}

function stopChatProgress() {
  if (state.chatProgressTimer) {
    window.clearInterval(state.chatProgressTimer);
    state.chatProgressTimer = null;
  }
  state.history = state.history.filter((item) => item.role !== "progress");
  persistState();
}

async function submitChat() {
  const query = $("question-input").value.trim();
  if (!query) {
    showBanner("请输入问题。", "error");
    return;
  }
  if (state.workspace && ["processing", "queued"].includes(state.workspace.status)) {
    showBanner("文件仍在处理中，请稍后再问。", "error");
    return;
  }

  hideBanner();
  setLoading(true);
  const history = buildHistoryPayload();

  const payload = {
    query,
    workspace_id: state.workspaceId,
    context: collectContext(),
    image_data_urls: [],
    temperature: Number($("temperature-input").value || 0.2),
    max_tokens: Number($("max-tokens-input").value || 512),
    history,
  };

  state.history.push({ role: "user", content: query });
  persistState();
  startChatProgress();

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || "对话失败");
    }

    const data = await response.json();
    stopChatProgress();
    state.history.push({
      role: "assistant",
      content: data.answer || "",
      citations: data.citations || [],
      model: data.model || "",
    });
    persistState();
    renderConversation();
    showBanner("回答完成。", "loading");
  } catch (error) {
    stopChatProgress();
    state.history.push({
      role: "assistant",
      content: `请求失败：${error.message}`,
      citations: [],
      model: "前端错误",
    });
    persistState();
    renderConversation();
    showBanner(`请求失败：${error.message}`, "error");
  } finally {
    setLoading(false);
  }
}

async function loadBootstrap() {
  const response = await fetch("/api/v1/ui/bootstrap");
  if (!response.ok) {
    throw new Error(`初始化失败：${response.status}`);
  }
  state.bootstrap = await response.json();
  renderBootstrap();
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`健康检查失败：${response.status}`);
    state.health = await response.json();
    updateHealth(state.health, false);
  } catch (_error) {
    updateHealth(null, true);
  }
}

function bindEvents() {
  document.querySelectorAll("[data-tab-target]").forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tabTarget));
  });

  $("submit-btn").addEventListener("click", submitChat);
  $("question-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!state.loading) {
        submitChat();
      }
    }
  });

  $("clear-history-btn").addEventListener("click", () => {
    state.history = [];
    persistState();
    renderConversation();
  });

  $("clear-workspace-btn").addEventListener("click", () => {
    $("text-files-input").value = "";
    $("image-files-input").value = "";
    $("context-input").value = "";
    renderWorkspaceAssets();
  });

  $("new-workspace-btn").addEventListener("click", async () => {
    stopWorkspacePolling();
    state.workspaceId = null;
    state.workspace = null;
    persistState();
    renderWorkspaceStatus();
    renderWorkspaceAssets();
    showBanner("已清空工作区。", "loading");
  });

  $("text-files-input").addEventListener("change", async (event) => {
    try {
      await uploadWorkspaceFiles(event.target.files);
      event.target.value = "";
    } catch (error) {
      showBanner(`上传失败：${error.message}`, "error");
    }
  });

  $("image-files-input").addEventListener("change", async (event) => {
    try {
      await uploadWorkspaceFiles(event.target.files);
      event.target.value = "";
    } catch (error) {
      showBanner(`上传失败：${error.message}`, "error");
    }
  });

  $("scope-select").addEventListener("change", renderWorkspaceAssets);
}

async function init() {
  restoreState();
  renderSuggestions();
  renderWorkspaceStatus();
  renderWorkspaceAssets();
  renderConversation();
  bindEvents();

  showBanner("正在连接服务...", "loading");
  try {
    await Promise.all([loadBootstrap(), loadHealth()]);
    if (state.workspaceId) {
      await refreshWorkspace().catch(() => null);
      if (state.workspace && ["queued", "processing"].includes(state.workspace.status)) {
        ensureWorkspacePolling();
      }
    }
    hideBanner();
  } catch (error) {
    showBanner(`初始化失败：${error.message}`, "error");
  }

  window.setInterval(loadHealth, 15000);
}

window.addEventListener("DOMContentLoaded", init);
