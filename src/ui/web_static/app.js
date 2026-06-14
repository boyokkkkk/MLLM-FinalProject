function app() {
  return {
    tab: "query",
    bootstrap: { app: {}, corpus: {} },
    healthOk: false,
    health: null,
    workspaceId: null,
    workspace: null,
    workspacePollTimer: null,
    messages: [],
    querySuggestions: [
      "回答图中的所有题目。",
      "继续解释第二题。",
      "总结这份材料的核心结论。",
    ],
    q: {
      question: "",
      context: "",
      scope: "workspace-first",
      temperature: 0.2,
      max_tokens: 1024,
      loading: false,
      images: [],
    },
    citationModal: {
      open: false,
      title: "",
      citation: null,
    },

    async init() {
      this.restoreState();
      await Promise.all([this.loadBootstrap(), this.loadHealth()]);
      if (this.workspaceId) {
        await this.refreshWorkspace();
        this.ensureWorkspacePolling();
      }
      this.$nextTick(() => this.attachInlineCitationHandlers());
      window.setInterval(() => this.loadHealth(), 15000);
    },

    persistState() {
      const fullState = {
        tab: this.tab,
        workspaceId: this.workspaceId,
        workspace: this.workspace,
        messages: this.messages,
        q: {
          scope: this.q.scope,
          temperature: this.q.temperature,
          max_tokens: this.q.max_tokens,
        },
      };
      const compactState = {
        tab: this.tab,
        workspaceId: this.workspaceId,
        workspace: this.workspace
          ? {
              workspace_id: this.workspace.workspace_id,
              status: this.workspace.status,
              progress: this.workspace.progress,
              progress_label: this.workspace.progress_label,
              last_error: this.workspace.last_error,
              counts: this.workspace.counts,
              assets: Array.isArray(this.workspace.assets)
                ? this.workspace.assets.map((asset) => ({
                    asset_id: asset.asset_id,
                    name: asset.name,
                    type: asset.type,
                    status: asset.status,
                    parser: asset.parser,
                    section_count: asset.section_count,
                    snippet: asset.snippet,
                  }))
                : [],
            }
          : null,
        messages: this.messages.slice(-12).map((message) => ({
          role: message.role,
          content: message.content,
          model: message.model || "",
        })),
        q: {
          scope: this.q.scope,
          temperature: this.q.temperature,
          max_tokens: this.q.max_tokens,
        },
      };

      try {
        window.sessionStorage.setItem("mllmproject_multirag_shell_v1", JSON.stringify(fullState));
      } catch (_error) {
        try {
          window.sessionStorage.setItem("mllmproject_multirag_shell_v1", JSON.stringify(compactState));
        } catch (_fallbackError) {
          try {
            window.sessionStorage.removeItem("mllmproject_multirag_shell_v1");
          } catch (_ignore) {
            // Ignore storage cleanup errors.
          }
        }
      }
    },

    restoreState() {
      try {
        const raw = window.sessionStorage.getItem("mllmproject_multirag_shell_v1");
        if (!raw) return;
        const saved = JSON.parse(raw);
        this.tab = saved.tab || "query";
        this.workspaceId = saved.workspaceId || null;
        this.workspace = saved.workspace || null;
        this.messages = Array.isArray(saved.messages) ? saved.messages : [];
        this.q.images = Array.isArray(saved.q?.images) ? saved.q.images : [];
        this.q.scope = saved.q?.scope || "workspace-first";
        this.q.temperature = Number(saved.q?.temperature ?? 0.2);
        this.q.max_tokens = Number(saved.q?.max_tokens ?? 1024);
      } catch (_error) {
        window.sessionStorage.removeItem("mllmproject_multirag_shell_v1");
      }
    },

    async loadBootstrap() {
      try {
        const response = await fetch("/api/v1/ui/bootstrap");
        if (!response.ok) return;
        this.bootstrap = await response.json();
      } catch (_error) {
        this.bootstrap = { app: {}, corpus: {} };
      }
    },

    async loadHealth() {
      try {
        const response = await fetch("/health");
        if (!response.ok) throw new Error("health failed");
        this.health = await response.json();
        this.healthOk = this.health && this.health.status === "ok";
      } catch (_error) {
        this.healthOk = false;
      }
    },

    healthLabel() {
      if (!this.healthOk) return "Backend disconnected";
      return `Healthy · v${this.health?.version || "-"}`;
    },

    currentTabLabel() {
      return {
        query: "Query mode",
        ingest: "Ingest mode",
        papers: "Workspace library",
      }[this.tab] || "Workspace";
    },

    workspaceAssetCount() {
      return (this.workspace?.assets || []).length;
    },

    workspaceStatusLabel() {
      if (!this.workspace) return "idle";
      return this.workspace.status || "idle";
    },

    workspaceSummaryPill() {
      if (!this.workspace) return "No workspace";
      return `${this.workspaceAssetCount()} assets · ${this.workspaceStatusLabel()}`;
    },

    workspaceProgressPercent() {
      const progress = Number(this.workspace?.progress || 0);
      return Math.max(0, Math.min(100, Math.round(progress * 100)));
    },

    switchTab(tab) {
      this.tab = tab;
      this.persistState();
    },

    showQueryIntro() {
      return !this.messages.length;
    },

    queryStatusLabel() {
      if (this.q.loading) return "Generating answer...";
      if (this.workspace?.status === "processing") return "Workspace assets are still processing.";
      if (this.workspace?.status === "failed") return `Workspace processing failed: ${this.workspace?.last_error || "unknown error"}`;
      if (this.q.images.length) return `This conversation is keeping ${this.q.images.length} temporary image(s) as follow-up context.`;
      return this.workspace ? "Conversation history will be kept for follow-up questions." : "You can also ask questions without uploading workspace files.";
    },

    setQuestion(question) {
      this.tab = "query";
      this.q.question = question;
      this.persistState();
    },

    newQuery() {
      this.tab = "query";
      this.messages = [];
      this.q.question = "";
      this.q.context = "";
      this.q.images = [];
      this.persistState();
    },

    async ensureWorkspace() {
      if (this.workspaceId) return;
      const response = await fetch("/api/v1/workspaces", { method: "POST" });
      if (!response.ok) {
        throw new Error("workspace creation failed");
      }
      this.workspace = await response.json();
      this.workspaceId = this.workspace.workspace_id;
      this.persistState();
    },

    async refreshWorkspace() {
      if (!this.workspaceId) return;
      const response = await fetch(`/api/v1/workspaces/${this.workspaceId}`);
      if (!response.ok) {
        throw new Error("workspace refresh failed");
      }
      this.workspace = await response.json();
      this.persistState();
    },

    ensureWorkspacePolling() {
      if (!this.workspaceId || this.workspacePollTimer) return;
      this.workspacePollTimer = window.setInterval(async () => {
        try {
          await this.refreshWorkspace();
          if (!["queued", "processing"].includes(this.workspace?.status)) {
            window.clearInterval(this.workspacePollTimer);
            this.workspacePollTimer = null;
          }
        } catch (_error) {
          window.clearInterval(this.workspacePollTimer);
          this.workspacePollTimer = null;
        }
      }, 2500);
    },

    async uploadWorkspaceFiles(event) {
      const files = Array.from(event.target.files || []);
      event.target.value = "";
      if (!files.length) return;
      await this.ensureWorkspace();
      const form = new FormData();
      for (const file of files) form.append("files", file);
      const response = await fetch(`/api/v1/workspaces/${this.workspaceId}/assets`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || "upload failed");
      }
      this.workspace = await response.json();
      this.persistState();
      this.ensureWorkspacePolling();
      this.tab = "ingest";
    },

    async removeAsset(assetId) {
      if (!this.workspaceId || !assetId) return;
      const response = await fetch(`/api/v1/workspaces/${this.workspaceId}/assets/${assetId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || "delete failed");
      }
      this.workspace = await response.json();
      this.persistState();
      this.ensureWorkspacePolling();
    },

    resetWorkspace() {
      if (this.workspacePollTimer) {
        window.clearInterval(this.workspacePollTimer);
        this.workspacePollTimer = null;
      }
      this.workspaceId = null;
      this.workspace = null;
      this.messages = [];
      this.q.context = "";
      this.q.images = [];
      this.persistState();
    },

    async handleImageSelect(event) {
      const files = Array.from(event.target.files || []);
      event.target.value = "";
      for (const file of files) {
        this.q.images.push({
          name: file.name,
          data_url: await this.fileToDataUrl(file),
        });
      }
      this.persistState();
    },

    removeImage(index) {
      this.q.images.splice(index, 1);
      this.persistState();
    },

    fileToDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("file read failed"));
        reader.readAsDataURL(file);
      });
    },

    buildHistoryPayload(limit = 8) {
      return this.messages
        .slice(-limit)
        .map((message) => ({
          role: message.role,
          content: String(message.content || "").trim(),
        }))
        .filter((message) => message.role === "user" || message.role === "assistant")
        .filter((message) => message.content);
    },

    formatScopeNote(scope) {
      if (scope === "workspace-first") return "Prioritize workspace context first.";
      if (scope === "context-only") return "Prioritize the pasted context only.";
      return "Prioritize global corpus retrieval.";
    },

    async runQuery() {
      const query = String(this.q.question || "").trim();
      if (!query || this.q.loading) return;

      this.q.loading = true;
      const historyPayload = this.buildHistoryPayload();
      const userMessage = { role: "user", content: query };
      this.messages.push(userMessage);
      this.persistState();

      const payload = {
        query,
        workspace_id: this.workspaceId,
        context: [
          `[ui_scope] ${this.formatScopeNote(this.q.scope)}`,
          ...(this.q.context.trim() ? [`[workspace_note]\n${this.q.context.trim()}`] : []),
        ],
        image_data_urls: this.q.images.map((item) => item.data_url),
        temperature: Number(this.q.temperature || 0.2),
        max_tokens: Number(this.q.max_tokens || 1024),
        history: historyPayload,
      };

      try {
        const response = await fetch("/api/v1/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const error = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(error.detail || "chat failed");
        }
        const data = await response.json();
        const assistantMessage = {
          role: "assistant",
          content: data.answer || "",
          citations: Array.isArray(data.citations) ? data.citations : [],
          model: data.model || "",
        };
        this.messages.push(assistantMessage);
        this.q.question = "";
        this.q.context = "";
        this.persistState();
        this.$nextTick(() => this.attachInlineCitationHandlers());
      } catch (error) {
        this.messages.push({
          role: "assistant",
          content: `Request failed: ${error.message}`,
          citations: [],
          model: "frontend-error",
        });
        this.persistState();
        this.$nextTick(() => this.attachInlineCitationHandlers());
      } finally {
        this.q.loading = false;
      }
    },

    sourceLabel(citation) {
      if (!citation?.source) return "Unknown source";
      return String(citation.source)
        .replace(/^workspace_file:/, "")
        .replace(/^workspace_note:/, "Pasted note")
        .replace(/^workspace_context:/, "Temporary context");
    },

    citationKey(citation, index) {
      return `${citation?.chunk_id || citation?.source || "citation"}-${index}`;
    },

    citationLocator(citation) {
      const parts = [];
      if (citation?.page) parts.push(`p.${citation.page}`);
      if (citation?.figure_id) parts.push(String(citation.figure_id));
      else if (citation?.figure_no) parts.push(`fig-${citation.figure_no}`);
      return parts.join(", ");
    },

    decorateAnswerText(text, messageIndex) {
      return String(text || "").replace(
        /\[(\d+)\]/g,
        `<span class="inline-citation" data-message-index="${messageIndex}" data-citation-index="$1">[$1]</span>`,
      );
    },

    renderMarkdown(text) {
      if (!text) return "";
      const html = typeof marked === "undefined" ? text : marked.parse(text);
      if (typeof renderMathInElement === "undefined") return html;
      try {
        const container = document.createElement("div");
        container.innerHTML = html;
        renderMathInElement(container, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ],
          throwOnError: false,
        });
        return container.innerHTML;
      } catch (_error) {
        return html;
      }
    },

    renderMessageAnswer(message, messageIndex) {
      return this.renderMarkdown(this.decorateAnswerText(message.content || "", messageIndex));
    },

    attachInlineCitationHandlers() {
      document.querySelectorAll(".inline-citation").forEach((node) => {
        node.onclick = () => {
          const messageIndex = Number(node.getAttribute("data-message-index"));
          const citationIndex = Number(node.getAttribute("data-citation-index")) - 1;
          const citation = this.messages[messageIndex]?.citations?.[citationIndex];
          if (citation) this.openCitation(citation, citationIndex);
        };
      });
    },

    openCitation(citation, index) {
      this.citationModal.open = true;
      this.citationModal.title = `Citation #${index + 1}`;
      this.citationModal.citation = citation;
    },

    closeCitationModal() {
      this.citationModal.open = false;
      this.citationModal.title = "";
      this.citationModal.citation = null;
    },

    closeOverlays() {
      this.closeCitationModal();
    },

    renderCitationSnippet() {
      return this.renderMarkdown(this.citationModal.citation?.snippet || "");
    },
  };
}
