"use strict";

const state = {
  data: null,
  selectedModels: new Set(),
  filters: {
    search: "",
    model: "",
    quant: "",
    profile: "",
    tag: "",
    platform: "",
    engine: "",
    backend: "",
    status: "",
    difficulty: "",
    sort: "task",
  },
};

const elements = {
  loading: document.querySelector("#loading"),
  error: document.querySelector("#load-error"),
  content: document.querySelector("#app-content"),
  introFacts: document.querySelector("#intro-facts"),
  timestamp: document.querySelector("#data-timestamp"),
  search: document.querySelector("#search"),
  model: document.querySelector("#model-filter"),
  quant: document.querySelector("#quant-filter"),
  profile: document.querySelector("#profile-filter"),
  tag: document.querySelector("#tag-filter"),
  platform: document.querySelector("#platform-filter"),
  engine: document.querySelector("#engine-filter"),
  backend: document.querySelector("#backend-filter"),
  status: document.querySelector("#status-filter"),
  difficulty: document.querySelector("#difficulty-filter"),
  sort: document.querySelector("#sort-tasks"),
  summary: document.querySelector("#active-summary"),
  modelGrid: document.querySelector("#model-grid"),
  comparison: document.querySelector("#run-comparison"),
  matrix: document.querySelector("#task-matrix"),
  dialog: document.querySelector("#detail-dialog"),
  dialogKicker: document.querySelector("#dialog-kicker"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogContent: document.querySelector("#dialog-content"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  if (value === null || value === undefined) return "not recorded";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatExactNumber(value) {
  if (value === null || value === undefined) return "not recorded";
  return new Intl.NumberFormat("en").format(value);
}

function formatBytes(value) {
  if (!value) return "not recorded";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index > 2 ? 1 : 0)} ${units[index]}`;
}

function formatDuration(milliseconds) {
  if (milliseconds === null || milliseconds === undefined) return "not recorded";
  let seconds = Math.max(0, Math.round(milliseconds / 1000));
  const days = Math.floor(seconds / 86400);
  seconds %= 86400;
  const hours = Math.floor(seconds / 3600);
  seconds %= 3600;
  const minutes = Math.floor(seconds / 60);
  seconds %= 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  if (seconds || parts.length === 0) parts.push(`${seconds}s`);
  return parts.slice(0, 3).join(" ");
}

function formatPercent(rate) {
  return `${(Number(rate || 0) * 100).toFixed(0)}%`;
}

function scoreTone(rate) {
  const percentage = Number(rate || 0) * 100;
  if (percentage >= 80) return "good";
  if (percentage >= 50) return "watch";
  return "low";
}

function scoreBlock(label, rate, primary = false) {
  const tone = scoreTone(rate);
  const percentage = Math.max(0, Math.min(100, Math.round(Number(rate || 0) * 100)));
  return `
    <div class="score-block${primary ? " primary" : ""} score-${tone}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(formatPercent(rate))}</strong>
      <div class="score-meter" aria-hidden="true"><span style="--score-fill: ${percentage}%"></span></div>
    </div>
  `;
}

function comparisonScore(passed, total, rate) {
  const tone = scoreTone(rate);
  const percentage = Math.max(0, Math.min(100, Math.round(Number(rate || 0) * 100)));
  return `<span class="comparison-score score-${tone}"><span>${escapeHtml(`${passed}/${total} (${formatPercent(rate)})`)}</span><i aria-hidden="true" style="--score-fill: ${percentage}%"></i></span>`;
}

function formatDate(value) {
  if (!value) return "not recorded";
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value)) + " UTC";
}

function modelLabel(model) {
  return [model.name, model.quant, model.inferenceProfile, model.tag].filter(Boolean).join(" · ");
}

function shortModelLabel(model) {
  return [model.name, model.quant || "quant not recorded", model.inferenceProfile, model.tag]
    .filter(Boolean)
    .join(" / ");
}

function outcomeLabel(type) {
  return {
    passed: "Pass",
    "verifier-failure": "Verifier fail",
    "agent-timeout": "Agent timeout",
    "endpoint-stall": "Endpoint stall",
    "harness-error": "Harness error",
    missing: "Missing",
  }[type] || type || "Unknown";
}

function resultFor(model, taskId) {
  return model.results.find((result) => result.taskId === taskId);
}

function uniqueOptions(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function fillSelect(select, values) {
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
}

function modelSearchBlob(model) {
  return [
    model.name,
    model.modelId,
    model.quant,
    model.tag,
    model.quantizationType,
    model.engine,
    model.engineVersion,
    model.backend,
    model.backendVersion,
    model.platform.id,
    model.platform.name,
    model.endpointOwner,
    model.inferenceProfile,
  ].filter(Boolean).join(" ").toLowerCase();
}

function visibleModels() {
  const models = state.data.models.filter((model) => {
    if (state.filters.model && model.name !== state.filters.model) return false;
    if (state.filters.quant && model.quant !== state.filters.quant) return false;
    if (state.filters.profile && model.inferenceProfile !== state.filters.profile) return false;
    if (state.filters.tag && model.tag !== state.filters.tag) return false;
    if (state.filters.platform && model.platform.id !== state.filters.platform) return false;
    if (state.filters.engine && model.engine !== state.filters.engine) return false;
    if (state.filters.backend && model.backend !== state.filters.backend) return false;
    return true;
  });
  const query = state.filters.search.trim().toLowerCase();
  if (!query) return models;
  const matchingModels = models.filter((model) => modelSearchBlob(model).includes(query));
  return matchingModels.length ? matchingModels : models;
}

function comparedModels() {
  return visibleModels().filter((model) => state.selectedModels.has(model.id));
}

function taskSearchBlob(task, models) {
  const resultText = models.flatMap((model) => {
    const result = resultFor(model, task.id);
    return result
      ? [modelLabel(model), model.engine, model.backend, model.platform.name, result.reason]
      : [];
  });
  return [
    task.id,
    task.name,
    task.description,
    task.category,
    task.difficulty,
    ...(task.keywords || []),
    ...(task.tags || []),
    ...resultText,
  ].join(" ").toLowerCase();
}

function resultMatchesStatus(result, status) {
  if (!result) return false;
  if (status === "passed") return result.passed;
  if (status === "failed") return !result.passed;
  if (status === "timeout") {
    return result.attempts.some((attempt) => ["agent-timeout", "endpoint-stall"].includes(attempt.outcomeType));
  }
  if (status === "harness-error") {
    return result.attempts.some((attempt) => attempt.outcomeType === "harness-error");
  }
  return true;
}

function filteredTasks(models) {
  const query = state.filters.search.trim().toLowerCase();
  const tasks = state.data.tasks.filter((task) => {
    if (state.filters.difficulty && task.difficulty !== state.filters.difficulty) return false;
    if (query && !taskSearchBlob(task, models).includes(query)) return false;
    if (state.filters.status) {
      return models.some((model) => resultMatchesStatus(resultFor(model, task.id), state.filters.status));
    }
    return true;
  });

  const difficultyOrder = { hard: 0, medium: 1, easy: 2, unknown: 3 };
  const averageDuration = (task) => {
    const durations = models.map((model) => resultFor(model, task.id)?.durationMs).filter((value) => value !== undefined);
    return durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : 0;
  };
  const failureCount = (task) => models.filter((model) => resultFor(model, task.id)?.passed === false).length;

  tasks.sort((a, b) => {
    if (state.filters.sort === "failures") return failureCount(b) - failureCount(a) || a.id.localeCompare(b.id);
    if (state.filters.sort === "duration") return averageDuration(b) - averageDuration(a) || a.id.localeCompare(b.id);
    if (state.filters.sort === "difficulty") {
      return (difficultyOrder[a.difficulty] ?? 4) - (difficultyOrder[b.difficulty] ?? 4) || a.id.localeCompare(b.id);
    }
    if (state.filters.sort === "category") return a.category.localeCompare(b.category) || a.id.localeCompare(b.id);
    return a.id.localeCompare(b.id);
  });
  return tasks;
}

function renderIntro() {
  const runCount = state.data.models.length;
  const platforms = uniqueOptions(state.data.models.map((model) => model.platform.name)).length;
  elements.introFacts.innerHTML = [
    `${state.data.tasks.length} tasks`,
    `${runCount} committed run${runCount === 1 ? "" : "s"}`,
    `${platforms} platform${platforms === 1 ? "" : "s"}`,
    "up to 2 attempts",
    "3h per attempt",
  ].map((fact) => `<span class="fact">${escapeHtml(fact)}</span>`).join("");
  elements.timestamp.textContent = `Dataset generated ${formatDate(state.data.generatedAt)}`;
}

function renderFilters() {
  fillSelect(elements.model, uniqueOptions(state.data.models.map((model) => model.name)));
  fillSelect(elements.quant, uniqueOptions(state.data.models.map((model) => model.quant)));
  fillSelect(elements.profile, uniqueOptions(state.data.models.map((model) => model.inferenceProfile)));
  fillSelect(elements.tag, uniqueOptions(state.data.models.map((model) => model.tag)));
  fillSelect(elements.platform, uniqueOptions(state.data.models.map((model) => model.platform.id)));
  for (const option of [...elements.platform.options]) {
    if (!option.value) continue;
    const names = uniqueOptions(state.data.models.filter((model) => model.platform.id === option.value).map((model) => model.platform.name));
    option.textContent = names.length === 1 ? `${names[0]} (${option.value})` : `${option.value} (${names.join(" / ")})`;
  }
  fillSelect(elements.backend, uniqueOptions(state.data.models.map((model) => model.backend)));
  fillSelect(elements.engine, uniqueOptions(state.data.models.map((model) => model.engine)));
  fillSelect(elements.difficulty, uniqueOptions(state.data.tasks.map((task) => task.difficulty)));
}

function renderSummary(tasks, models) {
  const cells = models.flatMap((model) => tasks.map((task) => resultFor(model, task.id))).filter(Boolean);
  const passed = cells.filter((result) => result.passed).length;
  const timeouts = cells.filter((result) => result.attempts.some((attempt) => ["agent-timeout", "endpoint-stall"].includes(attempt.outcomeType))).length;
  const passRate = cells.length ? passed / cells.length : 0;
  elements.summary.innerHTML = `
    <span><strong>${tasks.length}</strong> tasks shown</span>
    <span><strong>${models.length}</strong> model columns</span>
    <span class="summary-score score-${scoreTone(passRate)}"><strong>${passed}/${cells.length || 0}</strong> passing cells</span>
    <span class="summary-timeout"><strong>${timeouts}</strong> cells with timeout/stall attempts</span>
  `;
}

function renderModels() {
  const models = visibleModels();
  if (!models.length) {
    elements.modelGrid.innerHTML = '<div class="empty-state">No model runs match the active identity and runtime filters.</div>';
    return;
  }
  elements.modelGrid.innerHTML = models.map((model) => {
    const selected = state.selectedModels.has(model.id);
    const average = model.totalTasks ? model.totalDurationMs / model.totalTasks : 0;
    const primaryTone = scoreTone(model.passWithinAttemptsRate);
    return `
      <article class="model-card score-${primaryTone}${selected ? " selected" : ""}" data-model-id="${escapeHtml(model.id)}">
        <div class="model-card-head">
          <div>
            <p class="model-name">${escapeHtml(model.name)}</p>
            <p class="model-tag">${escapeHtml(model.quant || "quant not recorded")}</p>
          </div>
          <button class="model-select" type="button" data-action="toggle-model" aria-label="${selected ? "Remove" : "Add"} ${escapeHtml(modelLabel(model))} ${selected ? "from" : "to"} comparison" aria-pressed="${selected}">${selected ? "✓" : ""}</button>
        </div>
        <div class="score-row">
          ${scoreBlock("passed within attempts", model.passWithinAttemptsRate, true)}
          ${scoreBlock("pass@1", model.passAt1Rate)}
        </div>
        <div class="model-meta">
          ${model.inferenceProfile ? `<span class="badge">profile ${escapeHtml(model.inferenceProfile)}</span>` : ""}
          ${model.tag ? `<span class="badge">tag ${escapeHtml(model.tag)}</span>` : ""}
          <span class="badge">${escapeHtml(model.platform.name)}</span>
          <span class="badge">${escapeHtml(model.engine || "engine not recorded")}</span>
          <span class="badge">${escapeHtml(model.backend || "backend not recorded")}${model.backendVersion ? ` ${escapeHtml(model.backendVersion)}` : ""}</span>
        </div>
        <div class="model-card-footer">
          <span class="profile-note">${model.passedWithinAttempts}/${model.totalTasks} passed within attempts · ${escapeHtml(formatDuration(average))} avg</span>
          <button class="details-button" type="button" data-action="model-details">Run profile →</button>
        </div>
      </article>
    `;
  }).join("");
}

function renderComparison(models) {
  if (!models.length) {
    elements.comparison.innerHTML = '<div class="empty-state">Select at least one visible model run.</div>';
    return;
  }
  const rows = [
    ["Passed within attempts", (model) => comparisonScore(model.passedWithinAttempts, model.totalTasks, model.passWithinAttemptsRate), true],
    ["Pass@1", (model) => comparisonScore(model.passAt1, model.totalTasks, model.passAt1Rate), true],
    ["Recovery gain", (model) => `<span class="recovery-gain${model.passedWithinAttempts > model.passAt1 ? " gained" : ""}">+${model.passedWithinAttempts - model.passAt1} tasks</span>`, true],
    ["Average task time", (model) => formatDuration(model.totalTasks ? model.totalDurationMs / model.totalTasks : 0)],
    ["Total task time", (model) => formatDuration(model.totalDurationMs)],
    ["Input tokens", (model) => formatExactNumber(model.totalTokens.input)],
    ["Output tokens", (model) => formatExactNumber(model.totalTokens.output)],
    ["Context", (model) => `${formatExactNumber(model.contextLength)} tokens`],
    ["Quant", (model) => model.quant || "not recorded"],
    ["Tag", (model) => model.tag || "not set"],
    ["Platform", (model) => `${model.platform.name} (${model.platform.id})`],
    ["Engine", (model) => `${model.engine || "not recorded"}${model.engineVersion ? ` ${model.engineVersion}` : ""}`],
    ["Compute backend", (model) => `${model.backend || "not recorded"}${model.backendVersion ? ` ${model.backendVersion}` : ""}`],
    ["Inference profile", (model) => model.inferenceProfile || "default / not recorded"],
  ];
  elements.comparison.innerHTML = `
    <table aria-label="Selected model run comparison">
      <thead><tr><th>Metric</th>${models.map((model) => `<th class="comparison-model">${escapeHtml(shortModelLabel(model))}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map(([label, getter, markup]) => `<tr><td class="metric-label">${escapeHtml(label)}</td>${models.map((model) => `<td class="metric-value">${markup ? getter(model) : escapeHtml(getter(model))}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderMatrix(tasks, models) {
  if (!models.length) {
    elements.matrix.innerHTML = '<div class="empty-state">Select at least one visible model run to build the task matrix.</div>';
    return;
  }
  if (!tasks.length) {
    elements.matrix.innerHTML = '<div class="empty-state">No tasks match the current search and filters.</div>';
    return;
  }
  elements.matrix.innerHTML = `
    <table class="task-table" aria-label="Task results by model run">
      <thead>
        <tr>
          <th>Task</th>
          ${models.map((model) => `<th class="comparison-model">${escapeHtml(shortModelLabel(model))}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${tasks.map((task) => `
          <tr>
            <td>
              <button class="task-cell-button" type="button" data-action="task-details" data-task-id="${escapeHtml(task.id)}">
                <span class="task-id">${escapeHtml(task.id)}</span>
                <span class="task-meta-line"><span>${escapeHtml(task.difficulty)}</span><span>${escapeHtml(task.category)}</span></span>
              </button>
            </td>
            ${models.map((model) => {
              const result = resultFor(model, task.id);
              if (!result) return '<td class="result-cell missing"><span class="outcome missing">Not run</span></td>';
              return `
                <td class="result-cell ${escapeHtml(result.outcomeType)}">
                  <button class="result-cell-button" type="button" data-action="result-details" data-model-id="${escapeHtml(model.id)}" data-task-id="${escapeHtml(task.id)}">
                    <span class="result-mainline">
                      <span class="outcome ${escapeHtml(result.outcomeType)}">${escapeHtml(outcomeLabel(result.outcomeType))}</span>
                      <span class="result-duration">${escapeHtml(formatDuration(result.durationMs))}</span>
                    </span>
                    <span class="result-reason">${escapeHtml(result.reason)}</span>
                  </button>
                </td>`;
            }).join("")}
          </tr>`).join("")}
      </tbody>
    </table>
  `;
}

function render() {
  const models = comparedModels();
  const tasks = filteredTasks(models.length ? models : visibleModels());
  renderModels();
  renderComparison(models);
  renderMatrix(tasks, models);
  renderSummary(tasks, models);
}

function detailList(rows) {
  return `<dl class="detail-list">${rows.map(([label, value, className = ""]) => `<dt>${escapeHtml(label)}</dt><dd class="${escapeHtml(className)}">${escapeHtml(value ?? "not recorded")}</dd>`).join("")}</dl>`;
}

function setDialogHash(params) {
  const hash = new URLSearchParams(params).toString();
  history.replaceState(null, "", `${location.pathname}${location.search}#${hash}`);
}

function showDialog(kicker, title, html) {
  elements.dialogKicker.textContent = kicker;
  elements.dialogTitle.textContent = title;
  elements.dialogContent.innerHTML = html;
  if (!elements.dialog.open) elements.dialog.showModal();
}

function openModelDetails(model, updateHash = true) {
  const average = model.totalTasks ? model.totalDurationMs / model.totalTasks : 0;
  const failed = model.results.filter((result) => !result.passed);
  const failureBreakdown = failed.reduce((counts, result) => {
    counts[result.outcomeType] = (counts[result.outcomeType] || 0) + 1;
    return counts;
  }, {});
  const breakdown = Object.entries(failureBreakdown).map(([type, count]) => `${outcomeLabel(type)}: ${count}`).join(" · ") || "No final failures";
  const html = `
    <div class="dialog-badges">
      <span class="badge">within attempts ${escapeHtml(formatPercent(model.passWithinAttemptsRate))}</span>
      <span class="badge">pass@1 ${escapeHtml(formatPercent(model.passAt1Rate))}</span>
      <span class="badge">${escapeHtml(model.engine || "engine not recorded")}</span>
      <span class="badge">${escapeHtml(model.backend || "backend not recorded")}</span>
      <span class="badge">${escapeHtml(model.platform.name)}</span>
    </div>
    <div class="dialog-grid">
      <section class="detail-panel">
        <h3>Model identity</h3>
        ${detailList([
          ["Display name", model.name, "mono"],
          ["Endpoint model ID", model.modelId, "mono"],
          ["Quant", model.quant || "not recorded", "mono"],
          ["Tag", model.tag || "not set", "mono"],
          ["Endpoint quant type", model.quantizationType || "not recorded", "mono"],
          ["Parameters", model.parameterCount ? formatExactNumber(model.parameterCount) : "not recorded", "mono"],
          ["Model size", formatBytes(model.modelSizeBytes), "mono"],
        ])}
      </section>
      <section class="detail-panel">
        <h3>Execution profile</h3>
        ${detailList([
          ["Platform", `${model.platform.name} (${model.platform.id})`],
          ["Engine", `${model.engine || "not recorded"}${model.engineVersion ? ` ${model.engineVersion}` : ""}`, "mono"],
          ["Compute backend", `${model.backend || "not recorded"}${model.backendVersion ? ` ${model.backendVersion}` : ""}`, "mono"],
          ["Endpoint owner", model.endpointOwner || "not recorded", "mono"],
          ["Inference profile", model.inferenceProfile || "default / not recorded", "mono"],
          ["Context", `${formatExactNumber(model.contextLength)} tokens`, "mono"],
          ["Agent", `${model.agent.name || "not recorded"} ${model.agent.version || ""}`.trim(), "mono"],
        ])}
      </section>
      <section class="detail-panel">
        <h3>Aggregate result</h3>
        ${detailList([
          ["Within attempts", `${model.passedWithinAttempts}/${model.totalTasks} (${formatPercent(model.passWithinAttemptsRate)})`, "mono"],
          ["Pass@1", `${model.passAt1}/${model.totalTasks} (${formatPercent(model.passAt1Rate)})`, "mono"],
          ["Average task", formatDuration(average), "mono"],
          ["Total task time", formatDuration(model.totalDurationMs), "mono"],
          ["Input / output", `${formatExactNumber(model.totalTokens.input)} / ${formatExactNumber(model.totalTokens.output)}`, "mono"],
          ["Failure mix", breakdown],
        ])}
      </section>
      <section class="detail-panel">
        <h3>Reproducibility</h3>
        ${detailList([
          ["Harbor", model.harborVersion || "not recorded", "mono"],
          ["Terminal-Bench", model.terminalBenchVersion || "not recorded", "mono"],
          ["Task revision", model.terminalBenchRevision || "not recorded", "mono"],
          ["Profile hash", model.profileHash || "not recorded", "mono"],
          ["Exported", formatDate(model.exportedAt)],
        ])}
      </section>
    </div>
    <section class="dialog-section">
      <h3>Source evidence</h3>
      <div class="evidence-links">
        <a href="${escapeHtml(model.summaryUrl)}">Summary JSON ↗</a>
        <a href="${escapeHtml(model.runMetaUrl)}">Run metadata ↗</a>
        <button class="share-button" type="button" data-action="copy-link">Copy share link</button>
      </div>
    </section>`;
  showDialog("model run profile", modelLabel(model), html);
  if (updateHash) setDialogHash({ model: model.id });
}

function attemptHtml(attempt) {
  const evidence = attempt.evidence?.length
    ? `<ul class="evidence-list">${attempt.evidence.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul>`
    : "";
  return `
    <article class="attempt-card">
      <div class="attempt-card-head">
        <span class="attempt-title">Attempt ${escapeHtml(attempt.number || "?")}</span>
        <span class="outcome ${escapeHtml(attempt.outcomeType)}">${escapeHtml(outcomeLabel(attempt.outcomeType))}</span>
      </div>
      <div class="attempt-metrics">
        <span>${escapeHtml(formatDuration(attempt.durationMs))}</span>
        <span>${escapeHtml(formatExactNumber(attempt.tokens?.input || 0))} input</span>
        <span>${escapeHtml(formatExactNumber(attempt.tokens?.output || 0))} output</span>
        <span>${escapeHtml(formatExactNumber(attempt.agentSteps || 0))} steps</span>
        <span>reward ${escapeHtml(attempt.reward ?? "N/A")}</span>
      </div>
      <p>${escapeHtml(attempt.reason)}</p>
      ${evidence}
      <div class="evidence-links">
        ${attempt.transcriptUrl ? `<a href="${escapeHtml(attempt.transcriptUrl)}">Transcript ↗</a>` : '<span class="profile-note">Transcript unavailable</span>'}
      </div>
    </article>`;
}

function openResultDetails(model, task, result, updateHash = true) {
  const html = `
    <div class="dialog-badges">
      <span class="badge ${result.passed ? "passed" : "failed"}">${result.passed ? "Passed" : "Failed"}</span>
      <span class="badge">reward ${escapeHtml(result.reward ?? "N/A")}</span>
      <span class="badge">${escapeHtml(formatDuration(result.durationMs))}</span>
      <span class="badge">${escapeHtml(task.difficulty)} · ${escapeHtml(task.category)}</span>
    </div>
    <div class="dialog-grid">
      <section class="detail-panel">
        <h3>Task</h3>
        <p>${escapeHtml(task.description)}</p>
        ${detailList([
          ["Task ID", task.id, "mono"],
          ["Difficulty", task.difficulty],
          ["Category", task.category],
          ["Expert estimate", task.expertMinutes ? `${task.expertMinutes} min` : "not recorded", "mono"],
          ["Keywords", (task.keywords || []).join(", ") || "none"],
        ])}
      </section>
      <section class="detail-panel">
        <h3>Run</h3>
        ${detailList([
          ["Model", model.name, "mono"],
          ["Quant", model.quant || "not recorded", "mono"],
          ["Tag", model.tag || "not set", "mono"],
          ["Platform", `${model.platform.name} (${model.platform.id})`],
          ["Engine", `${model.engine || "not recorded"}${model.engineVersion ? ` ${model.engineVersion}` : ""}`, "mono"],
          ["Compute backend", `${model.backend || "not recorded"}${model.backendVersion ? ` ${model.backendVersion}` : ""}`, "mono"],
          ["Context", `${formatExactNumber(model.contextLength)} tokens`, "mono"],
          ["Inference profile", model.inferenceProfile || "default / not recorded", "mono"],
        ])}
      </section>
    </div>
    <section class="outcome-panel">
      <h3>Why this result</h3>
      <p>${escapeHtml(result.reason)}</p>
    </section>
    <section class="dialog-section">
      <h3>Attempt history</h3>
      <div class="attempt-list">${result.attempts.map(attemptHtml).join("")}</div>
    </section>
    <section class="dialog-section">
      <h3>Evidence</h3>
      <div class="evidence-links">
        <a href="${escapeHtml(result.resultUrl)}">Normalized result JSON ↗</a>
        <a href="${escapeHtml(task.sourceUrl)}">Task definition ↗</a>
        <a href="${escapeHtml(model.runMetaUrl)}">Run metadata ↗</a>
        <button class="share-button" type="button" data-action="copy-link">Copy share link</button>
      </div>
    </section>
    <section class="dialog-section">
      <details class="instruction">
        <summary>View evaluated task instruction</summary>
        <div class="instruction-body">${escapeHtml(task.instruction)}</div>
      </details>
    </section>`;
  showDialog("task result", `${task.id} × ${modelLabel(model)}`, html);
  if (updateHash) setDialogHash({ result: model.id, task: task.id });
}

function openTaskDetails(task, updateHash = true) {
  const models = comparedModels().length ? comparedModels() : visibleModels();
  const rows = models.map((model) => {
    const result = resultFor(model, task.id);
    return `
      <tr>
        <td class="comparison-model">${escapeHtml(shortModelLabel(model))}</td>
        <td>${result ? `<span class="outcome ${escapeHtml(result.outcomeType)}">${escapeHtml(outcomeLabel(result.outcomeType))}</span>` : "Not run"}</td>
        <td class="metric-value">${result ? escapeHtml(formatDuration(result.durationMs)) : "—"}</td>
        <td>${result ? `<button class="details-button" type="button" data-action="result-details" data-model-id="${escapeHtml(model.id)}" data-task-id="${escapeHtml(task.id)}">Inspect →</button>` : ""}</td>
      </tr>`;
  }).join("");
  const html = `
    <div class="dialog-badges">
      <span class="badge">${escapeHtml(task.difficulty)}</span>
      <span class="badge">${escapeHtml(task.category)}</span>
      ${(task.tags || []).map((tag) => `<span class="badge">${escapeHtml(tag)}</span>`).join("")}
    </div>
    <section class="detail-panel">
      <h3>What it evaluates</h3>
      <p>${escapeHtml(task.description)}</p>
      ${detailList([
        ["Expert estimate", task.expertMinutes ? `${task.expertMinutes} min` : "not recorded", "mono"],
        ["Junior estimate", task.juniorMinutes ? `${task.juniorMinutes} min` : "not recorded", "mono"],
        ["Keywords", (task.keywords || []).join(", ") || "none"],
      ])}
    </section>
    <section class="dialog-section">
      <h3>Results across selected runs</h3>
      <div class="comparison-wrap"><table><thead><tr><th>Model run</th><th>Outcome</th><th>Time</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
    </section>
    <section class="dialog-section">
      <div class="evidence-links">
        <a href="${escapeHtml(task.sourceUrl)}">Task definition ↗</a>
        <a href="${escapeHtml(task.instructionUrl)}">Instruction source ↗</a>
        <button class="share-button" type="button" data-action="copy-link">Copy share link</button>
      </div>
    </section>
    <section class="dialog-section">
      <details class="instruction"><summary>View evaluated task instruction</summary><div class="instruction-body">${escapeHtml(task.instruction)}</div></details>
    </section>`;
  showDialog("task profile", task.id, html);
  if (updateHash) setDialogHash({ task: task.id });
}

function findModel(id) {
  return state.data.models.find((model) => model.id === id);
}

function findTask(id) {
  return state.data.tasks.find((task) => task.id === id);
}

function handleAction(target) {
  const actionTarget = target.closest("[data-action]");
  if (!actionTarget) return;
  const action = actionTarget.dataset.action;
  const card = actionTarget.closest("[data-model-id]");
  const modelId = actionTarget.dataset.modelId || card?.dataset.modelId;
  const taskId = actionTarget.dataset.taskId;
  const model = modelId ? findModel(modelId) : null;
  const task = taskId ? findTask(taskId) : null;

  if (action === "toggle-model" && model) {
    if (state.selectedModels.has(model.id)) state.selectedModels.delete(model.id);
    else if (state.selectedModels.size < 4) state.selectedModels.add(model.id);
    else {
      elements.summary.innerHTML = "<strong>Comparison limit:</strong> remove a selected model before adding another.";
      return;
    }
    render();
  } else if (action === "model-details" && model) {
    openModelDetails(model);
  } else if (action === "result-details" && model && task) {
    const result = resultFor(model, task.id);
    if (result) openResultDetails(model, task, result);
  } else if (action === "task-details" && task) {
    openTaskDetails(task);
  } else if (action === "copy-link") {
    navigator.clipboard?.writeText(location.href).then(() => {
      actionTarget.textContent = "Copied";
      setTimeout(() => { actionTarget.textContent = "Copy share link"; }, 1200);
    });
  }
}

function closeDialog() {
  if (elements.dialog.open) elements.dialog.close();
  history.replaceState(null, "", `${location.pathname}${location.search}`);
}

function openFromHash() {
  if (!state.data || !location.hash) return;
  const params = new URLSearchParams(location.hash.slice(1));
  const modelId = params.get("result") || params.get("model");
  const taskId = params.get("task");
  const model = modelId ? findModel(modelId) : null;
  const task = taskId ? findTask(taskId) : null;
  if (params.has("result") && model && task) {
    const result = resultFor(model, task.id);
    if (result) openResultDetails(model, task, result, false);
  } else if (params.has("model") && model) openModelDetails(model, false);
  else if (task) openTaskDetails(task, false);
}

function bindEvents() {
  const inputMap = [
    [elements.search, "search", "input"],
    [elements.model, "model", "change"],
    [elements.quant, "quant", "change"],
    [elements.profile, "profile", "change"],
    [elements.tag, "tag", "change"],
    [elements.platform, "platform", "change"],
    [elements.engine, "engine", "change"],
    [elements.backend, "backend", "change"],
    [elements.status, "status", "change"],
    [elements.difficulty, "difficulty", "change"],
    [elements.sort, "sort", "change"],
  ];
  for (const [element, key, event] of inputMap) {
    element.addEventListener(event, () => {
      state.filters[key] = element.value;
      render();
    });
  }

  document.querySelector("#reset-filters").addEventListener("click", () => {
    Object.assign(state.filters, { search: "", model: "", quant: "", profile: "", tag: "", platform: "", engine: "", backend: "", status: "", difficulty: "", sort: "task" });
    for (const [element, key] of inputMap) element.value = state.filters[key];
    render();
  });
  document.querySelector("#select-all-models").addEventListener("click", () => {
    state.selectedModels.clear();
    visibleModels().slice(0, 4).forEach((model) => state.selectedModels.add(model.id));
    render();
  });
  document.querySelector("#clear-models").addEventListener("click", () => {
    state.selectedModels.clear();
    render();
  });
  document.querySelector("#close-dialog").addEventListener("click", closeDialog);
  elements.dialog.addEventListener("close", () => {
    if (location.hash) history.replaceState(null, "", `${location.pathname}${location.search}`);
  });
  elements.modelGrid.addEventListener("click", (event) => handleAction(event.target));
  elements.matrix.addEventListener("click", (event) => handleAction(event.target));
  elements.dialogContent.addEventListener("click", (event) => handleAction(event.target));
  window.addEventListener("hashchange", openFromHash);
}

async function initialize() {
  try {
    const response = await fetch("data.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status} while loading data.json`);
    state.data = await response.json();
    state.data.models.slice(0, 4).forEach((model) => state.selectedModels.add(model.id));
    renderIntro();
    renderFilters();
    bindEvents();
    render();
    elements.loading.hidden = true;
    elements.content.hidden = false;
    openFromHash();
  } catch (error) {
    elements.loading.hidden = true;
    elements.error.hidden = false;
    elements.error.textContent = `Could not load benchmark data: ${error.message}. Serve docs/ through a local HTTP server rather than opening index.html directly.`;
  }
}

initialize();
