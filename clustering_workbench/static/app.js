const state = {
  report: null,
  config: null,
  preprocessMethod: "standard",
  selectedCandidate: null,
  activeTab: "leaderboard",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const methodClass = (method) => method === "DBSCAN" ? "dbscan" : method === "GaussianMixture" ? "gmm" : "kmeans";
const methodShort = (method) => method === "GaussianMixture" ? "GMM" : method;
const fmt = (value, digits = 4) => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
const intFmt = (value) => value === null || value === undefined ? "—" : Number(value).toLocaleString("en-US");
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const truncate = (value, n = 22) => {
  const text = String(value ?? "—");
  return text.length > n ? `${text.slice(0, n)}…` : text;
};
const paramChips = (params) => Object.entries(params || {}).map(([key, value]) => `<span class="param-chip">${esc(key)}=${esc(value)}</span>`).join("");
const statusChip = (status) => `<span class="status ${esc(status)}">${esc(status)}</span>`;
const flagsHtml = (flags = []) => flags.map((flag) => `<span class="flag ${flag === "ALL_NOISE" ? "noise" : ""}">${esc(flag)}</span>`).join("");

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.classList.add("show");
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => node.classList.remove("show"), 3600);
}

function setRunState(kind, text) {
  const wrapper = $("#runState");
  wrapper.className = `run-state ${kind}`;
  $("#runStateText").textContent = text;
  [$("#runButton"), $("#sidebarRunButton")].forEach((button) => { button.disabled = kind === "running"; });
}

function readControls() {
  return {
    preprocess_method: state.preprocessMethod,
    sample_scale: Number($("#sampleScale").value || 1),
    max_workers: Number($("#maxWorkers").value || 4),
    global_budget: Number($("#globalBudget").value || 150),
    candidate_quota: Number($("#candidateQuota").value || 18),
    cache_blocks: Number($("#cacheBlocks").value || 8),
    snapshot_version: Number($("#snapshotVersion").value || 1),
    injections_enabled: $("#injectionsEnabled").checked,
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
  return payload;
}

async function bootstrap() {
  try {
    state.config = await fetchJson("/api/config");
    const select = $("#sampleScale");
    const baseRows = Number(state.config.base_rows || 240);
    select.innerHTML = state.config.sample_scale_options.map((value) => `<option value="${value}" ${value === 1 ? "selected" : ""}>${value.toFixed(1)}× · request ${Math.round(baseRows * value)} rows</option>`).join("");
    try {
      const response = await fetchJson("/api/report");
      if (response.available) {
        state.report = response.report;
        hydrateControlsFromReport();
        render();
      } else {
        await runExperiment();
      }
    } catch (error) {
      if (String(error.message).includes("No report")) await runExperiment();
      else throw error;
    }
  } catch (error) {
    setRunState("error", "Server unavailable");
    toast(error.message, true);
  }
}

function hydrateControlsFromReport() {
  if (!state.report) return;
  const pre = state.report.preprocess?.method;
  if (pre === "quantile" || pre === "standard") {
    state.preprocessMethod = pre;
    $$(".segment").forEach((button) => button.classList.toggle("active", button.dataset.value === pre));
  }
  const schedule = state.report.scheduler || {};
  if (schedule.max_workers) $("#maxWorkers").value = String(schedule.max_workers);
  if (schedule.global_iteration_budget) $("#globalBudget").value = schedule.global_iteration_budget;
  if (schedule.per_candidate_iteration_quota) $("#candidateQuota").value = schedule.per_candidate_iteration_quota;
  if (schedule.cache?.max_blocks) $("#cacheBlocks").value = schedule.cache.max_blocks;
  const version = state.report.snapshots?.global_version;
  if (version) $("#snapshotVersion").value = version;
  const scale = state.report.dataset?.sample_scale;
  if (scale) {
    const option = Array.from($("#sampleScale").options).find((item) => Number(item.value) === Number(scale));
    if (option) $("#sampleScale").value = option.value;
  }
}

async function runExperiment() {
  setRunState("running", "Computing · deterministic pipeline");
  toast("Running clustering, evaluation and snapshot commit…");
  try {
    const payload = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readControls()),
    });
    state.report = payload.report;
    state.selectedCandidate = null;
    render();
    setRunState("done", "Run complete · fingerprinted");
    toast(`Run complete · ${state.report.ranking?.length || 0} ranked candidates`);
  } catch (error) {
    setRunState("error", "Run failed");
    toast(error.message, true);
  }
}

async function verifyParity() {
  setRunState("running", "A/B parity · serial vs thread pool");
  toast("Executing serial and parallel passes in isolated workspaces…");
  $("#verifyButton").disabled = true;
  try {
    const result = await fetchJson("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readControls()),
    });
    state.verifyResult = result;
    if (state.activeTab === "scheduling") renderTab();
    setRunState("done", result.digest_equal ? "Parity verified" : "Parity mismatch");
    toast(result.digest_equal ? `A/B parity verified · ${result.serial_digest}` : "Parity mismatch detected", !result.digest_equal);
  } catch (error) {
    setRunState("error", "Parity check failed");
    toast(error.message, true);
  } finally {
    $("#verifyButton").disabled = false;
  }
}

function render() {
  const report = state.report;
  if (!report) return;
  const run = report.run || {};
  const dataset = report.dataset || {};
  const preprocess = report.preprocess || {};
  const scheduler = report.scheduler || {};
  const ranking = report.ranking || [];
  const candidates = report.candidates || [];
  const injections = report.injections || {};
  const status = run.status || "unknown";
  const statusNode = $("#runStatus");
  statusNode.textContent = status.replaceAll("_", " ").toUpperCase();
  statusNode.className = `status-chip ${status.includes("failed") ? "error" : status.includes("completed") ? "ok" : "warn"}`;
  $("#runId").textContent = run.run_id || "—";
  $("#runTime").textContent = `${run.duration_ms ?? "—"} ms · ${run.started_at ? new Date(run.started_at).toLocaleString() : ""}`;
  $("#digestPill").textContent = `digest ${truncate(report.determinism_digest || "—", 22)}`;
  $("#digestPill").title = report.determinism_digest || "";

  $("#kpiRows").textContent = intFmt(dataset.rows_generated);
  $("#kpiRowsSub").textContent = `${dataset.column_types ? Object.keys(dataset.column_types).length : 0} generated columns · seed ${dataset.seed ?? "—"}`;
  $("#kpiRanked").textContent = intFmt(ranking.length);
  $("#kpiRankedSub").textContent = `${candidates.filter((c) => c.status === "invalid").length} invalid · ${candidates.filter((c) => c.status === "budget_aborted").length} budget-aborted`;
  $("#kpiBudget").textContent = `${intFmt(scheduler.budget_reserved)} / ${intFmt(scheduler.global_iteration_budget)}`;
  $("#kpiBudgetSub").textContent = `${intFmt(scheduler.iterations_executed_total)} iterations actually executed`;
  $("#kpiInjection").textContent = `${injections.hit ?? 0} / ${injections.total ?? 0}`;
  $("#kpiInjectionSub").textContent = `${injections.missed ?? 0} missed · isolated fault paths`;

  $("#sideDataset").textContent = dataset.name || "—";
  $("#sideRows").textContent = intFmt(dataset.rows_generated);
  $("#sideFeatures").textContent = preprocess.shape ? `${preprocess.shape[0]} × ${preprocess.shape[1]}` : "—";
  $("#sideSource").textContent = preprocess.source || "—";
  $("#fpData").textContent = truncate(dataset.descriptor_fingerprint || "—", 29);
  $("#fpPre").textContent = truncate(preprocess.artifact_fingerprint || "—", 29);
  $("#fpDiag").textContent = truncate(report.diagnostics?.artifact_fingerprint || "—", 29);
  $("#preprocessMethod").textContent = preprocess.method ? `${preprocess.method} · ${preprocess.shape?.join("×")}` : "—";
  renderSnapshotNote();
  renderRankChart();
  renderPreprocessSummary();
  renderTab();
}

function renderSnapshotNote() {
  const container = $("#snapshotNote");
  const snap = state.report?.snapshots || {};
  if (!snap.enabled) {
    container.innerHTML = `<div class="section-kicker">Snapshot state</div><p>Snapshots disabled for this run.</p>`;
    return;
  }
  const ok = snap.valid;
  const manifest = snap.manifest || {};
  container.innerHTML = `<div class="section-kicker">Snapshot state</div>
    <p><span class="${ok ? "ok" : "warn"}">${ok ? "● RESUME-COMPATIBLE" : "● VERSION INVALID"}</span><br>
    ${esc(snap.reason || "—")}<br>
    ${manifest.completed_candidates ? `${manifest.completed_candidates.length} completed candidates · ${manifest.budget_consumed ?? 0} consumed iterations` : "No manifest details."}</p>`;
}

function renderRankChart() {
  const container = $("#rankChart");
  const ranking = state.report?.ranking || [];
  if (!ranking.length) {
    container.innerHTML = `<div class="empty-state">No rankable candidate in this run.</div>`;
    return;
  }
  const primary = state.report?.config?.evaluation?.primary_metric || "silhouette";
  $("#primaryMetricLabel").textContent = `primary · ${primary}`;
  const scores = ranking.map((row) => Number(row.metrics?.silhouette));
  const finite = scores.filter((value) => Number.isFinite(value));
  const max = finite.length ? Math.max(...finite, 0.0001) : 1;
  container.innerHTML = ranking.map((row) => {
    const score = row.metrics?.silhouette;
    const numeric = Number(score);
    const valid = Number.isFinite(numeric);
    const width = valid ? Math.max(2, Math.min(100, (Math.max(numeric, 0) / max) * 100)) : 0;
    return `<div class="rank-row">
      <span class="rank-index">#${row.rank}</span>
      <div class="rank-main">
        <div class="rank-name-row">
          <span class="rank-name" title="${esc(row.stable_key)}"><span class="method-dot ${methodClass(row.method)}"></span> ${esc(methodShort(row.method))} · ${paramChips(row.params)}</span>
          <span class="rank-flags">${flagsHtml(row.diagnostic_flags)}</span>
        </div>
        <div class="rank-track"><div class="rank-fill ${valid ? methodClass(row.method) : "none"}" style="width:${width}%"></div></div>
      </div>
      <span class="rank-score">${fmt(score, 4)}</span>
    </div>`;
  }).join("");
}

function renderPreprocessSummary() {
  const pre = state.report?.preprocess || {};
  const container = $("#preprocessSummary");
  if (!pre.method) { container.innerHTML = `<div class="empty-state">No preprocess artifact.</div>`; return; }
  const dropped = pre.dropped_columns || [];
  const stats = pre.statistics || {};
  const featureCount = pre.feature_names?.length || 0;
  container.innerHTML = `<div class="stat-stack">
    <div class="stat-line"><span>Training / total rows</span><strong>${pre.train_row_count} / ${pre.shape?.[0]}</strong></div>
    <div class="stat-line"><span>Kept features</span><strong>${featureCount}</strong></div>
    <div class="stat-line"><span>Content fingerprint</span><strong title="${esc(pre.content_fingerprint)}">${truncate(pre.content_fingerprint, 22)}</strong></div>
    <div class="stat-line"><span>Accumulation order</span><strong>row asc</strong></div>
  </div>${dropped.length ? `<div class="dropped-box"><strong>${dropped.length} redundant column${dropped.length > 1 ? "s" : ""} dropped</strong><br>${dropped.map((d) => `${esc(d.column)} ≡ ${esc(d.duplicate_of)}`).join("<br>")}</div>` : ""}
  <div class="dropped-box" style="background:#f4f7fb;border-color:#e0e7ef;color:#536277">${Object.keys(stats).filter((name) => name !== "_pipeline").slice(0, 4).map((name) => `${esc(name)}: fill ${fmt(stats[name]?.fill_value, 3)}`).join("<br>")}</div>`;
}

function renderTab() {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === state.activeTab));
  const panel = $("#tabPanel");
  const funcs = {
    leaderboard: renderLeaderboard,
    assignments: renderAssignments,
    scheduling: renderScheduling,
    reuse: renderReuse,
    snapshots: renderSnapshots,
    injections: renderInjections,
    failures: renderFailures,
    raw: renderRaw,
  };
  panel.innerHTML = funcs[state.activeTab]?.() || "";
  bindTabEvents();
}

function renderLeaderboard() {
  const rows = state.report?.candidates || [];
  if (!rows.length) return `<div class="empty-table">No candidates.</div>`;
  return `<div class="section-heading"><div><h3>Complete candidate ledger</h3><p>Every candidate is retained, including empty-cluster, all-noise, invalid and budget-aborted states. Ranking ties resolve by method then canonical hyperparameter dictionary.</p></div><div class="toolbar"><span class="panel-meta">${rows.length} candidates</span></div></div>
  <div class="table-wrap"><table><thead><tr><th>Rank</th><th>Candidate</th><th>Method</th><th>Parameters</th><th>Silhouette ↑</th><th>Calinski–Harabasz ↑</th><th>Density LL ↑</th><th>Status</th><th>Diagnostics</th><th>Source</th></tr></thead><tbody>
  ${rows.map((row) => `<tr>
    <td class="mono">${row.rank ? `#${row.rank}` : "—"}</td>
    <td class="mono" title="${esc(row.stable_key)}">${esc(row.display_id)}</td>
    <td><span class="method-badge"><span class="method-dot ${methodClass(row.method)}"></span>${esc(methodShort(row.method))}</span></td>
    <td>${paramChips(row.params)}</td>
    <td class="mono">${fmt(row.metrics?.silhouette, 4)}</td>
    <td class="mono">${fmt(row.metrics?.calinski_harabasz, 3)}</td>
    <td class="mono">${fmt(row.metrics?.density_log_likelihood, 4)}</td>
    <td>${statusChip(row.status)}</td>
    <td>${flagsHtml(row.diagnostic_flags)}</td>
    <td><span class="muted mono">${esc(row.source || row.evaluation_source || "—")}</span></td>
  </tr>`).join("")}</tbody></table></div>`;
}

function candidateOptions() {
  const candidates = (state.report?.candidates || []).filter((row) => row.labels);
  if (!state.selectedCandidate || !candidates.some((row) => row.stable_key === state.selectedCandidate)) {
    state.selectedCandidate = candidates[0]?.stable_key || null;
  }
  return candidates.map((row) => `<option value="${esc(row.stable_key)}" ${row.stable_key === state.selectedCandidate ? "selected" : ""}>${esc(row.display_id)} · ${esc(row.method)}${row.rank ? ` · rank ${row.rank}` : ""}</option>`).join("");
}

function renderAssignments() {
  const candidates = (state.report?.candidates || []).filter((row) => row.labels);
  if (!candidates.length) return `<div class="empty-table">No hard assignments available.</div>`;
  const selected = candidates.find((row) => row.stable_key === state.selectedCandidate) || candidates[0];
  state.selectedCandidate = selected.stable_key;
  const palette = ["#2457e6", "#0e9f8f", "#c77910", "#7b4bd8", "#d9586b", "#4a8fb8", "#9b7b36", "#596579"];
  const labels = selected.labels || [];
  const cells = labels.map((label, index) => {
    const color = label < 0 ? "#c5ced9" : palette[Math.abs(label) % palette.length];
    return `<span class="assignment-cell" style="background:${color}" title="sample ${index} · cluster ${label}"></span>`;
  }).join("");
  const unique = [...new Set(labels)].sort((a, b) => a - b);
  const legend = unique.map((label) => `<span class="legend-item"><span class="legend-swatch" style="background:${label < 0 ? "#c5ced9" : palette[Math.abs(label) % palette.length]}"></span>${label < 0 ? "noise" : `cluster ${label}`}</span>`).join("");
  const sizes = selected.diagnostics?.cluster_sizes || [];
  return `<div class="section-heading"><div><h3>Hard sample assignments</h3><p>One cell per generated sample in fixed row order. Noise stays visible byte-for-byte; the same assignment array participates in the determinism digest.</p></div><div class="toolbar"><select id="assignmentCandidate">${candidateOptions()}</select></div></div>
  <div class="assignment-ribbon">${cells}</div><div class="legend">${legend}</div>
  <div class="sub-grid three" style="margin-top:16px">
    <div class="sub-card"><h4>Assignment structure</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${intFmt(labels.length)}</strong><span>samples</span></div><div class="mini-kpi"><strong>${unique.filter((x) => x >= 0).length}</strong><span>clusters</span></div><div class="mini-kpi"><strong>${unique.filter((x) => x < 0).length ? labels.filter((x) => x < 0).length : 0}</strong><span>noise</span></div></div></div>
    <div class="sub-card"><h4>Cluster sizes</h4><div class="chip-list">${sizes.length ? sizes.map((size, index) => `<span class="chip">C${index}: ${size}</span>`).join("") : `<span class="muted">not reported</span>`}</div></div>
    <div class="sub-card"><h4>Labels fingerprint</h4><code class="mono" style="font-size:8.5px;word-break:break-all">${esc(selected.label_fingerprint || "—")}</code></div>
  </div>
  <div class="sub-grid" style="margin-top:14px">
    <div class="sub-card"><h4>Training trajectory · ${esc(selected.method)}</h4>${trajectorySvg(selected)}</div>
    <div class="sub-card"><h4>Boundary &amp; density diagnostics</h4><div class="stat-stack">
      <div class="stat-line"><span>Boundary check</span><strong>${selected.boundary_check ? `${esc(selected.boundary_check.decision)} @ ${fmt(selected.boundary_check.threshold, 4)}` : "not exercised"}</strong></div>
      <div class="stat-line"><span>Density log-likelihood</span><strong>${fmt(selected.metrics?.density_log_likelihood, 4)}</strong></div>
      <div class="stat-line"><span>Budget grant / executed</span><strong>${selected.budget?.grant ?? "—"} / ${selected.budget?.iterations_executed ?? "—"}</strong></div>
      <div class="stat-line"><span>Status</span><strong>${esc(selected.status)}</strong></div>
    </div></div>
  </div>`;
}

function trajectorySvg(candidate) {
  const values = candidate.trajectory || [];
  if (!values.length) return `<div class="trajectory-box"><div class="empty-state">No iterative trajectory (single-pass method or aborted candidate).</div></div>`;
  const width = 720, height = 180, padding = 18;
  const finite = values.filter(Number.isFinite);
  const min = Math.min(...finite), max = Math.max(...finite);
  const span = max - min || 1;
  const points = values.map((value, index) => {
    const x = padding + (values.length === 1 ? 0 : (index / (values.length - 1)) * (width - padding * 2));
    const y = height - padding - ((value - min) / span) * (height - padding * 2);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return `<div class="trajectory-box"><svg class="trajectory-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    ${[0, .25, .5, .75, 1].map((f) => `<line x1="${padding}" x2="${width-padding}" y1="${padding + f*(height-padding*2)}" y2="${padding + f*(height-padding*2)}" stroke="#e6ebf2" stroke-width="1"/>`).join("")}
    <polyline points="${points}" fill="none" stroke="#2457e6" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    ${values.map((value, index) => { const pair = points.split(" ")[index].split(","); return `<circle cx="${pair[0]}" cy="${pair[1]}" r="2.6" fill="#fff" stroke="#2457e6" stroke-width="1.6"/>`; }).join("")}
  </svg></div><div class="legend"><span class="legend-item">iterations ${values.length}</span><span class="legend-item">min ${fmt(min, 4)}</span><span class="legend-item">max ${fmt(max, 4)}</span></div>`;
}

function renderScheduling() {
  const schedule = state.report?.scheduler || {};
  const log = schedule.budget_log || [];
  const maxGrant = Math.max(1, ...log.map((row) => row.grant || 0));
  const verify = state.verifyResult;
  return `<div class="section-heading"><div><h3>Deterministic budget scheduler</h3><p>Grants are committed in candidate stable-key order before thread execution. Actual completion order may differ; ranking, labels, metric order, budget deduction sequence and failures do not.</p></div><div class="toolbar"><span class="panel-meta">${esc(schedule.mode)} · ${schedule.max_workers} workers</span></div></div>
  <div class="sub-grid three">
    <div class="sub-card"><h4>Global budget</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${intFmt(schedule.global_iteration_budget)}</strong><span>configured</span></div><div class="mini-kpi"><strong>${intFmt(schedule.budget_reserved)}</strong><span>reserved</span></div><div class="mini-kpi"><strong>${intFmt(schedule.iterations_executed_total)}</strong><span>executed</span></div></div></div>
    <div class="sub-card"><h4>Candidate quota</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${intFmt(schedule.per_candidate_iteration_quota)}</strong><span>max grant</span></div><div class="mini-kpi"><strong>${intFmt(schedule.budget_remaining)}</strong><span>unused global</span></div></div></div>
    <div class="sub-card"><h4>Resident block cache</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${intFmt(schedule.cache?.hits)}</strong><span>hits</span></div><div class="mini-kpi"><strong>${intFmt(schedule.cache?.evictions)}</strong><span>evictions</span></div><div class="mini-kpi"><strong>${intFmt(schedule.cache?.recomputes)}</strong><span>recomputes</span></div></div></div>
  </div>
  <div class="sub-grid">
    <div class="sub-card"><h4>Budget waterfall</h4><div class="budget-waterfall">${log.map((row) => `<div class="budget-row"><span class="budget-label" title="${esc(row.stable_key)}">${esc(truncate(row.stable_key, 23))}</span><div class="budget-track"><div class="budget-fill ${row.limit_reason === 'candidate_quota' ? 'quota' : row.limit_reason === 'global_budget' ? 'global' : ''}" style="width:${row.grant ? Math.max(3, row.grant/maxGrant*100) : 0}%"></div></div><span class="budget-meta">${row.grant}/${row.requested}</span></div>`).join("")}</div></div>
    <div class="sub-card"><h4>Completion order (non-canonical)</h4><div class="chip-list">${(schedule.completion_order || []).length ? schedule.completion_order.map((key) => `<span class="chip">${esc(truncate(key, 27))}</span>`).join("") : `<span class="muted">all candidates reused from cache</span>`}</div>
      <h4 style="margin-top:16px">Serial / parallel parity</h4>${verify ? `<div class="notice ${verify.digest_equal ? "ok" : ""}">${verify.digest_equal ? "✓ Verified" : "✕ Mismatch"} · ranking ${verify.ranking_equal ? "equal" : "diff"} · labels ${verify.labels_equal ? "equal" : "diff"} · budget ${verify.budget_equal ? "equal" : "diff"} · failures ${verify.failures_equal ? "equal" : "diff"}<br><code>${esc(verify.serial_digest)}</code></div>` : `<div class="notice">Run the A/B parity check to compare serial and thread-pool passes in isolated workspaces.</div>`}</div>
  </div>
  <div class="sub-card" style="margin-top:14px"><h4>Budget deduction ledger</h4><div class="table-wrap"><table><thead><tr><th>Seq</th><th>Stable key</th><th>Before</th><th>Grant</th><th>After</th><th>Requested</th><th>Limit reason</th></tr></thead><tbody>${log.map((row) => `<tr><td class="mono">${row.sequence}</td><td class="mono">${esc(row.stable_key)}</td><td class="mono">${row.budget_before}</td><td class="mono">${row.grant}</td><td class="mono">${row.budget_after}</td><td class="mono">${row.requested}</td><td class="mono">${esc(row.limit_reason)}</td></tr>`).join("")}</tbody></table></div></div>`;
}

function renderReuse() {
  const reuse = state.report?.reuse || {};
  const entries = reuse.entries || [];
  return `<div class="section-heading"><div><h3>Content-fingerprint reuse ledger</h3><p>Every stage records its identity fingerprint, artifact fingerprint and audit source. Identity mismatch triggers rebuild; identity match must be loaded verbatim.</p></div><div class="toolbar"><span class="panel-meta">${esc(Object.entries(reuse.status_counts || {}).map(([k,v]) => `${k} ${v}`).join(" · "))}</span></div></div>
  <div class="sub-grid three"><div class="sub-card"><h4>Reused</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${reuse.reused_total ?? 0}</strong><span>ledger entries</span></div></div></div><div class="sub-card"><h4>Rebuilt</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${reuse.rebuilt_total ?? 0}</strong><span>ledger entries</span></div></div></div><div class="sub-card"><h4>Compute calls</h4><div class="mini-kpis"><div class="mini-kpi"><strong>${reuse.counters?.preprocess_compute_calls ?? 0}</strong><span>preprocess</span></div><div class="mini-kpi"><strong>${reuse.counters?.cluster_compute_calls ?? 0}</strong><span>cluster</span></div><div class="mini-kpi"><strong>${reuse.counters?.evaluation_compute_calls ?? 0}</strong><span>evaluation</span></div></div></div></div>
  <div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>Phase</th><th>Logical key</th><th>Status</th><th>Source</th><th>Identity fingerprint</th><th>Artifact fingerprint</th><th>Change</th><th>Rebuilt despite match</th></tr></thead><tbody>${entries.map((entry) => `<tr><td>${esc(entry.phase)}</td><td class="mono">${esc(truncate(entry.logical_key, 32))}</td><td>${statusChip(entry.status)}</td><td class="mono">${esc(entry.source)}</td><td class="mono" title="${esc(entry.identity_fingerprint)}">${esc(truncate(entry.identity_fingerprint, 23))}</td><td class="mono" title="${esc(entry.artifact_fingerprint)}">${esc(truncate(entry.artifact_fingerprint, 23))}</td><td class="mono">${entry.identity_changed ? "yes" : "no"}</td><td class="mono">${entry.recomputed_despite_same_identity ? "yes" : "no"}</td></tr>`).join("")}</tbody></table></div>
  ${(reuse.retired || []).length ? `<div class="sub-card" style="margin-top:14px"><h4>Retired entries</h4><div class="chip-list">${reuse.retired.map((item) => `<span class="chip">${esc(item.logical_key)}</span>`).join("")}</div></div>` : ""}`;
}

function renderSnapshots() {
  const snap = state.report?.snapshots || {};
  const manifest = snap.manifest || {};
  const stages = manifest.stage_versions || {};
  return `<div class="section-heading"><div><h3>Versioned stage snapshots</h3><p>Stage payloads are written to temporary directories, promoted to immutable versions, then published by one atomic pointer replacement. Resume is fail-closed on version, config hash or injection hash mismatch.</p></div><span class="${snap.valid ? 'status completed' : 'status invalid'}">${snap.valid ? "resume-compatible" : "not resumed"}</span></div>
  <div class="snapshot-grid"><div class="snapshot-cell"><span>Global version</span><code>${esc(snap.global_version ?? "—")}</code></div><div class="snapshot-cell"><span>Config hash</span><code>${esc(snap.config_hash || "—")}</code></div><div class="snapshot-cell"><span>Injection hash</span><code>${esc(snap.injection_hash || "—")}</code></div><div class="snapshot-cell"><span>Reason</span><code>${esc(snap.reason || "—")}</code></div><div class="snapshot-cell"><span>Budget consumed</span><code>${esc(manifest.budget_consumed ?? "—")}</code></div><div class="snapshot-cell"><span>Budget reserved</span><code>${esc(manifest.budget_reserved ?? "—")}</code></div></div>
  <div class="sub-card" style="margin-top:14px"><h4>Published stages</h4><div class="table-wrap"><table><thead><tr><th>Stage</th><th>Version</th><th>Fingerprint</th><th>Committed files</th></tr></thead><tbody>${Object.entries(stages).map(([stage, info]) => `<tr><td>${esc(stage)}</td><td class="mono">${esc(info.version)}</td><td class="mono">${esc(truncate(info.fingerprint, 31))}</td><td class="mono">${esc((info.committed_files || []).join(", "))}</td></tr>`).join("")}</tbody></table></div></div>
  ${snap.invalidated_manifest ? `<div class="notice" style="margin-top:14px">An incompatible prior manifest was archived, not merged. Reason: <code>${esc(snap.reason)}</code>.<br>Previous config hash: <code>${esc(snap.invalidated_manifest.config_hash)}</code></div>` : ""}`;
}

function renderInjections() {
  const injections = state.report?.injections || {};
  const items = injections.items || [];
  if (!items.length) return `<div class="empty-table">Injections disabled for this run.</div>`;
  return `<div class="section-heading"><div><h3>Fault injection isolation</h3><p>Each injected fault maps to a stable business code. Injection IDs are evidence anchors; isolation keeps unaffected candidates and columns running.</p></div><span class="panel-meta">${injections.hit}/${injections.total} hit</span></div>
  <div class="injection-grid">${items.map((item) => `<article class="injection-card"><div class="injection-top"><span class="injection-id">${esc(item.id)}</span><span class="hit-pill ${item.hit ? "" : "missed"}">${item.hit ? "HIT" : "MISSED"}</span></div><div class="chip-list"><span class="chip">${esc(item.type)}</span><span class="chip">${esc(item.stage)}</span><span class="chip">${esc(item.business_code)}</span></div><p>${item.isolated ? "Fault path isolated; run continued." : "Fault path not isolated."}</p><div class="evidence">${esc(JSON.stringify(item.evidence, null, 2))}</div></article>`).join("")}</div>`;
}

function renderFailures() {
  const failures = state.report?.failures || [];
  const catalog = state.report?.business_codes || [];
  return `<div class="section-heading"><div><h3>Stable failure inventory</h3><p>Sorted by stage, item, code and message. Warnings remain visible and do not erase the candidate or downstream artifacts.</p></div><span class="panel-meta">${failures.length} entries</span></div>
  ${failures.length ? `<div class="table-wrap"><table><thead><tr><th>Severity</th><th>Code</th><th>Stage</th><th>Item</th><th>Message</th><th>Details</th></tr></thead><tbody>${failures.map((failure) => `<tr><td><span class="severity ${esc(failure.severity)}">${esc(failure.severity)}</span></td><td class="failure-code">${esc(failure.code)}</td><td class="mono">${esc(failure.stage)}</td><td class="mono">${esc(truncate(failure.item, 28))}</td><td>${esc(failure.message)}</td><td><code class="mono">${esc(truncate(JSON.stringify(failure.details), 74))}</code></td></tr>`).join("")}</tbody></table></div>` : `<div class="notice ok">No failures or warnings.</div>`}
  <details class="sub-card" style="margin-top:14px"><summary style="cursor:pointer;font-weight:700;font-size:11.5px">Stable business code catalog · ${catalog.length} codes</summary><div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Code</th><th>Kind</th><th>Severity</th><th>Description</th></tr></thead><tbody>${catalog.map((item) => `<tr><td class="failure-code">${esc(item.code)}</td><td class="mono">${esc(item.kind)}</td><td><span class="severity ${esc(item.severity)}">${esc(item.severity)}</span></td><td>${esc(item.description)}</td></tr>`).join("")}</tbody></table></div></details>`;
}

function renderRaw() {
  const report = state.report || {};
  return `<div class="section-heading"><div><h3>Raw machine-readable report</h3><p>The same report is persisted to <code>outputs/latest_report.json</code> with a deterministic core in <code>outputs/latest_core.json</code>.</p></div><div class="toolbar"><button class="button button-ghost" id="downloadReport">Download JSON</button></div></div><pre class="json-view">${esc(JSON.stringify(report, null, 2))}</pre>`;
}

function bindTabEvents() {
  const assignment = $("#assignmentCandidate");
  if (assignment) assignment.addEventListener("change", (event) => { state.selectedCandidate = event.target.value; renderTab(); });
  const download = $("#downloadReport");
  if (download) download.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state.report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "clustering-diagnostics-report.json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

$$("#tabs .tab").forEach((tab) => tab.addEventListener("click", () => { state.activeTab = tab.dataset.tab; renderTab(); }));
$$("#preprocessSegmented .segment").forEach((button) => button.addEventListener("click", () => {
  state.preprocessMethod = button.dataset.value;
  $$("#preprocessSegmented .segment").forEach((item) => item.classList.toggle("active", item === button));
}));
$("#runButton").addEventListener("click", runExperiment);
$("#sidebarRunButton").addEventListener("click", runExperiment);
$("#verifyButton").addEventListener("click", verifyParity);
bootstrap();
