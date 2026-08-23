// GENERIC_PLATFORM_PROVIDER_C_UI_V1
const state = {
  runId: null,
  stageStatus: {},
  lastPayload: {},
  images: {},
};

const stages = [
  {
    id: "stage0",
    label: "Stage 0",
    title: "Intake & condition profiling",
    description: "Upload, preserve raw evidence, and assess page condition.",
    actionLabel: "Run profiling",
    endpoint: runId => `/analyze/${runId}`,
  },
  {
    id: "stage1",
    label: "Stage 1",
    title: "Restoration",
    description: "Degradation-aware enhancement while preserving evidence.",
    actionLabel: "Run Stage 1",
    endpoint: runId => `/pipeline/stage1/restore/${runId}`,
  },
  {
    id: "stage2",
    label: "Stage 2",
    title: "Damage & uncertainty",
    description: "Localize damage and visually uncertain regions.",
    actionLabel: "Run Stage 2",
    endpoint: runId => `/pipeline/stage2/damage/${runId}`,
  },
  {
    id: "stage3",
    label: "Stage 3",
    title: "Layout",
    description: "Find text-bearing regions without claiming line segmentation.",
    actionLabel: "Run Stage 3",
    endpoint: runId => `/pipeline/stage3/layout/${runId}`,
  },
  {
    id: "stage4",
    label: "Stage 4",
    title: "Line segmentation",
    description: "Script-aware, loss-aware physical line segmentation.",
    actionLabel: "Run Stage 4",
    endpoint: runId => `/pipeline/stage4/segment/${runId}`,
  },
  {
    id: "stage5",
    label: "Stage 5",
    title: "HTR",
    description: "Execute A/B, trigger Provider C when policy requires it, preserve canonical H(A+B), and route adaptively.",
    actionLabel: "Run Stage 5",
    endpoint: runId => `/pipeline/stage5/htr/${runId}`,
  },
  {
    id: "stage6",
    label: "Stage 6",
    title: "Reconstruction & trust",
    description: "Run 6A–6F and Stage-6G scholar-last adaptive routing.",
    actionLabel: "Run Stage 6",
    endpoint: runId => `/pipeline/stage6/run/${runId}`,
  },
];

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};

  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const detail =
      payload.detail ||
      payload.error ||
      `${response.status} ${response.statusText}`;
    const error = new Error(detail);
    error.payload = payload;
    throw error;
  }

  return payload;
}

function setApiStatus(text, tone = "neutral") {
  const el = document.getElementById("apiStatus");
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function setRunState(text) {
  document.getElementById("runState").textContent = text;
}

function renderStages() {
  const host = document.getElementById("stageTrack");
  host.innerHTML = "";

  stages.forEach((stage, index) => {
    const status = state.stageStatus[stage.id] || "pending";
    const card = document.createElement("article");
    card.className = `stage-card ${status === "done" ? "done" : status === "running" ? "running" : status === "failed" ? "failed" : ""}`;

    card.innerHTML = `
      <div class="stage-number">${stage.label}</div>
      <h3>${stage.title}</h3>
      <p>${stage.description}</p>
      <button
        data-stage-id="${stage.id}"
        ${state.runId ? "" : "disabled"}
      >${status === "running" ? "Running…" : stage.actionLabel}</button>
    `;

    card.querySelector("button").onclick = () => runStage(index);
    host.appendChild(card);
  });
}

function addImages(images = {}) {
  Object.entries(images || {}).forEach(([name, src]) => {
    if (typeof src === "string" && src.startsWith("data:image/")) {
      state.images[name] = src;
    }
  });
  renderImages();
}

function renderImages() {
  const host = document.getElementById("imageGrid");
  const entries = Object.entries(state.images);

  if (!entries.length) {
    host.innerHTML = '<div class="empty">No run images yet.</div>';
    return;
  }

  host.innerHTML = entries.map(([name, src]) => `
    <div class="image-card">
      <img src="${src}" alt="${esc(name)}" />
      <strong>${esc(name.replaceAll("_", " "))}</strong>
    </div>
  `).join("");
}

function collectMetrics(payload) {
  const metrics = [];

  const push = (name, value) => {
    if (value !== null && value !== undefined && value !== "") {
      metrics.push([name, value]);
    }
  };

  push("research stage", payload.research_stage);
  push("stage name", payload.stage_name);
  push("num lines", payload.num_lines);
  push("num regions", payload.num_regions);

  const signals =
    payload.signals ||
    payload.final_orchestration?.signals ||
    payload.orchestration_report?.signals ||
    {};

  push("S", signals.S);
  push("H · canonical A+B", signals.H);
  push("T", signals.T);

  if (payload.metrics) {
    push("segmentation confidence", payload.metrics.segmentation_confidence);
    push("losslessness", payload.metrics.losslessness_score);
  }

  if (payload.stage6) {
    push("reconstructed", payload.stage6.reconstructed_lines);
    push("unresolved", payload.stage6.unresolved_lines);
    push("translation eligible", payload.stage6.translation_eligible_lines);
  }

  return metrics.slice(0, 12);
}

function renderResult(title, payload) {
  document.getElementById("resultTitle").textContent = title;

  const summaryParts = [];
  if (payload.pipeline_position) summaryParts.push(payload.pipeline_position);
  if (payload.stage_name) summaryParts.push(payload.stage_name);
  if (payload.next_action) summaryParts.push(`Next: ${payload.next_action}`);
  if (payload.summary && Array.isArray(payload.summary)) {
    summaryParts.push(payload.summary.join(" "));
  }

  document.getElementById("resultSummary").textContent =
    summaryParts.join(" · ") || "Stage completed.";

  const metrics = collectMetrics(payload);
  // GENERIC_PLATFORM_METRIC_LAYOUT_FIX_V1
  document.getElementById("metricGrid").innerHTML = metrics.map(([name, value]) => {
    const textValue =
      typeof value === "string" && String(value).length > 12;

    return `
      <div class="metric-card">
        <span>${esc(name)}</span>
        <strong class="${textValue ? "metric-text-value" : ""}">
          ${esc(fmt(value))}
        </strong>
      </div>
    `;
  }).join("");

  document.getElementById("jsonPanel").textContent =
    JSON.stringify(payload, null, 2);

  if (payload.raw_image) addImages({ raw: payload.raw_image });
  if (payload.images) addImages(payload.images);
}

function renderDecision(orchestration) {
  const host = document.getElementById("decisionPanel");
  const final = orchestration?.final_decision || orchestration?.final_orchestration || {};
  const signals = final.signals || orchestration?.signals || {};

  if (!final || !Object.keys(final).length) {
    host.textContent = "No final orchestration decision is available yet.";
    return;
  }

  host.innerHTML = `
    <table class="evidence-table">
      <tr><td>Status</td><td>${esc(final.overall_status ?? final.status ?? "—")}</td></tr>
      <tr><td>Next action</td><td>${esc(final.next_action ?? "—")}</td></tr>
      <tr><td>S</td><td>${esc(fmt(signals.S))}</td></tr>
      <tr><td>H</td><td>${esc(fmt(signals.H))}</td></tr>
      <tr><td>T</td><td>${esc(fmt(signals.T))}</td></tr>
      <tr><td>Scholar review</td><td>${final.scholar_review_required === true ? "Required" : "Not currently required"}</td></tr>
    </table>
    <div class="guardrail">
      S, H and T are evidence/readiness/trust signals. They are not CER/WER,
      probability of correctness, or scholar-verified accuracy.
    </div>
  `;
}


// GENERIC_GROUND_TRUTH_ATTACHMENT_UI_V1
function setGtAttachStatus(text, tone = "neutral") {
  const el = document.getElementById("gtAttachStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function setGtAttachMessage(text, isError = false) {
  const el = document.getElementById("gtAttachMessage");
  if (!el) return;
  el.textContent = text;
  el.className = isError ? "gt-message error" : "gt-message";
}

function ensureGroundTruthLineTemplate(lines = []) {
  const textarea = document.getElementById("gtLines");
  if (!textarea || textarea.value.trim()) return;

  const ids = lines.map(row => row?.line_id).filter(Boolean);
  if (!ids.length) return;

  textarea.value = ids.map(id => `${id}\t`).join("\n");
}

function parseGroundTruthLines(raw) {
  const rows = String(raw || "")
    .split(/\r?\n/)
    .map(line => line.trimEnd())
    .filter(line => line.trim());

  if (!rows.length) {
    throw new Error("No verified Ground-Truth lines were entered.");
  }

  const seen = new Set();

  return rows.map((line, index) => {
    const tabIndex = line.indexOf("\t");

    if (tabIndex < 0) {
      throw new Error(
        `GT row ${index + 1} must use a TAB between line_id and reference text.`
      );
    }

    const lineId = line.slice(0, tabIndex).trim();
    const referenceText = line.slice(tabIndex + 1).trim();

    if (!/^line_\d{3,}$/.test(lineId)) {
      throw new Error(`Invalid line ID on GT row ${index + 1}: ${lineId}`);
    }

    if (!referenceText) {
      throw new Error(`Reference text is empty for ${lineId}.`);
    }

    if (seen.has(lineId)) {
      throw new Error(`Duplicate Ground-Truth line ID: ${lineId}`);
    }

    seen.add(lineId);

    return {
      line_id: lineId,
      reference_text: referenceText,
      text_status: "verified_reference",
      alignment_status:
        document.getElementById("gtAlignmentStatus")?.value || "verified",
    };
  });
}

async function attachVerifiedGroundTruth() {
  if (!state.runId) {
    setGtAttachMessage(
      "Create or load a manuscript run before attaching Ground Truth.",
      true
    );
    return;
  }

  const confirmation =
    document.getElementById("gtConfirmVerified")?.checked === true;

  if (!confirmation) {
    setGtAttachMessage(
      "Confirmation is required: machine output cannot self-declare as Ground Truth.",
      true
    );
    return;
  }

  const verifiedByRole =
    document.getElementById("gtVerifiedByRole")?.value?.trim() || "";

  if (!verifiedByRole) {
    setGtAttachMessage(
      "Enter the accountable verifier role, such as Professor or Sanskrit scholar.",
      true
    );
    return;
  }

  let lines;

  try {
    lines = parseGroundTruthLines(
      document.getElementById("gtLines")?.value || ""
    );
  } catch (err) {
    setGtAttachMessage(err.message, true);
    return;
  }

  const sourceStatus =
    document.getElementById("gtSourceStatus")?.value || "scholar_verified";
  const alignmentStatus =
    document.getElementById("gtAlignmentStatus")?.value || "verified";
  const script =
    document.getElementById("gtScript")?.value?.trim() || "Devanagari";
  const language =
    document.getElementById("gtLanguage")?.value?.trim() || "Sanskrit";
  const datasetId =
    document.getElementById("gtDatasetId")?.value?.trim() ||
    `${state.runId}_verified_gt_v1`;
  const notes =
    document.getElementById("gtNotes")?.value?.trim() || "";

  const payload = {
    ground_truth: {
      schema_version: "1.0-generic-run",
      dataset_id: datasetId,
      case_id: state.runId,
      manuscript_id: state.runId,
      script,
      language,
      reference_scope: "diplomatic_transcription_reference",
      source_text_status: sourceStatus,
      physical_line_alignment_status: alignmentStatus,
      professor_supplied_physical_line_breaks: false,
      translation_ground_truth_available: false,
      lines,
      scientific_guardrails: {
        cer_wer_allowed_for_stage5_transcription: true,
        cer_wer_are_not_H: true,
        cer_wer_are_not_T: true,
        stage6_translation_is_not_professor_validated: true,
        professor_line_breaks_not_claimed:
          sourceStatus !== "professor_supplied",
      },
    },
    provenance: {
      schema_version: "1.0-generic-run",
      dataset_id: datasetId,
      verification: {
        status: sourceStatus,
        verified_by_role: verifiedByRole,
        physical_line_alignment_status: alignmentStatus,
        notes,
      },
      evaluation_source: "generic_run_ground_truth_runtime_v1",
      permissions_note:
        "Redistribution/publication rights for supplied reference text remain the responsibility of the research project.",
    },
  };

  const button = document.getElementById("gtAttachButton");

  if (button) {
    button.disabled = true;
    button.textContent = "Validating & Benchmarking…";
  }

  setGtAttachStatus("VALIDATING", "warning");
  setGtAttachMessage(
    "Validating provenance, line alignment, and A/B/C benchmark contract."
  );

  try {
    const response = await requestJson(
      `/pipeline/ground-truth/${state.runId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    const best = response?.best_provider_by_strict_cer;

    setGtAttachStatus("VERIFIED GT ATTACHED", "success");
    setGtAttachMessage(
      best
        ? `Ground Truth attached. Provider ${best} has the lowest strict CER for this run.`
        : "Ground Truth attached and benchmarked."
    );

    await refreshOrchestration();
  } catch (err) {
    const detail =
      err?.payload?.detail ||
      err?.payload?.error ||
      err?.message ||
      "Ground-Truth attachment failed.";

    setGtAttachStatus("REJECTED", "danger");
    setGtAttachMessage(detail, true);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Attach Verified GT & Benchmark A/B/C";
    }
  }
}


function renderStage5(payload) {
  const host = document.getElementById("stage5Panel");
  const readiness = payload.htr_readiness || {};
  const transcription = payload.transcription || {};
  const providerC = payload.provider_c || {};
  const adaptive = payload.adaptive_evidence || payload.adaptive_provider_c || {};
  const lines = transcription.lines || [];

  ensureGroundTruthLineTemplate(lines);

  // GENERIC_GROUND_TRUTH_READY_MESSAGE_POLISH_V1
  if (payload.ground_truth?.available === true) {
    setGtAttachStatus("VERIFIED GT ATTACHED", "success");
    setGtAttachMessage(
      "Verified transcription Ground Truth is attached; A/B/C benchmark metrics are shown below."
    );
  } else {
    setGtAttachStatus("NOT ATTACHED", "neutral");
    setGtAttachMessage(
      "Stage 5 is ready. Enter only externally verified line-aligned transcription, provide verifier provenance, then attach to benchmark Providers A/B/C."
    );
  }

const groundTruth = payload.ground_truth || {};
  const gtBenchmark = payload.ground_truth_benchmark || {};
  const gtProviders = gtBenchmark.providers || {};
  const gtRanking = gtBenchmark.rankings?.strict_cer_best_to_worst || [];
  const gtBestKey = gtRanking.length ? gtRanking[0] : null;
  const gtBest = gtBestKey ? gtProviders[gtBestKey] : null;
  const gtAvailable =
    groundTruth.available === true ||
    Object.keys(gtProviders).length > 0;

  const pctMetric = value =>
    typeof value === "number"
      ? `${(value * 100).toFixed(2)}%`
      : "N/A";

  const gtProviderRows = ["A", "B", "C"]
    .filter(key => gtProviders[key])
    .map(key => {
      const item = gtProviders[key];
      const bestBadge =
        key === gtBestKey ? " · best strict CER" : "";
      return `
        <tr>
          <td>Provider ${esc(key)}${esc(bestBadge)}</td>
          <td>${esc(pctMetric(item.strict_cer))}</td>
          <td>${esc(pctMetric(item.wer))}</td>
          <td>${esc(pctMetric(item.content_cer))}</td>
        </tr>
      `;
    })
    .join("");

  const providerAText =
    transcription.provider_a_page_display ||
    lines.map(x => x.provider_a_devanagari_display || "").filter(Boolean).join("\n");

  const providerBText =
    transcription.provider_b_page_display ||
    lines.map(x => x.provider_b_devanagari_display || "").filter(Boolean).join("\n");

  const providerCText =
    providerC.page_display ||
    (providerC.lines || [])
      .map(x => x.devanagari_display || x.devanagari_text || "")
      .filter(Boolean)
      .join("\n");

  const hAB =
    adaptive.H_ab ??
    adaptive.baseline?.H_ab ??
    payload.signals?.H ??
    readiness.H;

  const hAC = adaptive.H_ac ?? null;
  const hBC = adaptive.H_bc ?? null;

  const cAttempted =
    adaptive.attempted === true ||
    adaptive.provider_c_executed === true ||
    providerC.executed === true;

  const cStatus =
    adaptive.status ||
    (providerC.executed ? "completed" : providerC.available ? "available" : "not triggered");

  host.innerHTML = `
    <table class="evidence-table">
      <tr><td>Canonical H(A+B)</td><td>${esc(fmt(hAB))}</td></tr>
      <tr><td>A/B provider agreement</td><td>${esc(fmt(payload.provider_comparison?.mean_content_char_similarity))}</td></tr>
      <tr><td>Provider C attempted</td><td>${cAttempted ? "Yes" : "No"}</td></tr>
      <tr><td>Provider C status</td><td>${esc(cStatus)}</td></tr>
      <tr><td>H(A+C) · comparative evidence</td><td>${esc(fmt(hAC))}</td></tr>
      <tr><td>H(B+C) · comparative evidence</td><td>${esc(fmt(hBC))}</td></tr>
      <tr><td>Best comparative pair</td><td>${esc(adaptive.best_candidate_pair ?? "—")}</td></tr>
      <tr><td>ΔH vs canonical A+B</td><td>${esc(fmt(adaptive.delta_H))}</td></tr>
      <tr><td>Ground Truth attached</td><td>${gtAvailable ? "Yes" : "No"}</td></tr>
      <tr><td>Best strict CER</td><td>${gtBest ? `${esc(gtBestKey)} · ${esc(pctMetric(gtBest.strict_cer))}` : "N/A"}</td></tr>
    </table>

    <div class="provider-columns provider-columns-three">
      <div class="provider-box">
        <strong>Provider A · TrOCR Sanskrit baseline</strong>
        <pre>${esc(providerAText || "No display transcription returned.")}</pre>
      </div>
      <div class="provider-box">
        <strong>Provider B · TrOCR Vedic Devanagari</strong>
        <pre>${esc(providerBText || "No display transcription returned.")}</pre>
      </div>
      <div class="provider-box">
        <strong>Provider C · Qwen visual HTR candidate</strong>
        <pre>${esc(providerCText || (cAttempted ? "Provider C executed, but no display transcription was returned." : "Provider C was not triggered for this run."))}</pre>
      </div>
    </div>

${gtAvailable ? `
      <div class="routing-recommendations">
        <strong>Verified transcription Ground Truth</strong>
        <table class="evidence-table">
          <tr>
            <td>Provider</td>
            <td>Strict CER</td>
            <td>WER</td>
            <td>Content CER</td>
          </tr>
          ${gtProviderRows}
        </table>
        <div>
          Primary metric: strict CER. Lower is better.
          WER is spacing/tokenization sensitive. Content CER is secondary only.
          ${gtBest ? `Best strict CER: Provider ${esc(gtBestKey)} · ${esc(pctMetric(gtBest.strict_cer))}.` : ""}
        </div>
      </div>
    ` : `
      <div class="guardrail">
        No externally verified transcription Ground Truth is attached to this run.
        CER/WER/content-CER therefore remain unavailable.
      </div>
    `}

    <div class="guardrail">
      H(A+B) remains the canonical Stage-5 readiness signal. H(A+C), H(B+C)
      and ΔH are comparative evidence only; none of these values is recognition
      accuracy or scholar verification. Provider C remains an independent
      candidate evidence stream.
    </div>
  `;
}

// GENERIC_PLATFORM_ROUTING_LABEL_POLISH_V1
function humanizeRoutingValue(value) {
  if (value === null || value === undefined || value === "") return "—";

  const text = String(value);

  const known = {
    scholar_review_after_machine_exhaustion:
      "Scholar review after machine exhaustion",
    run_third_htr_provider:
      "Provider C / third HTR provider",
    no_additional_executable_machine_strategy_available:
      "No additional executable machine strategy available",
    configured_machine_retry_budget_exhausted:
      "Configured machine retry budget exhausted",
    machine_retry_required:
      "Machine retry required",
    route_to_scholar_review:
      "Route to scholar review",
    untrusted_or_unresolved:
      "Untrusted or unresolved",
    stage5_recognition:
      "Stage 5 recognition",
    run_alternate_visual_htr:
      "Alternate visual HTR",
    run_expanded_context_retrieval:
      "Expanded context retrieval",
  };

  if (known[text]) return known[text];

  const words = text.replaceAll("_", " ").trim();
  if (!words) return "—";
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function renderStage6(payload) {
  const host = document.getElementById("stage6Panel");
  const stage6 = payload.stage6 || {};
  const routing = payload.adaptive_routing || payload.final_orchestration || {};
  const signals = payload.signals || {};
  const retryState = routing.retry_state || {};
  const attemptedActions = Array.isArray(retryState.attempted_actions)
    ? retryState.attempted_actions
    : [];
  const recommendations = Array.isArray(routing.recommendations)
    ? routing.recommendations
    : [];

  host.innerHTML = `
    <table class="evidence-table">
      <tr><td>T</td><td>${esc(fmt(signals.T))}</td></tr>
      <tr><td>Trust status</td><td>${esc(humanizeRoutingValue(stage6.trust_status))}</td></tr>
      <tr><td>Reconstructed lines</td><td>${esc(stage6.reconstructed_lines ?? "—")}</td></tr>
      <tr><td>Partially supported lines</td><td>${esc(stage6.partially_supported_lines ?? "—")}</td></tr>
      <tr><td>Abstained lines</td><td>${esc(stage6.abstained_lines ?? "—")}</td></tr>
      <tr><td>Normalized lines</td><td>${esc(stage6.normalized_lines ?? "—")}</td></tr>
      <tr><td>Unresolved lines</td><td>${esc(stage6.unresolved_lines ?? "—")}</td></tr>
      <tr><td>Translation eligible lines</td><td>${esc(stage6.translation_eligible_lines ?? "—")}</td></tr>
      <tr><td>Adaptive decision</td><td>${esc(humanizeRoutingValue(routing.decision))}</td></tr>
      <tr><td>Selected retry</td><td>${esc(humanizeRoutingValue(routing.selected_retry_action))}</td></tr>
      <tr><td>Machine retry attempts</td><td>${esc(retryState.attempt_count ?? 0)}</td></tr>
      <tr><td>Attempted machine actions</td><td>${esc(attemptedActions.length ? attemptedActions.map(humanizeRoutingValue).join(", ") : "None recorded")}</td></tr>
      <tr><td>Machine retry exhausted</td><td>${routing.machine_retry_exhausted === true ? "Yes" : "No"}</td></tr>
      <tr><td>Exhaustion reason</td><td>${esc(humanizeRoutingValue(routing.machine_retry_exhaustion_reason))}</td></tr>
      <tr><td>Scholar review</td><td>${routing.scholar_review_required ? "Required" : "Not currently required"}</td></tr>
    </table>

    ${recommendations.length ? `
      <div class="routing-recommendations">
        <strong>Routing interpretation</strong>
        ${recommendations.map(item => `<div>${esc(item)}</div>`).join("")}
      </div>
    ` : ""}

    <div class="guardrail">
      Stage 6 is abstention-capable. T is semantic/transcription trust-readiness,
      not calibrated probability or accuracy. Linguistic plausibility or retrieval
      evidence cannot override insufficient visual manuscript evidence, and this
      result does not imply scholar-validated translation.
    </div>
  `;
}

async function refreshOrchestration() {
  if (!state.runId) return null;

  let orchestration = null;

  try {
    orchestration = await requestJson(`/orchestration/${state.runId}`);
    renderDecision(orchestration);

    if (orchestration.images) addImages(orchestration.images);
  } catch (err) {
    console.warn("Orchestration refresh failed", err);
  }

  try {
    const uiState = await requestJson(`/pipeline/ui-state/${state.runId}`);

    if (uiState.stage5?.available) {
      renderStage5(uiState.stage5);
    }

    if (uiState.stage6?.available) {
      renderStage6(uiState.stage6);
    }
  } catch (err) {
    console.warn("Generic platform evidence refresh failed", err);
  }

  return orchestration;
}


// GENERIC_PLATFORM_RUN_STATE_HYDRATION_V1
const ACTIVE_RUN_STORAGE_KEY = "adaptive_manuscript_platform_active_run_id";

function persistActiveRun(runId) {
  if (!runId) return;

  try {
    localStorage.setItem(ACTIVE_RUN_STORAGE_KEY, runId);
  } catch (err) {
    console.warn("Could not persist active run_id in localStorage", err);
  }

  try {
    const url = new URL(window.location.href);
    url.searchParams.set("run_id", runId);
    window.history.replaceState({}, "", url);
  } catch (err) {
    console.warn("Could not persist run_id in the page URL", err);
  }
}

function restorableRunId() {
  try {
    const fromUrl = new URL(window.location.href).searchParams.get("run_id");
    if (fromUrl) return fromUrl;
  } catch (err) {
    console.warn("Could not read run_id from URL", err);
  }

  try {
    return localStorage.getItem(ACTIVE_RUN_STORAGE_KEY);
  } catch (err) {
    console.warn("Could not read active run_id from localStorage", err);
    return null;
  }
}

function nonEmptyObject(value) {
  return Boolean(
    value
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.keys(value).length
  );
}

function hydrateStageStatus(orchestration, uiState) {
  const reports = orchestration?.layer_reports || {};

  if (orchestration?.run_id) {
    state.stageStatus.stage0 = "done";
  }

  for (let stageNumber = 1; stageNumber <= 6; stageNumber += 1) {
    if (nonEmptyObject(reports[`layer${stageNumber}`])) {
      state.stageStatus[`stage${stageNumber}`] = "done";
    }
  }

  if (uiState?.stage5?.available) {
    for (let stageNumber = 0; stageNumber <= 5; stageNumber += 1) {
      state.stageStatus[`stage${stageNumber}`] = "done";
    }
  }

  if (uiState?.stage6?.available) {
    for (let stageNumber = 0; stageNumber <= 6; stageNumber += 1) {
      state.stageStatus[`stage${stageNumber}`] = "done";
    }
  }
}

function renderHydratedResult(uiState) {
  if (uiState?.stage6?.available) {
    const routing =
      uiState.stage6.final_orchestration
      || uiState.stage6.adaptive_routing
      || {};

    renderResult(
      "Stage 6 · Reconstruction & trust",
      {
        run_id: state.runId,
        research_stage: 6,
        stage_name: "semantic_interpretation_and_trust",
        signals: uiState.stage6.signals || {},
        next_action: routing.next_action,
        final_orchestration: routing,
      }
    );
    return;
  }

  if (uiState?.stage5?.available) {
    const routing = uiState.stage5.adaptive_evidence || {};

    renderResult(
      "Stage 5 · HTR",
      {
        run_id: state.runId,
        research_stage: 5,
        stage_name: "sequence_based_htr",
        signals: uiState.stage5.signals || {},
        next_action: routing.next_action,
      }
    );
  }
}

async function loadPersistedRunEvidence(runId) {
  const [orchestration, uiState] = await Promise.all([
    requestJson(`/orchestration/${runId}`),
    requestJson(`/pipeline/ui-state/${runId}`),
  ]);

  return { orchestration, uiState };
}

async function restorePersistedRun() {
  const runId = restorableRunId();
  if (!runId) return false;

  try {
    const { orchestration, uiState } =
      await loadPersistedRunEvidence(runId);

    if (
      orchestration?.run_id !== runId
      || uiState?.run_id !== runId
    ) {
      throw new Error("Backend run_id did not match requested restored run.");
    }

    state.runId = runId;
    state.stageStatus = {};
    state.lastPayload = {};
    state.images = {};

    persistActiveRun(runId);

    document.getElementById("runId").textContent = runId;
    document.getElementById("runAllButton").disabled = false;

    hydrateStageStatus(orchestration, uiState);
    renderStages();

    renderDecision(orchestration);

    if (orchestration.images) {
      addImages(orchestration.images);
    }

    if (uiState.stage5?.available) {
      renderStage5(uiState.stage5);
    }

    if (uiState.stage6?.available) {
      renderStage6(uiState.stage6);
    }

    renderHydratedResult(uiState);

    if (uiState.stage6?.available) {
      setRunState(
        "Existing run restored · Stage 6 backend evidence loaded"
      );
    } else if (uiState.stage5?.available) {
      setRunState(
        "Existing run restored · Stage 5 backend evidence loaded"
      );
    } else {
      setRunState("Existing run restored from persisted backend evidence");
    }

    return true;
  } catch (err) {
    console.warn("Persisted run restoration failed", err);
    return false;
  }
}

async function reconcileTransportFailure(stage) {
  try {
    const { orchestration, uiState } =
      await loadPersistedRunEvidence(state.runId);

    const reports = orchestration?.layer_reports || {};
    let completed = false;

    if (stage.id === "stage0") {
      completed = Boolean(orchestration?.run_id);
    } else if (stage.id === "stage5") {
      completed = uiState?.stage5?.available === true;
    } else if (stage.id === "stage6") {
      completed = uiState?.stage6?.available === true;
    } else {
      const number = Number(stage.id.replace("stage", ""));
      completed = nonEmptyObject(reports[`layer${number}`]);
    }

    if (!completed) return false;

    hydrateStageStatus(orchestration, uiState);
    state.stageStatus[stage.id] = "done";

    renderStages();
    renderDecision(orchestration);

    if (orchestration.images) {
      addImages(orchestration.images);
    }

    if (uiState.stage5?.available) {
      renderStage5(uiState.stage5);
    }

    if (uiState.stage6?.available) {
      renderStage6(uiState.stage6);
    }

    renderHydratedResult(uiState);

    setRunState(
      `${stage.label} completed on backend · browser response recovered from persisted evidence`
    );

    return true;
  } catch (reconcileErr) {
    console.warn("Backend completion reconciliation failed", reconcileErr);
    return false;
  }
}

async function runStage(index) {
  if (!state.runId) return;

  const stage = stages[index];
  const priorStatus = state.stageStatus[stage.id] || "pending";

  state.stageStatus[stage.id] = "running";
  renderStages();
  setRunState(`${stage.label} running…`);

  try {
    const payload = await requestJson(
      stage.endpoint(state.runId),
      { method: "POST" }
    );

    state.stageStatus[stage.id] = "done";
    state.lastPayload = payload;

    renderResult(`${stage.label} · ${stage.title}`, payload);

    if (stage.id === "stage5") renderStage5(payload);
    if (stage.id === "stage6") renderStage6(payload);

    await refreshOrchestration();

    setRunState(`${stage.label} completed`);
    renderStages();
    return payload;
  } catch (err) {
    const payload = err.payload || {};
    const transportLikeFailure =
      !err.payload
      || (
        typeof err.payload === "object"
        && Object.keys(err.payload).length === 0
      );

    if (transportLikeFailure && priorStatus !== "done") {
      const reconciled = await reconcileTransportFailure(stage);
      if (reconciled) {
        return {
          run_id: state.runId,
          reconciled_from_persisted_backend_evidence: true,
          stage_id: stage.id,
        };
      }
    }

    state.stageStatus[stage.id] = "failed";
    renderStages();

    renderResult(`${stage.label} failed`, payload);

    setRunState(`${stage.label} failed: ${err.message}`);
    throw err;
  }
}

async function upload() {
  const input = document.getElementById("fileInput");
  const file = input.files?.[0];

  if (!file) {
    setRunState("Choose an image before uploading.");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  document.getElementById("uploadButton").disabled = true;
  setRunState("Uploading manuscript…");

  try {
    const payload = await requestJson("/upload", {
      method: "POST",
      body: form,
    });

    state.runId = payload.run_id;
    persistActiveRun(state.runId);
    state.stageStatus = {};
    state.lastPayload = payload;
    state.images = {};

    document.getElementById("runId").textContent = state.runId;
    document.getElementById("runAllButton").disabled = false;

    if (payload.raw_image) addImages({ raw: payload.raw_image });
    renderResult("Stage 0 · Raw manuscript preserved", payload);
    renderStages();

    setRunState("Upload complete · ready for Stage 0 profiling");
  } catch (err) {
    setRunState(`Upload failed: ${err.message}`);
  } finally {
    document.getElementById("uploadButton").disabled = false;
  }
}

async function runAll() {
  if (!state.runId) return;

  const button = document.getElementById("runAllButton");
  button.disabled = true;
  button.textContent = "Pipeline running…";

  try {
    for (let i = 0; i < stages.length; i += 1) {
      if (state.stageStatus[stages[i].id] === "done") continue;
      await runStage(i);
    }

    setRunState("Current generic pipeline completed.");
  } catch (err) {
    console.error(err);
  } finally {
    button.disabled = false;
    button.textContent = "Run Current Pipeline";
  }
}

async function boot() {
  renderStages();
  await restorePersistedRun();

  document.getElementById("uploadButton").onclick = upload;
  document.getElementById("runAllButton").onclick = runAll;

  document.getElementById("gtAttachButton").onclick =
    attachVerifiedGroundTruth;

  document.getElementById("toggleJsonButton").onclick = () => {
    document.getElementById("jsonPanel").classList.toggle("hidden");
  };

  try {
    const health = await requestJson("/health");
    setApiStatus(
      `API READY · ${health.pipeline || "PIPELINE"}`,
      "ok"
    );
  } catch (err) {
    setApiStatus("API ERROR", "fail");
    setRunState(`API unavailable: ${err.message}`);
  }
}

boot();
