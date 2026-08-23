const state = {
  catalog: null,
  selectedCaseId: "golden_success",
  selectedModeId: "auto_adaptive",
  selectedTranslationLang: "hindi",
};

const fallbackGlossary = [
  {
    term: "HTR",
    expansion: "Handwritten Text Recognition",
    meaning: "The stage that reads handwritten or historical text from manuscript images. Think of it as OCR designed for difficult handwriting.",
    direction: "Higher quality is better"
  },
  {
    term: "Ground Truth",
    expansion: "Verified Reference Transcription",
    meaning: "A trusted human/scholar-written version of the manuscript used as the answer key for measuring recognition accuracy.",
    direction: "Required for real CER/WER"
  },
  {
    term: "CER",
    expansion: "Character Error Rate",
    meaning: "How many characters the machine got wrong compared with verified ground truth. CER of 10% means roughly 10 character-level errors per 100 characters.",
    direction: "Lower is better"
  },
  {
    term: "WER",
    expansion: "Word Error Rate",
    meaning: "How many words are inserted, deleted or substituted compared with the verified reference text.",
    direction: "Lower is better"
  },
  {
    term: "S",
    expansion: "Segmentation Readiness",
    meaning: "Our Stage-4 evidence score indicating whether the manuscript has been split into usable text lines well enough to continue.",
    direction: "Higher is stronger evidence"
  },
  {
    term: "H",
    expansion: "HTR Readiness",
    meaning: "Our Stage-5 evidence score combining completion, script integrity, decoder reliability, sequence quality and agreement between recognition providers.",
    direction: "Higher is stronger evidence"
  },
  {
    term: "T",
    expansion: "Semantic Trust",
    meaning: "Our Stage-6 trust signal asking whether reconstruction evidence is strong enough to permit a trusted downstream interpretation or translation.",
    direction: "Higher is stronger evidence"
  },
  {
    term: "Provider",
    expansion: "Independent Recognition Capability",
    meaning: "A model or engine used to read the manuscript, such as TrOCR or Qwen3-VL. Multiple providers give independent evidence.",
    direction: "Diversity reduces single-model dependence"
  },
  {
    term: "Adaptive Retry",
    expansion: "Targeted Second Attempt",
    meaning: "The orchestrator selectively invokes another capability only when evidence shows the current path is weak.",
    direction: "Triggered only when useful"
  },
  {
    term: "ΔH",
    expansion: "Change in HTR Readiness",
    meaning: "How much the HTR readiness changed after a retry. A negative ΔH means the retry did not improve the current evidence metric.",
    direction: "Positive means improvement"
  },
  {
    term: "Abstention",
    expansion: "Refusal to Pretend Certainty",
    meaning: "Instead of generating a fluent but unsupported answer, the system stops and says the evidence is insufficient.",
    direction: "A safety feature"
  },
  {
    term: "Scholar Review",
    expansion: "Human Expert Escalation",
    meaning: "The final route when the currently available machine capabilities cannot produce enough evidence for a safe conclusion.",
    direction: "Used after machine evidence is exhausted"
  },
];

const orchestratorSteps = [
  ["Observe", "Read stage evidence such as S, H, T, provider agreement and unresolved lines."],
  ["Diagnose", "Identify where the weakness is: segmentation, recognition, reconstruction or trust."],
  ["Select", "Choose the next useful capability instead of invoking every model."],
  ["Retry", "Re-run only the weak part, for example an independent HTR provider."],
  ["Compare", "Measure whether new evidence improved the decision state, such as ΔH."],
  ["Decide", "Proceed, keep as candidate, abstain, or route to scholar review."],
];

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function metricTone(key, value) {
  if (key === "S") return value >= 0.8 ? "success" : "warning";
  if (key === "T") return value > 0.5 ? "success" : "danger";
  if (key.startsWith("H")) return value >= 0.5 ? "success" : "warning";
  if (key === "delta_H") return value >= 0 ? "success" : "danger";
  return "success";
}

function renderModes() {
  const host = document.getElementById("modeSwitcher");
  const desc = document.getElementById("modeDescription");
  host.innerHTML = "";

  state.catalog.available_execution_modes.forEach(mode => {
    const button = document.createElement("button");
    button.className = `mode-button ${mode.id === state.selectedModeId ? "active" : ""}`;
    button.textContent = mode.label;
    button.onclick = () => {
      state.selectedModeId = mode.id;
      renderModes();
    };
    host.appendChild(button);
  });

  const active = state.catalog.available_execution_modes.find(m => m.id === state.selectedModeId);
  desc.textContent = active ? active.description : "";
}

function renderStages() {
  const host = document.getElementById("stageTrack");
  host.innerHTML = "";

  state.catalog.stage_model.forEach(stage => {
    const card = document.createElement("article");
    const adaptive = stage.stage >= 4;
    card.className = `stage-card ${adaptive ? "adaptive-stage" : "core-stage"}`;
    card.innerHTML = `
      <div class="stage-number">STAGE ${stage.stage}</div>
      <h3>${stage.name}</h3>
      <p>${stage.role}</p>
    `;
    host.appendChild(card);
  });
}

function renderOrchestrator() {
  const host = document.getElementById("orchestratorLoop");
  host.innerHTML = "";

  orchestratorSteps.forEach(([title, text], idx) => {
    const step = document.createElement("div");
    step.className = "orchestrator-step";
    step.innerHTML = `
      <span class="step-num">${idx + 1}</span>
      <h4>${title}</h4>
      <p>${text}</p>
    `;
    host.appendChild(step);
  });

  const examples = document.getElementById("intelligenceExamples");
  examples.innerHTML = `
    <div class="intelligence-example">
      <strong>Golden Case · Avoid unnecessary AI</strong>
      Stage-4 produced 11/11 lines with S = 0.8854, so the orchestrator accepted the classical segmentation and did not spend a Qwen call.
    </div>
    <div class="intelligence-example warning">
      <strong>Difficult Case · Retry only the weak stage</strong>
      H(A+B) = 0.3371 and T = 0.0 indicated recognition/trust weakness. The orchestrator invoked Qwen as independent Provider C, measured ΔH = -0.0073, rejected the ineffective retry and routed to scholar review.
    </div>
  `;
}

function renderCaseCards() {
  const host = document.getElementById("caseGrid");
  host.innerHTML = "";

  state.catalog.cases.forEach(item => {
    const selected = item.id === state.selectedCaseId;
    const success = item.id === "golden_success";
    const card = document.createElement("article");
    card.className = `panel case-card ${selected ? "selected" : ""}`;

    const metricEntries = Object.entries(item.metrics || {})
      .filter(([_, value]) => value !== null && value !== undefined)
      .slice(0, 5)
      .map(([key, value]) => `
        <div class="mini-metric">
          <span>${key.replaceAll("_", " ")}</span>
          <strong>${fmt(value)}</strong>
        </div>
      `).join("");

    card.innerHTML = `
      <div class="section-kicker">${item.tagline}</div>
      <h3>${item.display_name}</h3>
      <div class="case-role ${success ? "success" : "review"}">
        ${success ? "CAPABILITY" : "JUDGEMENT + SAFE ABSTENTION"}
      </div>
      <ul class="evidence-list">
        ${(item.headline_evidence || []).map(x => `<li>${x}</li>`).join("")}
      </ul>
      <div class="case-metrics">${metricEntries}</div>
    `;

    card.onclick = async () => {
      state.selectedCaseId = item.id;
      renderCaseCards();
      await renderSelectedCase();
      document.getElementById("caseDetailSection").scrollIntoView({ behavior: "smooth" });
    };

    host.appendChild(card);
  });
}

function renderMetricGrid(caseItem) {
  const host = document.getElementById("metricGrid");
  host.innerHTML = "";

  Object.entries(caseItem.metrics || {}).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    const card = document.createElement("div");
    card.className = `metric-card ${metricTone(key, value)}`;
    card.innerHTML = `
      <span>${key.replaceAll("_", " ")}</span>
      <strong>${fmt(value)}</strong>
    `;
    host.appendChild(card);
  });

  document.getElementById("metricDisclaimer").textContent =
    caseItem.id === "difficult_adaptive"
      ? "S = segmentation readiness, H = HTR readiness, T = semantic trust. These are evidence signals, not accuracy probabilities."
      : "S/H/T remain evidence and trust signals. Professor-grounded CER/WER are shown separately below as Stage-5 transcription accuracy metrics.";
}

function renderDecisionPanel(caseItem, uiState) {
  const host = document.getElementById("decisionPanel");
  host.innerHTML = "";

  const flow = document.createElement("div");
  flow.className = "decision-flow";

  let steps = [];
  if (caseItem.id === "difficult_adaptive") {
    steps = uiState.adaptive_path || [];
  } else {
    steps = [
      "Classical Stage-4 segmentation accepted because S was strong",
      "Qwen3-VL Stage-5 HTR completed for 11/11 lines",
      "Stage-6 normalization candidates generated",
      "Hindi and English translation candidates exposed for scholar assistance",
      "Scholar validation remains pending",
    ];
  }

  steps.forEach((step, index) => {
    const el = document.createElement("div");
    let tone = "";
    if (caseItem.id === "difficult_adaptive") {
      if (index >= steps.length - 2) tone = "danger";
      else if (index >= 5) tone = "warning";
    } else if (index === steps.length - 1) {
      tone = "warning";
    } else {
      tone = "success";
    }
    el.className = `decision-step ${tone}`;
    el.textContent = step;
    flow.appendChild(el);
  });

  host.appendChild(flow);
}

function renderEvidencePanel(caseItem, uiState) {
  const host = document.getElementById("evidencePanel");

  if (caseItem.id === "difficult_adaptive") {
    const evidence = uiState.stage5_evidence || {};
    host.innerHTML = `
      <table class="evidence-table">
        ${Object.entries(evidence).map(([k, v]) => `
          <tr>
            <td>${k.replaceAll("_", " ")}</td>
            <td>${fmt(v)}</td>
          </tr>
        `).join("")}
      </table>
      <div class="metric-disclaimer">
        Provider-C retry selected ${uiState.provider_retry?.selected_pair || "—"};
        ΔH = ${fmt(uiState.provider_retry?.delta_H)}.
        The existing readiness metric did not improve.
      </div>
    `;
  } else {
    const metrics = caseItem.metrics || {};
    host.innerHTML = `
      <table class="evidence-table">
        <tr><td>Stage-4 accepted lines</td><td>11 / 11</td></tr>
        <tr><td>Stage-5 recognized lines</td><td>${metrics.recognized_lines ?? "—"} / 11</td></tr>
        <tr><td>Stage-6 normalized candidates</td><td>${metrics.normalized_lines ?? "—"} / 11</td></tr>
        <tr><td>Stage-6 translated candidates</td><td>${metrics.translated_lines ?? "—"} / 11</td></tr>
        <tr><td>Page-level translation</td><td>Available</td></tr>
      </table>
      <div class="metric-disclaimer">
        Recognition completion does not equal scholar-verified transcription accuracy.
      </div>
    `;
  }
}

function renderScientificPanel(caseItem, uiState) {
  const host = document.getElementById("scientificPanel");
  const sci = caseItem.scientific_status || {};
  const isSuccess = caseItem.id === "golden_success";
  const gt = uiState?.professor_ground_truth || null;
  const gtMetrics = uiState?.metrics?.professor_gt_stage5 || null;
  const gtAvailable = isSuccess && gt?.available === true;
  const bestCer = gtMetrics?.provider_c?.strict_cer_percent;

  const cerDisplay =
    gtAvailable && typeof bestCer === "number"
      ? `${bestCer.toFixed(2)}% · Provider C`
      : (sci.cer ?? "Unavailable");

  const werDisplay =
    gtAvailable
      ? "Available · spacing-sensitive"
      : (sci.wer ?? "Unavailable");

  host.innerHTML = `
    <div class="status-box ${isSuccess ? "success" : "review"}">
      <strong>${sci.display_label || caseItem.status}</strong>
      ${isSuccess
        ? gtAvailable
          ? "Professor-supplied Sanskrit Ground Truth is now available for Stage-5 transcription evaluation. Stage-6 translation remains an AI candidate pending scholar validation."
          : "A useful end-to-end candidate exists, but verified reference transcription is still pending."
        : "The system intentionally withholds a safe translation and routes the manuscript for expert review."}
    </div>
    <table class="evidence-table">
      <tr><td>Ground truth available</td><td>${gtAvailable ? "Yes" : (sci.ground_truth_available ? "Yes" : "No")}</td></tr>
      <tr><td>Best strict CER</td><td>${cerDisplay}</td></tr>
      <tr><td>WER</td><td>${werDisplay}</td></tr>
      <tr><td>Scholar-verified translation</td><td>${sci.scholar_verified ? "Yes" : "No"}</td></tr>
    </table>
  `;
}

// PROFESSOR_GT_RENDER_V1
function renderProfessorGroundTruth(caseItem, uiState, summary) {
  const panel = document.getElementById("professorGtPanel");
  const host = document.getElementById("professorGtContent");
  const status = document.getElementById("professorGtStatus");

  const gt = uiState?.professor_ground_truth || null;
  const metrics = uiState?.metrics?.professor_gt_stage5 || null;
  const benchmark = summary?.stage5?.professor_gt_benchmark || null;

  if (
    caseItem.id !== "golden_success" ||
    gt?.available !== true ||
    !metrics
  ) {
    panel.classList.add("hidden");
    host.innerHTML = "";
    return;
  }

  panel.classList.remove("hidden");
  status.textContent = gt.status_label || "PROFESSOR GROUND TRUTH AVAILABLE";

  const providerMeta = benchmark?.providers || {};

  const providers = [
    {
      key: "A",
      fallbackLabel: "TrOCR Sanskrit baseline",
      metric: metrics.provider_a,
      meta: providerMeta.A
    },
    {
      key: "B",
      fallbackLabel: "TrOCR Vedic Devanagari",
      metric: metrics.provider_b,
      meta: providerMeta.B
    },
    {
      key: "C",
      fallbackLabel: "Qwen3-VL via Bedrock Mantle",
      metric: metrics.provider_c,
      meta: providerMeta.C
    }
  ];

  const cards = providers.map(item => {
    const strictCer =
      typeof item.metric?.strict_cer_percent === "number"
        ? `${item.metric.strict_cer_percent.toFixed(2)}%`
        : "—";

    const contentCer =
      typeof item.metric?.content_cer_percent === "number"
        ? `${item.metric.content_cer_percent.toFixed(2)}%`
        : "—";

    const wer =
      typeof item.metric?.wer_percent === "number"
        ? `${item.metric.wer_percent.toFixed(2)}%`
        : "—";

    const label = item.meta?.label || item.fallbackLabel;
    const model = item.meta?.model_id || "";
    const best = metrics.best_provider === item.key;

    return `
      <div class="gt-provider-card ${best ? "best" : ""}">
        <div class="gt-provider-heading">
          <div>
            <span class="gt-provider-key">Provider ${item.key}</span>
            <strong>${label}</strong>
          </div>
          ${best ? '<span class="gt-best-badge">BEST CER</span>' : ""}
        </div>
        ${model ? `<div class="gt-model-id">${model}</div>` : ""}
        <div class="gt-score-row">
          <div class="gt-score">
            <span>Strict CER</span>
            <strong>${strictCer}</strong>
          </div>
          <div class="gt-score secondary">
            <span>Content CER</span>
            <strong>${contentCer}</strong>
          </div>
        </div>
        <div class="gt-wer">WER ${wer} · spacing/tokenization sensitive</div>
      </div>
    `;
  }).join("");

  host.innerHTML = `
    <div class="gt-provenance">
      <div>
        <span>Reference text</span>
        <strong>${gt.source_text_status || "PROFESSOR SUPPLIED"}</strong>
      </div>
      <div>
        <span>11-line mapping</span>
        <strong>${gt.physical_line_alignment || "PROJECT VISUAL REVIEWED"}</strong>
      </div>
      <div>
        <span>Translation GT</span>
        <strong>${gt.translation_ground_truth_available ? "AVAILABLE" : "NOT AVAILABLE"}</strong>
      </div>
    </div>

    <div class="gt-provider-grid">${cards}</div>

    <div class="gt-interpretation">
      <strong>Research interpretation:</strong>
      lower CER is better. Provider C is strongest on this Golden manuscript.
      CER/WER measure Stage-5 transcription accuracy; they do not replace H
      (HTR readiness) or T (Stage-6 semantic trust), and they do not validate
      the Stage-6 translation.
    </div>
  `;
}

function renderTranslationText(caseItem) {
  const host = document.getElementById("translationOutcome");
  const note = document.getElementById("translationNote");

  if (state.selectedTranslationLang === "hindi") {
    host.textContent =
      caseItem.hindi_translation_candidate ||
      "Hindi translation candidate is not available.";
    note.textContent =
      "Hindi rendering of the current Stage-6 interpretation; not a separate ground-truth source.";
  } else {
    host.textContent = caseItem.page_translation_candidate || "";
    note.textContent =
      "English translation candidate generated from the current Stage-6 normalized interpretation.";
  }
}

function renderScholarWorkspace(caseItem) {
  const workspace = document.getElementById("scholarWorkspace");
  const noSafe = document.getElementById("noSafeOutput");

  if (caseItem.id !== "golden_success") {
    workspace.classList.add("hidden");
    noSafe.classList.remove("hidden");
    return;
  }

  workspace.classList.remove("hidden");
  noSafe.classList.add("hidden");

  document.getElementById("devanagariOutcome").textContent =
    caseItem.normalized_page_candidate || "Normalized Devanagari candidate unavailable.";

  document.getElementById("plainSummary").textContent =
    caseItem.plain_english_summary || "";

  const unresolved = caseItem.unresolved_items || [];
  document.getElementById("unresolvedBox").innerHTML = unresolved.length
    ? `<strong>Unresolved / scholar-review items</strong><ul>${unresolved.map(x => `<li>${x}</li>`).join("")}</ul>`
    : "";

  wireTranslationTabs(caseItem);
  renderTranslationText(caseItem);
}

function renderGlossary() {
  const host = document.getElementById("glossaryGrid");
  const glossary = state.catalog.glossary || fallbackGlossary;
  host.innerHTML = "";

  glossary.forEach(item => {
    const card = document.createElement("article");
    card.className = "glossary-card";
    card.innerHTML = `
      <h4>${item.term}</h4>
      <span class="term-expansion">${item.expansion || ""}</span>
      <p>${item.meaning}</p>
      ${item.direction ? `<span class="direction">${item.direction}</span>` : ""}
    `;
    host.appendChild(card);
  });
}

async function renderSelectedCase() {
  const caseItem = state.catalog.cases.find(x => x.id === state.selectedCaseId);
  if (!caseItem) return;

  const [uiState, summary] = await Promise.all([
    getJson(`/api/showcase/cases/${caseItem.id}/ui-state`),
    getJson(`/api/showcase/cases/${caseItem.id}/summary`),
  ]);

  document.getElementById("detailTitle").textContent = caseItem.display_name;
  document.getElementById("executionState").textContent =
    caseItem.execution_state || "VALIDATED PRIOR EXECUTION";

  renderMetricGrid(caseItem);
  renderDecisionPanel(caseItem, uiState);
  renderEvidencePanel(caseItem, uiState);
  renderScientificPanel(caseItem, uiState);
  renderProfessorGroundTruth(caseItem, uiState, summary);
  renderScholarWorkspace(caseItem);
}

function renderGuardrails() {
  const host = document.getElementById("guardrailGrid");
  host.innerHTML = "";

  state.catalog.scientific_guardrails.forEach(item => {
    const card = document.createElement("div");
    card.className = "guardrail-card";
    card.textContent = item;
    host.appendChild(card);
  });
}

async function boot() {
  const status = document.getElementById("apiStatus");

  try {
    const health = await getJson("/api/showcase/health");
    state.catalog = await getJson("/api/showcase/catalog");

    status.textContent = `API READY · ${health.case_count} CASES`;
    status.classList.add("ok");

    renderModes();
    renderStages();
    renderOrchestrator();
    renderCaseCards();
    renderGlossary();
    renderGuardrails();
    await renderSelectedCase();
  } catch (err) {
    status.textContent = "API ERROR";
    console.error(err);
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div style="background:#f4e2df;color:#8a3e37;padding:12px 18px;font-family:sans-serif;">
        Showcase dashboard failed to load: ${String(err)}
      </div>`
    );
  }
}

boot();
