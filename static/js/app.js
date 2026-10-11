/* Moodalyx — frontend logic. Talks to main.py: GET /health, POST /predict. */
(function () {
  "use strict";
  const { $, $$, el, api, fmt, renderBars, renderMatrix, plotXY, legend, setBusy, setStatus, callout, toast } = window.RUI;

  // Label order matches main.emotion_labels / the model's output layer.
  const LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"];
  const COLORS = { sadness: "#3f6fb5", joy: "#e08a3c", love: "#c8578f", anger: "#c4403a", fear: "#7a5aa6", surprise: "#3f9b7a" };
  const MAX_TOKENS = 50;
  const MAX_BATCH = 20;
  // Sentences used in final_clean.ipynb cell 39 (the first was predicted as "fear" there).
  const EXAMPLES = [
    "I can't believe how happy I am right now, this is amazing!",
    "I feel so alone and hopeless today.",
    "I am furious that they cancelled the trip at the last minute.",
    "I feel terrified when walking down dark alleyways alone.",
    "I was shocked and completely surprised by the unexpected gift!",
  ];

  const state = { ready: false, mode: "single", history: [] };

  /* JS mirror of main.preprocess_text, used only to *display* what the model sees. */
  function clean(text) {
    return text.toLowerCase().replace(/'/g, "").replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
  }

  /* ------------------------------------------------------------------ */
  /* Health                                                             */
  /* ------------------------------------------------------------------ */
  async function checkHealth() {
    try {
      const h = await api("/health", { timeout: 6000 });
      state.ready = !!h.model_loaded;
      if (state.ready) setStatus($("#api-status"), "ok", "Model loaded");
      else { setStatus($("#api-status"), "warn", "Model loading…"); setTimeout(checkHealth, 3000); }
    } catch {
      state.ready = false;
      setStatus($("#api-status"), "err", "API unreachable");
      setTimeout(checkHealth, 5000);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Input                                                              */
  /* ------------------------------------------------------------------ */
  function updateCount() {
    const v = $("#text-input").value;
    if (state.mode === "batch") {
      const lines = v.split("\n").filter((l) => l.trim()).length;
      $("#text-count").textContent = `${lines} / ${MAX_BATCH} lines`;
    } else {
      const words = v.trim() ? v.trim().split(/\s+/).length : 0;
      $("#text-count").textContent = `${fmt.num(v.length)} characters · ${words} word${words === 1 ? "" : "s"}`;
    }
  }

  function validateSingle(text) {
    if (!text.trim()) return "Enter some text to analyse.";
    if (text.length > 2000) return "Text must be 2,000 characters or fewer.";
    if (!clean(text)) return "No letters or digits remain after cleaning (the model ignores punctuation and emoji), so there is nothing to classify.";
    return "";
  }

  function setError(msg) {
    $("#text-error").textContent = msg;
    $("#text-input").setAttribute("aria-invalid", msg ? "true" : "false");
  }

  async function predict(text) {
    return api("/predict", { method: "POST", body: { text }, timeout: 30000 });
  }

  async function analyse() {
    const raw = $("#text-input").value;
    if (state.mode === "batch") return analyseBatch(raw);
    const msg = validateSingle(raw);
    setError(msg);
    if (msg) { $("#text-input").focus(); return; }
    const btn = $("#analyse-btn");
    setBusy(btn, true, "Analysing…");
    try {
      const res = await predict(raw);
      renderResult(res);
      addHistory(res);
    } catch (err) {
      const title = err.status === 503 ? "Model is still loading" : err.status === 422 ? "Input rejected by the API" : "Prediction failed";
      $("#result-body").replaceChildren(callout("err", title, [err.message]));
      $("#result-badge").textContent = "error";
    } finally {
      setBusy(btn, false);
    }
  }

  async function analyseBatch(raw) {
    const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
    let msg = "";
    if (!lines.length) msg = "Enter one sentence per line.";
    else if (lines.length > MAX_BATCH) msg = `At most ${MAX_BATCH} lines at a time (you entered ${lines.length}).`;
    else {
      const bad = lines.findIndex((l) => validateSingle(l));
      if (bad >= 0) msg = `Line ${bad + 1}: ${validateSingle(lines[bad])}`;
    }
    setError(msg);
    if (msg) return;
    const btn = $("#analyse-btn");
    setBusy(btn, true, `Analysing 0 / ${lines.length}…`);
    let done = 0, failed = 0;
    const results = new Array(lines.length);
    const worker = async (queue) => {
      while (queue.length) {
        const i = queue.shift();
        try { results[i] = await predict(lines[i]); } catch (err) { results[i] = { error: err.message, text: lines[i] }; failed++; }
        done++;
        $(".btn-label", btn).textContent = `Analysing ${done} / ${lines.length}…`;
      }
    };
    const queue = lines.map((_, i) => i);
    await Promise.all([worker(queue), worker(queue), worker(queue)]);
    setBusy(btn, false);
    for (let i = results.length - 1; i >= 0; i--) if (!results[i].error) addHistory(results[i]);
    const ok = results.filter((r) => !r.error);
    if (ok.length) renderResult(ok[ok.length - 1], { batchSummary: { total: lines.length, failed, results: ok } });
    if (failed) toast(`${failed} line${failed === 1 ? "" : "s"} failed: ${results.find((r) => r.error).error}`, "err", 6000);
    $("#history-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ------------------------------------------------------------------ */
  /* Results                                                            */
  /* ------------------------------------------------------------------ */
  function renderResult(res, { batchSummary } = {}) {
    const probs = LABELS.map((l) => ({ label: l, p: res.all_probabilites[l] ?? 0 }));
    const sorted = [...probs].sort((a, b) => b.p - a.p);
    const top = res.predicted_emotion;
    const margin = sorted[0].p - sorted[1].p;
    const cleaned = clean(res.text);
    const tokens = cleaned ? cleaned.split(" ") : [];
    const bars = el("div");
    renderBars(bars, sorted.map((x) => ({ label: x.label, value: x.p, color: COLORS[x.label], top: x.label === top })), { ariaLabel: "Probability for each emotion label", valueFormat: (v) => (v < 0.001 ? "<0.1%" : fmt.pct(v, 1)) });

    const cleanedNode = el("div", { class: "cleaned" });
    if (tokens.length > MAX_TOKENS) {
      cleanedNode.append(tokens.slice(0, MAX_TOKENS).join(" ") + " ", el("mark", { title: "Truncated: not seen by the model", text: tokens.slice(MAX_TOKENS).join(" ") }));
    } else cleanedNode.textContent = cleaned;

    const uncertainty = margin < 0.2
      ? callout("warn", "Low margin", [`The top two labels are close (${sorted[0].label} ${fmt.pct(sorted[0].p)} vs ${sorted[1].label} ${fmt.pct(sorted[1].p)}). Treat this prediction as uncertain.`])
      : null;

    $("#result-badge").textContent = batchSummary ? `last of ${batchSummary.total}` : "live";
    $("#result-body").replaceChildren(el("div", { class: "stack" }, [
      batchSummary ? callout("info", `Analysed ${batchSummary.results.length} of ${batchSummary.total} lines`, ["The most recent result is shown here; all results are in the session table below."]) : null,
      el("div", { class: "pred-hero" }, [
        el("div", { class: "pred-swatch", style: { background: COLORS[top] }, "aria-hidden": "true" }),
        el("div", {}, [
          el("p", { class: "subhead", text: "Predicted label" }),
          el("div", { class: "pred-word", text: top }),
          el("p", { class: "pred-sub", text: `p = ${res.confidence.toFixed(3)} · margin over “${sorted[1].label}”: ${(margin * 100).toFixed(1)} pts` }),
        ]),
      ]),
      uncertainty,
      el("p", { class: "subhead", text: "Probability distribution (softmax)" }),
      bars,
      el("p", { class: "subhead", text: `What the model reads (${Math.min(tokens.length, MAX_TOKENS)} of ${tokens.length} tokens)` }),
      cleanedNode,
      el("p", { class: "hint", text: tokens.length > MAX_TOKENS ? "Highlighted words exceed the 50-token limit and are cut off before inference." : "Cleaning replicated in the browser for display; the API applies the same rules server-side. Words outside the 10,000-word vocabulary become <unk>." }),
    ]));
  }

  function addHistory(res) {
    state.history.unshift({ ...res, at: Date.now() });
    state.history = state.history.slice(0, 50);
    renderHistory();
  }

  function renderHistory() {
    const panel = $("#history-panel");
    panel.hidden = !state.history.length;
    const rows = state.history.map((h) => el("tr", {}, [
      el("td", { class: "text-cell" }, [el("span", { text: h.text, title: h.text })]),
      el("td", {}, [labelCell(h.predicted_emotion)]),
      el("td", { class: "num", text: h.confidence.toFixed(3) }),
      el("td", {}, [el("span", { class: "dist", role: "img", "aria-label": LABELS.map((l) => `${l} ${fmt.pct(h.all_probabilites[l])}`).join(", ") },
        LABELS.map((l) => el("span", { style: { width: `${h.all_probabilites[l] * 100}%`, background: COLORS[l] }, title: `${l}: ${fmt.pct(h.all_probabilites[l])}` })))]),
      el("td", {}, [el("button", { class: "btn btn-sm btn-ghost", type: "button", text: "Show", onclick: () => { setMode("single"); $("#text-input").value = h.text; updateCount(); renderResult(h); $("#analyse").scrollIntoView({ behavior: "smooth" }); } })]),
    ]));
    $("#history-table").replaceChildren(
      el("thead", {}, [el("tr", {}, [el("th", { text: "Text" }), el("th", { text: "Label" }), el("th", { class: "num", text: "p" }), el("th", { text: "Distribution" }), el("th", { html: '<span class="sr-only">Actions</span>' })])]),
      el("tbody", {}, rows)
    );
  }

  function labelCell(l) {
    return el("span", { class: "label-cell" }, [el("span", { class: "bar-swatch", style: { background: COLORS[l] }, "aria-hidden": "true" }), l]);
  }

  /* ------------------------------------------------------------------ */
  /* Evaluation                                                         */
  /* ------------------------------------------------------------------ */
  function stat(label, value, sub, prov) {
    return el("div", { class: "stat" }, [el("div", { class: "stat-label", text: label }), el("div", { class: "stat-value", text: value }), sub ? el("div", { class: "stat-sub", text: sub }) : null, prov ? el("div", { class: "provenance", style: { marginTop: "8px" }, text: prov }) : null]);
  }

  function topConfusions(cm, labels, k = 3) {
    const cells = [];
    cm.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v) cells.push({ i, j, v }); }));
    return cells.sort((a, b) => b.v - a.v).slice(0, k).map((c) => `${labels[c.i]} → ${labels[c.j]} (${c.v})`).join(", ");
  }

  async function loadEvaluation() {
    const root = $("#eval-root");
    let e;
    try { e = await api("/static/data/evaluation.json", { timeout: 10000 }); }
    catch (err) {
      root.replaceChildren(callout("warn", "Evaluation results not found", ["static/data/evaluation.json is missing. Generate it with: python evaluation/evaluate_artifact.py", err.message]));
      return;
    }
    const m = e.api_path;
    const pretty = (s) => s.charAt(0).toUpperCase() + s.slice(1);

    const matrixBox = el("div");
    let normalize = false;
    const draw = () => renderMatrix(matrixBox, m.confusion_matrix, e.labels.map(pretty), { normalize, color: "#0d5c56", caption: "Test-set confusion matrix" });
    const toggle = el("div", { class: "segmented", role: "group", "aria-label": "Confusion matrix values" }, [
      el("button", { type: "button", "aria-pressed": "true", text: "Counts" }), el("button", { type: "button", "aria-pressed": "false", text: "Row %" }),
    ]);
    $$("button", toggle).forEach((b, i) => b.addEventListener("click", () => { normalize = i === 1; $$("button", toggle).forEach((x, j) => x.setAttribute("aria-pressed", String(i === j))); draw(); }));
    draw();

    const perClass = el("table", { class: "table" }, [
      el("thead", {}, [el("tr", {}, ["Label", "Precision", "Recall", "F1", "Support"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))]),
      el("tbody", {}, m.per_class.map((c) => el("tr", {}, [
        el("td", {}, [labelCell(c.label)]), el("td", { class: "num", text: c.precision.toFixed(3) }), el("td", { class: "num", text: c.recall.toFixed(3) }),
        el("td", { class: "num", text: c.f1.toFixed(3) }), el("td", { class: "num", text: c.support }),
      ]))),
    ]);

    const relBox = el("div");
    const bins = e.calibration.bins;
    plotXY(relBox, [
      { name: "ideal", color: "#b5b2a8", points: [[0, 0], [1, 1]], dashed: true, width: 1.2 },
      { name: "model", color: "#0d5c56", points: bins.map((b) => [b.mean_confidence, b.accuracy]), markers: true },
    ], { xLabel: "Mean predicted confidence (bin)", yLabel: "Observed accuracy", xDomain: [0, 1], yDomain: [0, 1], height: 280, ariaLabel: "Reliability diagram" });

    const recorded = [
      { name: "Simple RNN (2 layers)", acc: 0.2645, note: "stopped after 8 epochs" },
      { name: "LSTM (2 layers)", acc: 0.3395, note: "stopped after 3 epochs" },
      { name: "GRU (2 layers)", acc: 0.2890, note: "stopped after 3 epochs" },
      { name: "Bidirectional GRU (deployed)", acc: 0.9210, note: "best epoch 5 of 8" },
    ];
    const recBars = el("div");
    renderBars(recBars, recorded.map((r) => ({ label: r.name, value: r.acc, color: r.name.includes("deployed") ? "#0d5c56" : "#9c9a92" })), { valueFormat: (v) => v.toFixed(3), ariaLabel: "Recorded test accuracy by architecture" });

    const maxCount = Math.max(...e.class_counts);
    const distBars = el("div");
    renderBars(distBars, e.labels.map((l, i) => ({ label: l, value: e.class_counts[i] / maxCount, color: COLORS[l] })), { valueFormat: (v) => fmt.num(Math.round(v * maxCount)), ariaLabel: "Test sentences per label" });

    const errors = el("ul", { class: "error-list" }, e.high_confidence_errors.slice(0, 8).map((x) => el("li", {}, [
      el("q", { text: x.text }),
      el("span", { class: "err-meta" }, ["true ", el("strong", { text: x.true }), " · predicted ", el("strong", { text: x.predicted }), ` · p = ${x.confidence.toFixed(3)}`]),
      el("button", { class: "btn btn-sm", type: "button", text: "Run live", onclick: () => { setMode("single"); $("#text-input").value = x.text; updateCount(); analyse(); $("#analyse").scrollIntoView({ behavior: "smooth" }); } }),
    ])));

    $("#param-count").textContent = `${fmt.num(e.model.trainable_params)} trainable`;

    root.replaceChildren(
      el("div", { class: "grid-4" }, [
        stat("Test accuracy", m.accuracy.toFixed(3), `${Math.round(m.accuracy * e.n)} / ${e.n} sentences`, "re-evaluated"),
        stat("Macro-F1", m.macro_f1.toFixed(3), "Unweighted mean over 6 labels", "re-evaluated"),
        stat("Weighted F1", m.weighted_f1.toFixed(3), "Weighted by label frequency", "re-evaluated"),
        stat("Expected calibration error", e.calibration.ece.toFixed(3), "10 confidence bins (lower is better)", "re-evaluated"),
      ]),
      el("div", { class: "eval-grid" }, [
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Confusion matrix (test)" }), toggle]),
          el("div", { class: "panel-body stack-sm" }, [matrixBox, el("p", { class: "hint", text: `Rows: true label. Columns: prediction. Largest confusions: ${topConfusions(m.confusion_matrix, e.labels)}.` })]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Per-label metrics (test)" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "table-wrap" }, [perClass]),
          el("div", { class: "panel-body stack-sm" }, [el("p", { class: "subhead", text: "Test sentences per label" }), distBars]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Reliability diagram" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body stack-sm" }, [relBox, legend([{ label: "Model", color: "#0d5c56" }, { label: "Perfect calibration", color: "#b5b2a8" }]),
            el("p", { class: "hint", text: `Each point is a confidence bin; points below the diagonal mean the model is over-confident there. Bins: ${bins.map((b) => `${b.lo.toFixed(1)}–${b.hi.toFixed(1)} (n=${b.n})`).join(", ")}.` })]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Architectures compared in the notebook" }), el("span", { class: "provenance", text: "notebook cells 21–33" })]),
          el("div", { class: "panel-body stack-sm" }, [recBars, el("p", { class: "hint", text: "Recorded test accuracy. The three unidirectional baselines stopped after 3–8 epochs at near-chance accuracy (the majority class is ≈35%), which points to failed optimisation in those runs rather than an inherent limit of RNNs, LSTMs or GRUs. Treat this as a record of the project's experiments, not a controlled architecture comparison." })]),
        ]),
        el("div", { class: "panel span-all" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Most confident errors (test split)" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body stack-sm" }, [
            el("p", { class: "hint", text: "Test sentences the model got wrong with the highest confidence. Some dataset labels are themselves debatable (labels were derived from hashtags), which is a known property of this dataset." }),
            errors,
          ]),
        ]),
      ]),
      el("p", { class: "hint", style: { marginTop: "16px" }, text: `Generated ${new Date(e.generated_at).toLocaleString()} by ${e.script} on ${e.dataset} (n = ${e.n}). Scoring the raw notebook-style text instead of the API-cleaned text gives accuracy ${e.notebook_path.accuracy.toFixed(3)}; the two paths agree on ${(e.agreement_api_vs_notebook * 100).toFixed(1)}% of sentences. Mean out-of-vocabulary rate: ${(e.tokenizer.mean_oov_rate_test * 100).toFixed(1)}% of words.` })
    );
  }

  /* ------------------------------------------------------------------ */
  /* Wiring                                                             */
  /* ------------------------------------------------------------------ */
  function setMode(mode) {
    state.mode = mode;
    $$("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    $("#text-label").textContent = mode === "batch" ? "Sentences (one per line)" : "Text";
    $("#text-input").placeholder = mode === "batch" ? "One sentence per line, up to 20 lines" : "e.g. i feel hopeful about the results of the experiment";
    $("#text-hint").textContent = mode === "batch" ? "Each line is sent as a separate request. Results are added to the session table." : "Up to 2,000 characters. Only the first 50 tokens are used by the model. Ctrl/⌘ + Enter to analyse.";
    $(".btn-label", $("#analyse-btn")).textContent = mode === "batch" ? "Analyse all lines" : "Analyse";
    setError("");
    updateCount();
  }

  $("#examples").replaceChildren(...EXAMPLES.map((t) => el("button", { type: "button", class: "chip example", text: t, onclick: () => {
    if (state.mode === "batch") { const v = $("#text-input").value.trim(); $("#text-input").value = v ? `${v}\n${t}` : t; updateCount(); }
    else { $("#text-input").value = t; updateCount(); analyse(); }
  } })));
  $("#text-form").addEventListener("submit", (e) => { e.preventDefault(); analyse(); });
  $("#text-input").addEventListener("input", () => { updateCount(); if ($("#text-error").textContent) setError(""); });
  $("#text-input").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); analyse(); } });
  $("#clear-btn").addEventListener("click", () => { $("#text-input").value = ""; updateCount(); setError(""); $("#text-input").focus(); });
  $$("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#history-clear").addEventListener("click", () => { state.history = []; renderHistory(); });

  updateCount();
  checkHealth();
  loadEvaluation();
})();
