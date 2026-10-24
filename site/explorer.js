const manifestURL = new URL("assets/explorer/manifest.json", document.baseURI);
const deltaManifestURL = new URL("assets/explorer/delta-manifest.json", document.baseURI);
const root = document.getElementById("explorer");
const byId = (id) => document.getElementById(`explorer-${id}`);
const measured = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const number = (value, digits = 0) => value.toLocaleString("en-GB", { maximumFractionDigits: digits });

export function scoreLabel(metric) {
  if (metric?.status === "not_applicable") return "N/A";
  return measured(metric?.value) && metric.value <= 100 ? number(metric.value, 1) : "Not measured";
}

export function costLabel(value) {
  if (!measured(value)) return "Not measured";
  if (value > 0 && value < 0.0001) return "< $0.0001";
  return `$${value.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;
}

export function timeLabel(value) {
  if (!measured(value)) return "Not measured";
  if (value < 60) return `${number(value, 1)} s`;
  if (value < 3600) return `${number(value / 60, 1)} min`;
  return `${number(value / 3600, 2)} h`;
}

function safeURL(value) {
  if (typeof value !== "string" || !value) throw new Error("Missing asset URL");
  const url = new URL(value, document.baseURI);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("Unsupported asset URL");
  return url.href;
}

function validateManifest(data) {
  if (data?.schema_version !== 1 || !Array.isArray(data.objects) || !data.objects.length) {
    throw new Error("Comparison data is unavailable");
  }
  const identities = new Set();
  for (const object of data.objects) {
    if (typeof object.id !== "string" || identities.has(object.id)
        || typeof object.label !== "string" || !Array.isArray(object.variants) || !object.variants.length) {
      throw new Error("Invalid comparison object");
    }
    identities.add(object.id);
    safeURL(object.reference?.src);
    safeURL(object.reference?.source_url);
    const variants = new Set();
    for (const variant of object.variants) {
      if (typeof variant.id !== "string" || variants.has(variant.id)
          || typeof variant.label !== "string" || typeof variant.model !== "string"
          || !Number.isInteger(variant.repetition) || variant.repetition < 0) {
        throw new Error("Invalid comparison reconstruction");
      }
      variants.add(variant.id);
      if (variant.src !== null) safeURL(variant.src);
      safeURL(variant.source_url);
    }
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function metric(value, detail) {
  const cell = element("td", "explorer-metric");
  cell.append(element("span", "metric-value explorer-metric-value", value));
  if (detail) cell.append(element("span", "metric-coverage explorer-metric-detail", detail));
  return cell;
}

function renderReadout(variant) {
  const scores = variant.scores || {};
  byId("scores").replaceChildren(...["formal", "visual", "kinematic"].map((name) => {
    const part = scores[name];
    return metric(scoreLabel(part));
  }));
  const usage = variant.usage || {};
  const totalCost = measured(usage.total_cost_usd);
  const inferenceCost = measured(usage.inference_cost_usd);
  const cost = totalCost ? usage.total_cost_usd : usage.inference_cost_usd;
  const inferenceCostKind = variant.inference_cost_kind || usage.inference_cost_kind;
  const pricingDerived = inferenceCost && typeof inferenceCostKind === "string"
    && /pricing[- ]derived/i.test(inferenceCostKind);
  const costDetail = totalCost
    ? `Inference ${costLabel(usage.inference_cost_usd)}${pricingDerived ? " (pricing-derived)" : ""} · compute ${costLabel(usage.compute_cost_usd)}`
    : pricingDerived ? "Pricing-derived" : "";
  const tokenPart = (value) => measured(value) ? number(value) : "not measured";
  byId("cost-label").textContent = `${totalCost ? "Total cost" : inferenceCost ? "Inference cost" : "Cost"} (USD)`;
  byId("usage").replaceChildren(
    metric(costLabel(cost), costDetail),
    metric(measured(usage.total_tokens) ? number(usage.total_tokens) : "Not measured",
      `Input ${tokenPart(usage.input_tokens)} · output ${tokenPart(usage.output_tokens)}`),
    metric(timeLabel(usage.elapsed_seconds), `Model requests: ${timeLabel(usage.model_request_seconds)}`),
  );
  byId("trial-label").textContent = `Trial ${variant.repetition + 1} · fidelity scores / 100`;
  const provenance = [];
  if (Number.isInteger(variant.max_output_tokens) && variant.max_output_tokens > 0) {
    provenance.push(`Output limit: ${number(variant.max_output_tokens)} tokens/request`);
  } else if (variant.output_limit === "provider-native") {
    provenance.push("Output limit: provider-native");
  }
  if (variant.orichalcum_pin) provenance.push(`Orichalcum ${variant.orichalcum_pin}`);
  byId("provenance").textContent = provenance.join(" · ");
  byId("candidate-title").textContent = variant.label;
  byId("candidate-title").title = variant.model;
  byId("result-source").href = safeURL(variant.source_url);
  byId("readout").hidden = false;
}

const state = { data: null, deltaData: null, object: null, variant: null, viewer: null, mode: "swipe", pair: null };

function distanceLabel(metres) {
  if (!measured(metres)) return "—";
  return metres < 1 ? `${number(metres * 1000, metres < .001 ? 3 : 1)} mm` : `${number(metres, 2)} m`;
}

function showMessage(message) {
  byId("preview-message").textContent = message;
  byId("preview-message").hidden = !message;
}

function renderViewerState(view) {
  const mode = view.mode || state.mode;
  state.mode = mode;
  byId("stage").dataset.mode = mode;
  byId("stage").setAttribute("aria-busy", String(Boolean(view.loading || view.deltaLoading)));
  byId("mode-swipe").disabled = !view.referenceReady || Boolean(view.errors?.webgl);
  byId("mode-delta").disabled = !view.referenceReady || !view.candidateReady || !state.pair?.src || Boolean(view.errors?.webgl);
  for (const name of ["swipe", "delta"]) {
    const active = mode === name;
    byId(`mode-${name}`).setAttribute("aria-pressed", String(active));
    byId(`mode-${name}`).classList.toggle("is-dark", active);
  }
  byId("reset").disabled = !view.referenceReady || Boolean(view.errors?.webgl);
  const swipeReady = mode === "swipe" && view.referenceReady && view.candidateReady && !view.errors?.webgl;
  byId("swipe").hidden = !swipeReady;
  byId("swipe").disabled = !swipeReady;
  byId("divider").hidden = !swipeReady;
  byId("delta-controls").hidden = mode !== "delta";
  byId("threshold").disabled = !view.deltaReady;
  const maximum = view.colourMaxMetres ?? (state.pair?.reference_diagonal_m || 0) * .05;
  byId("delta-max").textContent = `≥ ${distanceLabel(maximum)}`;
  byId("point-count").textContent = view.deltaReady
    ? `${number(view.visiblePointCount || 0)} / ${number(view.pointCount || 0)} points`
    : view.loading || view.deltaLoading ? "Loading points…" : "Points unavailable";
  let message = "";
  if (view.errors?.webgl) message = "3D viewer unavailable";
  else if (view.loading || view.deltaLoading) message = mode === "delta" ? "Loading mesh delta…" : "Loading geometry…";
  else if (!view.referenceReady) message = "3D preview unavailable";
  else if (!view.candidateReady) message = "Reconstruction unavailable";
  else if (mode === "delta" && !view.deltaReady) message = "Mesh delta unavailable";
  else if (mode === "delta" && view.visiblePointCount === 0) message = "No points above this threshold";
  showMessage(message);
  byId("status").textContent = message || `${state.object?.label || "Object"}: ${mode === "delta" ? "mesh delta cloud" : "swipe comparison"} ready.`;
}

function updateSplit() {
  const split = Number(byId("swipe").value);
  byId("stage").style.setProperty("--split", `${split}%`);
  byId("swipe").setAttribute("aria-valuetext", `${split}% reference, ${100 - split}% reconstruction`);
  state.viewer?.setSplit(split / 100);
}

function updateThreshold() {
  const metres = Number(byId("threshold").value) / 100 * (state.pair?.reference_diagonal_m || 0);
  byId("threshold-value").textContent = distanceLabel(metres);
  byId("threshold").setAttribute("aria-valuetext", distanceLabel(metres));
  state.viewer?.setThreshold(metres);
}

function selectVariant(id) {
  const variant = state.object.variants.find((item) => item.id === id) || state.object.variants[0];
  state.variant = variant;
  state.pair = state.deltaData?.pairs?.[`${state.object.id}/${variant.id}`] || null;
  if (state.pair?.src && (state.pair.source_reference_sha256 !== state.object.reference.sha256
      || state.pair.source_candidate_sha256 !== variant.sha256)) state.pair = null;
  byId("variant").value = variant.id;
  renderReadout(variant);
  renderAttribution();
  if (!state.viewer) {
    showMessage("3D viewer unavailable");
    byId("stage").setAttribute("aria-busy", "false");
    return;
  }
  if (!variant.src || !state.pair?.src) state.mode = "swipe";
  state.viewer.setMode(state.mode);
  byId("threshold").value = "0";
  updateThreshold();
  state.viewer.setPair({
    referenceUrl: safeURL(state.object.reference.src),
    candidateUrl: variant.src ? safeURL(variant.src) : null,
    deltaUrl: state.pair?.src ? safeURL(state.pair.src) : null,
    deltaMeta: state.pair || {},
  }).catch(() => {
    // The renderer reports failures per resource while retaining valid geometry.
    renderViewerState(state.viewer.getState());
  });
}

function selectObject(id) {
  const previousRun = state.variant?.run_id;
  const previousId = state.variant?.id;
  const object = state.data.objects.find((item) => item.id === id) || state.data.objects[0];
  state.object = object;
  byId("object").value = object.id;
  byId("reference-title").textContent = object.label;
  byId("reference-source").href = safeURL(object.reference.source_url);
  const options = object.variants.map((variant) => {
    const label = [variant.label, variant.harness || "Harness unrecorded", `${variant.reasoning || "default"} reasoning`];
    if (variant.orichalcum_pin) label.push(`Orichalcum ${variant.orichalcum_pin.slice(0, 12)}`);
    return new Option(label.join(" · "), variant.id);
  });
  byId("variant").replaceChildren(...options);
  const next = object.variants.find((variant) => previousRun && variant.run_id === previousRun)
    || object.variants.find((variant) => variant.id === previousId) || object.variants[0];
  selectVariant(next.id);
}

async function fetchJSON(url) {
  const response = await fetch(url, { credentials: "omit" });
  if (!response.ok) throw new Error("Comparison data unavailable");
  return response.json();
}

function renderAttribution() {
  const data = state.data;
  const alignment = state.pair?.alignment_status === "authored" ? " · authored alignment" : "";
  byId("selection").textContent = `Neutral materials · shared metric scale${alignment}`;
  if (data.reference_license?.label && data.reference_license?.url) {
    byId("selection").append(document.createTextNode(" · "));
    const link = element("a", "", "NVIDIA / CC BY 4.0");
    link.href = safeURL(data.reference_license.url);
    byId("selection").append(link);
  }
}

async function initialise() {
  root.dataset.state = "loading";
  byId("status").textContent = "Loading comparison…";
  try {
    const [data, delta, renderer] = await Promise.all([
      fetchJSON(manifestURL).then(validateManifest),
      fetchJSON(deltaManifestURL).catch(() => null),
      import(new URL("comparison-viewer.js?v=20261024", document.baseURI).href).catch(() => null),
    ]);
    state.data = data;
    state.deltaData = delta?.schema_version === 1 ? delta : null;
    byId("object").replaceChildren(...data.objects.map((object) => new Option(object.label, object.id)));
    byId("object").disabled = false;
    byId("variant").disabled = false;
    byId("object").addEventListener("change", (event) => selectObject(event.target.value));
    byId("variant").addEventListener("change", (event) => selectVariant(event.target.value));
    byId("swipe").addEventListener("input", updateSplit);
    byId("threshold").addEventListener("input", updateThreshold);
    byId("reset").addEventListener("click", () => {
      state.viewer?.reset();
      byId("swipe").value = "50";
      updateSplit();
    });
    for (const mode of ["swipe", "delta"]) {
      byId(`mode-${mode}`).addEventListener("click", () => {
        state.mode = mode;
        state.viewer?.setMode(mode);
      });
    }
    if (renderer) {
      try {
        state.viewer = new renderer.ComparisonViewer(byId("canvas"), { onState: renderViewerState });
        byId("stage").comparisonViewer = state.viewer;
      } catch {
        showMessage("3D viewer unavailable");
      }
    }
    root.dataset.state = "ready";
    selectObject(data.objects[0].id);
  } catch {
    root.dataset.state = "error";
    byId("status").textContent = "3D viewer unavailable. Results are available on Hugging Face.";
    byId("stage").setAttribute("aria-busy", "false");
    showMessage("Preview unavailable");
    byId("object").replaceChildren(new Option("Preview unavailable", ""));
    byId("variant").replaceChildren(new Option("View published results below", ""));
  }
}

if (root && new URLSearchParams(location.search).get("snapshot") !== "1") {
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        initialise();
      }
    }, { rootMargin: "350px" });
    observer.observe(root);
  } else {
    initialise();
  }
}
