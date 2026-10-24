const DEFAULT_ORBIT = "35deg 70deg 105%";
const manifestURL = new URL("assets/explorer/manifest.json", document.baseURI);
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

function metric(title, value, detail, score) {
  const group = element("div", "explorer-metric");
  group.append(element("dt", "", title), element("dd", "explorer-metric-value", value));
  if (score !== undefined) {
    const bar = element("dd", "explorer-score-track");
    bar.setAttribute("aria-hidden", "true");
    const fill = element("span");
    fill.style.width = measured(score) && score <= 100 ? `${score}%` : "0%";
    bar.append(fill);
    group.append(bar);
  }
  if (detail) group.append(element("dd", "explorer-metric-detail", detail));
  return group;
}

function renderReadout(variant) {
  const scores = variant.scores || {};
  byId("scores").replaceChildren(...["formal", "visual", "kinematic"].map((name) => {
    const part = scores[name];
    const title = `${name[0].toUpperCase()}${name.slice(1)} fidelity`;
    const detail = part?.status === "not_applicable" ? "No applicable reference criteria"
      : measured(part?.value) ? "Score out of 100" : "Measurement unavailable";
    return metric(title, scoreLabel(part), detail, part?.status === "not_applicable" ? null : part?.value ?? null);
  }));
  const usage = variant.usage || {};
  const totalCost = measured(usage.total_cost_usd);
  const inferenceCost = measured(usage.inference_cost_usd);
  const cost = totalCost ? usage.total_cost_usd : usage.inference_cost_usd;
  const inferenceCostKind = variant.inference_cost_kind || usage.inference_cost_kind;
  const pricingDerived = inferenceCost && typeof inferenceCostKind === "string"
    && /pricing[- ]derived/i.test(inferenceCostKind);
  const costDetail = (totalCost
    ? `USD · inference ${costLabel(usage.inference_cost_usd)} · compute ${costLabel(usage.compute_cost_usd)}`
    : inferenceCost ? "USD · compute cost not measured" : "USD · cost not recorded")
    + (pricingDerived ? " · inference calculated from pricing" : "");
  const tokenPart = (value) => measured(value) ? number(value) : "not measured";
  byId("usage").replaceChildren(
    metric(totalCost ? "Total cost" : inferenceCost ? "Inference cost" : "Cost", costLabel(cost), costDetail),
    metric("Tokens", measured(usage.total_tokens) ? number(usage.total_tokens) : "Not measured",
      `Input ${tokenPart(usage.input_tokens)} · output ${tokenPart(usage.output_tokens)}`),
    metric("Elapsed time", timeLabel(usage.elapsed_seconds), `Model requests: ${timeLabel(usage.model_request_seconds)}`),
  );
  byId("trial-label").textContent = `Trial ${variant.repetition + 1} · per reconstruction`;
  const provenance = [`${variant.harness || "Harness unrecorded"}`, `${variant.reasoning || "default"} reasoning`];
  if (Number.isInteger(variant.max_output_tokens) && variant.max_output_tokens > 0) {
    provenance.push(`Output allowance: ${number(variant.max_output_tokens)} tokens per request`);
  } else if (variant.output_limit === "provider-native") {
    provenance.push("Provider-native output allowance");
  }
  if (variant.orichalcum_pin) provenance.push(`Orichalcum ${variant.orichalcum_pin}`);
  byId("provenance").textContent = provenance.join(" · ");
  byId("candidate-title").textContent = variant.label;
  byId("candidate-title").title = variant.model;
  byId("candidate-detail").textContent = `${variant.harness || "Harness unrecorded"} · ${variant.reasoning || "default"} reasoning · Trial ${variant.repetition + 1}`;
  byId("result-source").href = safeURL(variant.source_url);
  byId("readout").hidden = false;
}

const state = { data: null, object: null, variant: null, models: {}, ready: {}, failed: {}, syncing: false };

function updateStatus() {
  if (!state.object || !state.variant) return;
  const failed = ["reference", "candidate"].filter((role) => state.failed[role]);
  if (failed.length) {
    byId("status").textContent = "A 3D preview is unavailable. The recorded measurements and source links remain available.";
  } else if (state.ready.reference && state.ready.candidate) {
    byId("status").textContent = `${state.object.label} · ${state.variant.label} · Trial ${state.variant.repetition + 1}. Both previews ready.`;
  } else {
    byId("status").textContent = `Loading ${state.object.label} previews…`;
  }
  byId("reset").disabled = !Object.values(state.ready).some(Boolean);
}

function syncRotation(sourceRole, desiredOrbit = null) {
  const targetRole = sourceRole === "reference" ? "candidate" : "reference";
  if (!byId("link").checked || state.syncing || !state.ready[sourceRole] || !state.ready[targetRole]) return;
  const source = state.models[sourceRole];
  const target = state.models[targetRole];
  const from = desiredOrbit || source.getCameraOrbit();
  const to = target.getCameraOrbit();
  state.syncing = true;
  target.cameraOrbit = `${from.theta}rad ${from.phi}rad ${to.radius}m`;
  target.jumpCameraToGoal();
  state.syncing = false;
}

function setRotationButtons(role, enabled) {
  root.querySelectorAll(`[data-view="${role}"]`).forEach((button) => { button.disabled = !enabled; });
}

function mountModel(role, asset, objectLabel) {
  const stage = byId(`${role}-stage`);
  state.ready[role] = false;
  state.failed[role] = false;
  setRotationButtons(role, false);
  stage.setAttribute("aria-busy", "true");
  if (asset.src === null) {
    state.models[role] = null;
    state.failed[role] = true;
    stage.setAttribute("aria-busy", "false");
    const message = element("div", "explorer-preview-error");
    message.append(element("p", "", "3D preview unavailable"),
      element("p", "", "This recorded reconstruction has no browser geometry. Its measured results remain available below."));
    stage.replaceChildren(message);
    updateStatus();
    return;
  }
  const viewer = document.createElement("model-viewer");
  state.models[role] = viewer;
  viewer.setAttribute("alt", `${objectLabel}, ${role === "reference" ? "NV25 reference" : state.variant.label + " reconstruction"}. Drag to rotate or use the rotation buttons below.`);
  viewer.setAttribute("camera-controls", "");
  viewer.setAttribute("disable-pan", "");
  viewer.setAttribute("touch-action", "pan-y");
  viewer.setAttribute("interaction-prompt", "none");
  viewer.setAttribute("camera-orbit", DEFAULT_ORBIT);
  viewer.setAttribute("shadow-intensity", "0.35");
  viewer.setAttribute("exposure", "0.65");
  viewer.setAttribute("tone-mapping", "neutral");
  viewer.setAttribute("loading", "eager");
  const loading = element("p", "explorer-placeholder", "Loading geometry…");
  loading.setAttribute("aria-hidden", "true");
  viewer.addEventListener("load", () => {
    if (state.models[role] !== viewer) return;
    loading.remove();
    stage.setAttribute("aria-busy", "false");
    state.ready[role] = true;
    setRotationButtons(role, true);
    syncRotation(role === "reference" ? "candidate" : "reference");
    updateStatus();
  });
  viewer.addEventListener("error", () => {
    if (state.models[role] !== viewer) return;
    state.failed[role] = true;
    state.ready[role] = false;
    setRotationButtons(role, false);
    stage.setAttribute("aria-busy", "false");
    const message = element("div", "explorer-preview-error");
    message.append(element("p", "", "3D preview unavailable"), element("p", "", "Open the source link below to inspect the recorded asset."));
    stage.replaceChildren(message);
    updateStatus();
  });
  viewer.addEventListener("camera-change", (event) => {
    if (state.models[role] === viewer && event.detail?.source === "user-interaction") syncRotation(role);
  });
  viewer.src = safeURL(asset.src);
  stage.replaceChildren(viewer, loading);
  updateStatus();
}

function selectVariant(id) {
  const variant = state.object.variants.find((item) => item.id === id) || state.object.variants[0];
  state.variant = variant;
  byId("variant").value = variant.id;
  renderReadout(variant);
  mountModel("candidate", variant, state.object.label);
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
  // Invalidate the old reconstruction before either new preview can emit events.
  state.ready.candidate = false;
  state.models.candidate = null;
  mountModel("reference", object.reference, object.label);
  selectVariant(next.id);
}

function rotate(role, direction) {
  if (!state.ready[role]) return;
  const viewer = state.models[role];
  const orbit = viewer.getCameraOrbit();
  const step = Math.PI / 12;
  const theta = orbit.theta + (direction === "left" ? -step : direction === "right" ? step : 0);
  const phi = Math.min(Math.PI - 0.05, Math.max(0.05, orbit.phi + (direction === "up" ? -step : direction === "down" ? step : 0)));
  viewer.cameraOrbit = `${theta}rad ${phi}rad ${orbit.radius}m`;
  viewer.jumpCameraToGoal();
  // cameraOrbit is reactive: getCameraOrbit() still reports the previous frame
  // here. Give both elements the same requested angles in this update cycle.
  syncRotation(role, { theta, phi });
}

async function initialise() {
  root.dataset.state = "loading";
  byId("status").textContent = "Loading comparison data and 3D viewer…";
  try {
    const [data] = await Promise.all([
      fetch(manifestURL, { credentials: "omit" }).then((response) => {
        if (!response.ok) throw new Error("Comparison data unavailable");
        return response.json();
      }).then(validateManifest),
      import(new URL("vendor/model-viewer.min.js", document.baseURI).href),
    ]);
    await customElements.whenDefined("model-viewer");
    state.data = data;
    byId("object").replaceChildren(...data.objects.map((object) => new Option(object.label, object.id)));
    byId("object").disabled = false;
    byId("variant").disabled = false;
    byId("object").addEventListener("change", (event) => selectObject(event.target.value));
    byId("variant").addEventListener("change", (event) => selectVariant(event.target.value));
    byId("link").addEventListener("change", () => syncRotation("reference"));
    byId("reset").addEventListener("click", () => {
      for (const role of ["reference", "candidate"]) {
        if (state.ready[role]) {
          state.models[role].cameraOrbit = DEFAULT_ORBIT;
          state.models[role].jumpCameraToGoal();
        }
      }
    });
    root.querySelectorAll("[data-rotate]").forEach((button) => {
      button.addEventListener("click", () => rotate(button.dataset.view, button.dataset.rotate));
    });
    byId("selection").textContent = typeof data.selection_policy === "string" ? data.selection_policy : "The first recorded trial is shown for every model run.";
    if (data.reference_license?.label && data.reference_license?.url) {
      byId("selection").append(document.createTextNode(" Reference assets: "));
      const link = element("a", "", data.reference_license.label);
      link.href = safeURL(data.reference_license.url);
      byId("selection").append(link, document.createTextNode("."));
    }
    root.dataset.state = "ready";
    selectObject(data.objects[0].id);
  } catch {
    root.dataset.state = "error";
    byId("status").textContent = "The interactive comparison is temporarily unavailable. Published results remain available on Hugging Face.";
    for (const role of ["reference", "candidate"]) {
      const stage = byId(`${role}-stage`);
      stage.setAttribute("aria-busy", "false");
      stage.replaceChildren(element("p", "explorer-placeholder", "Preview unavailable"));
    }
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
