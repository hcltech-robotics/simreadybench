import "./vendor/d3-7.9.0/d3.min.js";
import { availableScore, paretoFrontier } from "./pareto-math.js?v=nv100-20261029";
import { compactPointLabel, placePointLabels } from "./pareto-labels.js?v=viewer-labels-20261029";

const d3 = globalThis.d3;
const root = document.getElementById("pareto-root");
const scoreTypes = [
  { id: "formal", label: "Formal fidelity", colour: "#0072b2" },
  { id: "visual", label: "Visual fidelity", colour: "#d55e00" },
  { id: "kinematic", label: "Kinematic fidelity", colour: "#009e73" },
];
const axes = {
  cost_usd_per_asset: {
    label: "Inference cost", unit: "$ / asset", divisor: 1,
    tick: d3.format("$,.2f"), format: (value) => `USD ${number(value, 4)}`,
  },
  tokens_per_asset: {
    label: "Tokens", unit: "tokens / asset", divisor: 1,
    tick: d3.format("~s"), format: (value) => number(value, 0),
  },
  elapsed_seconds_per_asset: {
    label: "Trial time", unit: "min / asset", divisor: 60,
    tick: d3.format("~g"), format: (value) => `${number(value / 60, 2)} min`,
  },
};

function number(value, digits = 2) {
  return value.toLocaleString("en-GB", { maximumFractionDigits: digits });
}

function publicLabel(value) {
  return String(value ?? "").replace(/\bOrichalcum(?:@|\s+)([0-9a-f]{9,64})\b/gi,
    (_, pin) => `Orichalcum …${pin.slice(-8)}`);
}

function configLabel(run) {
  if (run.model_display_name) return `${run.model_display_name} · Reasoning: ${run.reasoning_display} · ${run.authoring_label || run.harness}`;
  if (run.label) return run.label;
  const parts = [run.model, run.harness, run.reasoning || "default reasoning"];
  if (run.orichalcum_pin) parts.push(`Orichalcum …${run.orichalcum_pin.slice(-8)}`);
  return parts.map(publicLabel).filter(Boolean).join(" · ");
}

function safeSource(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function element(name, text, className) {
  const node = document.createElement(name);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function metricCoverage(metric) {
  if (!metric) return "Not measured";
  const parts = [];
  if (Number.isInteger(metric.measured_attempts) && Number.isInteger(metric.total_attempts)) {
    parts.push(`${metric.measured_attempts}/${metric.total_attempts} executions`);
  }
  if (Number.isInteger(metric.measured_assets) && Number.isInteger(metric.total_assets)) {
    parts.push(`${metric.measured_assets}/${metric.total_assets} assets`);
  }
  return parts.join(" · ") || (metric.complete ? "Complete coverage" : "Coverage not recorded");
}

function scoreText(score) {
  if (Number.isFinite(score?.value)) return number(score.value);
  if (score?.status === "not_applicable") return "N/A";
  if (Number.isFinite(score?.lower) && Number.isFinite(score?.upper)) {
    return `${number(score.lower)}–${number(score.upper)} (bounds)`;
  }
  return "Not measured";
}

function scoreCoverage(score) {
  const parts = [];
  if (Number.isInteger(score?.measured_attempts) && Number.isInteger(score?.eligible_attempts)) {
    parts.push(`${score.measured_attempts}/${score.eligible_attempts} eligible reconstructions`);
  }
  if (Number.isInteger(score?.measured_assets) && Number.isInteger(score?.eligible_assets)) {
    parts.push(`${score.measured_assets}/${score.eligible_assets} eligible assets`);
  }
  return parts.join(" · ");
}

function addDetail(cell, text) {
  if (text) cell.append(element("span", publicLabel(text), "metric-coverage"));
}

function makePoints(runs, metricId, scoreIds) {
  const points = [];
  for (const run of runs) {
    const metric = run.metrics?.[metricId];
    for (const scoreId of scoreIds) {
      const score = availableScore(run, metric, scoreId);
      if (!score) continue;
      const complete = metric.complete === true && score.complete === true;
      points.push({
        id: `${run.id}:${scoreId}`, run, metric, score, scoreId,
        x: score.x_value / axes[metricId].divisor, rawX: score.x_value, y: score.value,
        resourceIncomplete: metric.complete !== true,
        complete,
        group: run.frontier_group || run.id,
      });
    }
  }
  return points;
}

async function initialise() {
  if (!root || root._paretoInitialised) return;
  root._paretoInitialised = true;
  const plot = document.getElementById("pareto-plot");
  const tooltip = document.getElementById("pareto-tooltip");
  const status = document.getElementById("pareto-status");
  const table = document.getElementById("pareto-table-body");
  const xSelect = document.getElementById("pareto-x");
  const separate = document.getElementById("pareto-separate");
  const scoreButtons = [...root.querySelectorAll("[data-pareto-score]")];
  const controls = [...scoreButtons, separate, xSelect];
  const isSnapshot = new URLSearchParams(window.location.search).has("snapshot");
  controls.forEach((control) => { control.disabled = true; });
  root.dataset.loadState = "loading";
  tooltip.hidden = true;
  tooltip.replaceChildren();

  let data;
  try {
    const response = await fetch("assets/leaderboard.json", { credentials: "omit", cache: "no-store" });
    if (!response.ok) throw new Error("Measurements unavailable");
    data = await response.json();
    if (!Array.isArray(data.runs) || !data.runs.every((run) => typeof run.id === "string"
      && typeof run.model === "string" && run.scores && run.metrics)) throw new Error("Invalid measurements");
  } catch {
    root.dataset.loadState = "error";
    plot.querySelectorAll(".pareto-point").forEach((point) => {
      point.removeAttribute("tabindex");
      point.setAttribute("role", "img");
    });
    status.textContent = table.children.length
      ? "Live measurements unavailable. The captured chart and table remain visible."
      : "Trajectory measurements are temporarily unavailable.";
    return;
  }

  const runs = data.runs;
  const symbols = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond,
    d3.symbolCross, d3.symbolStar, d3.symbolWye];
  const familyName = (run) => [run.model_display_name || run.model_family || run.model,
    run.authoring_label || run.harness].filter(Boolean).join(' · ');
  const families = [...new Set(runs.map(familyName))];
  const familySymbol = new Map(families.map((family, index) => [family, symbols[index % symbols.length]]));
  let selectedScore = "formal";
  let selectedPoint = null;
  let resizeFrame = null;
  xSelect.value = "cost_usd_per_asset";
  separate.checked = true;
  scoreButtons.forEach((button) => {
    const active = button.dataset.paretoScore === selectedScore;
    button.setAttribute("aria-pressed", String(active));
    button.classList.toggle("is-dark", active);
  });
  controls.forEach((control) => { control.disabled = false; });

  function hideTooltip() {
    tooltip.hidden = true;
    selectedPoint = null;
    plot.querySelectorAll(".pareto-point").forEach((point) => point.classList.remove("is-selected"));
  }

  function showTooltip(event, point) {
    const anchor = event.currentTarget;
    const target = anchor.getBoundingClientRect();
    const centre = { x: target.left + target.width / 2, y: target.top + target.height / 2 };
    const nearby = [...plot.querySelectorAll(".pareto-point")].filter((node) => {
      const bounds = node.getBoundingClientRect();
      return Math.hypot(bounds.left + bounds.width / 2 - centre.x,
        bounds.top + bounds.height / 2 - centre.y) <= 24;
    });
    selectedPoint = point.id;
    plot.querySelectorAll(".pareto-point").forEach((node) => node.classList.toggle("is-selected", node.dataset.pointId === point.id));
    tooltip.replaceChildren();
    const close = element("button", "×", "pareto-tooltip-close");
    close.type = "button";
    close.setAttribute("aria-label", "Close configuration details");
    close.addEventListener("click", hideTooltip);
    tooltip.append(close);
    if (nearby.length > 1) {
      const label = element("label", `Nearby observations (${nearby.length})`, "pareto-nearby-label");
      label.htmlFor = "pareto-nearby";
      const chooser = element("select", undefined, "pareto-nearby");
      chooser.id = "pareto-nearby";
      nearby.forEach((node) => {
        const neighbour = node.__data__;
        const scoreName = scoreTypes.find((score) => score.id === neighbour.scoreId).label;
        const option = element("option", `${configLabel(neighbour.run)} · ${scoreName}`);
        option.value = neighbour.id;
        option.selected = neighbour.id === point.id;
        chooser.append(option);
      });
      chooser.addEventListener("change", () => {
        const selected = nearby.find((node) => node.__data__.id === chooser.value);
        if (selected) {
          showTooltip({ currentTarget: selected }, selected.__data__);
          document.getElementById("pareto-nearby")?.focus({ preventScroll: true });
        }
      });
      tooltip.append(label, chooser);
    }
    tooltip.append(element("strong", configLabel(point.run)));
    const scoreType = scoreTypes.find((score) => score.id === point.scoreId);
    const cumulative = point.scoreId === "cumulative";
    const measurement = element("p", `${scoreType.label}: ${number(point.y)}${cumulative ? "" : " / 100"}`, "pareto-tooltip-score");
    measurement.style.color = scoreType.colour;
    tooltip.append(measurement);
    tooltip.append(element("p", `${axes[xSelect.value].label}: ${axes[xSelect.value].format(point.rawX)} per asset`));
    tooltip.append(element("p", metricCoverage(point.metric), "table-note"));
    const measuredScoreCoverage = scoreCoverage(point.score);
    if (measuredScoreCoverage) tooltip.append(element("p", `Score coverage: ${measuredScoreCoverage}`, "table-note"));
    if (cumulative) {
      tooltip.append(element("p", "Mean of the three standardised subscores", "table-note"));
      for (const type of scoreTypes.filter((score) => score.id !== "cumulative")) {
        const detail = element("p", `${type.label}: ${scoreText(point.run.scores[type.id])} / 100`, "table-note");
        detail.style.color = type.colour;
        addDetail(detail, scoreCoverage(point.run.scores[type.id]));
        tooltip.append(detail);
      }
    }

    if (point.run.protocol) tooltip.append(element("p", publicLabel(point.run.protocol), "table-note"));
    if (point.run.evaluator_version) tooltip.append(element("p", `Evaluator: ${point.run.evaluator_version}`, "table-note"));
    tooltip.append(element("p", point.resourceIncomplete ? "Provisional frontier position: resource total is incomplete." : point.frontier
      ? "On the frontier for this score and evaluator" : "Dominated within this score and evaluator", "table-note"));
    const source = safeSource(point.metric.source_url || point.run.source_url);
    if (source) {
      const link = element("a", "Results");
      link.href = source;
      tooltip.append(link);
    }
    tooltip.hidden = false;
    const wrapper = plot.parentElement.getBoundingClientRect();
    const left = Math.max(8, Math.min(target.left - wrapper.left + 14,
      wrapper.width - tooltip.offsetWidth - 8));
    const preferredTop = target.bottom - wrapper.top + 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${Math.max(8, Math.min(preferredTop, wrapper.height - tooltip.offsetHeight - 8))}px`;
  }

  function renderTable() {
    table.replaceChildren();
    for (const run of runs) {
      const row = element("tr");
      row.append(element("th", configLabel(run)));
      row.firstChild.scope = "row";
      addDetail(row.firstChild, run.protocol);
      addDetail(row.firstChild, run.evaluator_version);
      for (const scoreType of scoreTypes) {
        const score = run.metrics[xSelect.value].paired_scores?.[scoreType.id] || run.scores[scoreType.id];
        const cell = element("td", scoreText(score), `metric-value pareto-${scoreType.id}-text`);
        addDetail(cell, scoreCoverage(score));
        if (scoreType.id === "cumulative") addDetail(cell, "Mean z-score");
        row.append(cell);
      }
      for (const [metricId, axis] of Object.entries(axes)) {
        const metric = run.metrics[metricId];
        const cell = element("td", Number.isFinite(metric?.value) ? axis.format(metric.value) : "Not measured", "metric-value");
        addDetail(cell, metricCoverage(metric));
        if (metric?.reason) addDetail(cell, metric.reason);
        row.append(cell);
      }
      const sourceCell = element("td");
      const source = safeSource(run.source_url);
      if (source) {
        const link = element("a", "View");
        link.href = source;
        sourceCell.append(link);
      } else sourceCell.textContent = "Unavailable";
      row.append(sourceCell);
      table.append(row);
    }
  }

  function renderLegend() {
    const legend = document.getElementById("pareto-run-legend");
    legend.replaceChildren();
    families.forEach((family) => {
      const item = element("li");
      item.dataset.family = family;
      const icon = d3.create("svg").attr("viewBox", "-12 -12 24 24").attr("aria-hidden", "true");
      icon.append("path").attr("d", d3.symbol().type(familySymbol.get(family)).size(65)());
      item.append(icon.node(), element("span", family));
      legend.append(item);
    });
  }

  function renderChart() {
    hideTooltip();
    const metricId = xSelect.value;
    const activeScores = selectedScore === "all" ? scoreTypes.filter((score) => score.id !== "cumulative")
      : scoreTypes.filter((score) => score.id === selectedScore);
    const points = makePoints(runs, metricId, activeScores.map((score) => score.id));
    const grouped = d3.group(points, (point) => point.scoreId, (point) => point.group);
    const frontiers = [];
    for (const [scoreId, evaluatorGroups] of grouped) {
      for (const [group, groupPoints] of evaluatorGroups) {
        const frontier = paretoFrontier(groupPoints);
        frontier.forEach((point) => { point.frontier = true; });
        frontiers.push({ scoreId, group, points: frontier });
      }
    }
    const isSeparate = selectedScore === "all";
    separate.disabled = selectedScore !== "all";
    const panels = isSeparate ? activeScores.map((score) => [score]) : [activeScores];
    const width = Math.max(300, Math.round(plot.getBoundingClientRect().width));
    const panelHeight = isSeparate ? 255 : width < 500 ? 340 : 410;
    const height = panelHeight * panels.length;
    const margin = { top: isSeparate ? 36 : 24, right: 22, bottom: 38, left: width < 500 ? 46 : 58 };
    const maxX = d3.max(points, (point) => point.x) || 1;
    const x = d3.scaleLinear().domain([0, maxX * 1.09]).nice().range([margin.left, width - margin.right]);
    plot.replaceChildren();
    const svg = d3.select(plot).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", width).attr("height", height).attr("role", "group")
      .attr("aria-label", `${activeScores.map((score) => score.label).join(", ")} against ${axes[metricId].label.toLowerCase()} per asset`);
    svg.append("title").text("Score and inference efficiency");
    svg.append("desc").text("Higher scores and lower inference use are better. Focus or select a marker for configuration, measurement coverage and source. Colours identify fidelity scores; shapes identify model families. Shading shows the area attained below each frontier.");
    panels.forEach((panelScores, panelIndex) => {
      const panel = svg.append("g").attr("transform", `translate(0,${panelIndex * panelHeight})`);
      const cumulative = panelScores[0].id === "cumulative";
      const cumulativeValues = runs.map((run) => run.scores.cumulative?.value).filter(Number.isFinite);
      const [low, high] = cumulativeValues.length ? d3.extent([0, ...cumulativeValues]) : [-1, 1];
      const padding = (high - low || 1) * 0.12;
      const plottedY = points.filter(point => panelScores.some(score => score.id === point.scoreId)).map(point => point.y);
      const range = d3.extent(plottedY);
      const gap = Math.max(1, (range[1] - range[0]) * .2);
      const scoreDomain = panelScores.length === 1 && plottedY.length
        ? [Math.max(0, range[0] - gap), Math.min(101, range[1] + gap)] : [0, 100];
      const y = d3.scaleLinear().domain(cumulative ? [low - padding, high + padding] : scoreDomain)
        .range([panelHeight - margin.bottom, margin.top]);
      if (cumulative) y.nice();
      panel.append("g").attr("class", "pareto-grid").attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat("")).call((g) => g.select(".domain").remove());
      panel.append("g").attr("class", "pareto-axis").attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(9)).call((g) => g.select(".domain").remove());
      panel.append("g").attr("class", "pareto-axis").attr("transform", `translate(0,${panelHeight - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(width < 500 ? 4 : 7).tickFormat(axes[metricId].tick).tickSize(4).tickPadding(8));
      panel.append("text").attr("class", "pareto-axis-title").attr("x", margin.left).attr("y", isSeparate ? 17 : 13)
        .text(cumulative ? "Cumulative score · mean z-score" : isSeparate ? `${panelScores[0].label} / 100` : "Fidelity / 100");
      const panelIds = new Set(panelScores.map((score) => score.id));
      frontiers.filter((frontier) => panelIds.has(frontier.scoreId) && frontier.points.length)
        .forEach((frontier) => {
          const colour = scoreTypes.find((score) => score.id === frontier.scoreId).colour;
          const attained = [...frontier.points, { ...frontier.points.at(-1), x: x.domain()[1] }];
          if (!frontier.points.some(point => point.resourceIncomplete)) panel.append("path").attr("class", "pareto-frontier-area")
            .attr("data-score", frontier.scoreId).attr("data-frontier-group", frontier.group)
            .attr("data-frontier-points", frontier.points.length).attr("fill", colour)
            .attr("d", d3.area().curve(d3.curveStepAfter).x((point) => x(point.x))
              .y0(y(y.domain()[0])).y1((point) => y(point.y))(attained));
          panel.append("path").attr("class", "pareto-frontier").attr("data-score", frontier.scoreId)
            .attr("stroke", colour)
            .attr("stroke-dasharray", frontier.points.some(point => point.resourceIncomplete) ? "5,3" : null)
            .attr("d", d3.line().curve(d3.curveStepAfter).x((point) => x(point.x)).y((point) => y(point.y))(attained));
        });
      const panelPoints = points.filter((point) => panelIds.has(point.scoreId));
      const markers = panel.selectAll(".pareto-point").data(panelPoints, (point) => point.id).join("g")
        .attr("class", "pareto-point").attr("data-run-id", (point) => point.run.id)
        .attr("data-family", (point) => familyName(point.run))
        .attr("data-score", (point) => point.scoreId).attr("data-point-id", (point) => point.id)
        .attr("data-frontier", (point) => String(Boolean(point.frontier))).attr("data-frontier-provisional", point => String(point.resourceIncomplete)).attr("data-complete", (point) => String(point.complete))
        .attr("transform", (point) => `translate(${x(point.x)},${y(point.y)})`)
        .attr("fill", (point) => point.resourceIncomplete ? "white" : scoreTypes.find((score) => score.id === point.scoreId).colour)
        .attr("stroke", (point) => scoreTypes.find((score) => score.id === point.scoreId).colour)
        .attr("tabindex", isSnapshot ? null : 0).attr("role", isSnapshot ? "img" : "button")
        .attr("aria-label", (point) => `${configLabel(point.run)}. ${scoreTypes.find((score) => score.id === point.scoreId).label}: ${number(point.y)}${point.scoreId === "cumulative" ? " mean z-score" : " / 100"}. ${axes[metricId].label}: ${axes[metricId].format(point.rawX)} per asset. ${metricCoverage(point.metric)}. ${point.resourceIncomplete ? "Provisional frontier position." : point.frontier ? "On frontier." : "Dominated."}`)
        .on("pointerdown", (event) => { event.currentTarget._pointerFocus = true; })
        .on("pointercancel blur", (event) => { event.currentTarget._pointerFocus = false; })
        .on("pointerenter", (event, point) => { if (event.pointerType !== "touch") showTooltip(event, point); })
        .on("focus", (event, point) => { if (!event.currentTarget._pointerFocus) showTooltip(event, point); })
        .on("click", (event, point) => {
          event.stopPropagation();
          event.currentTarget._pointerFocus = false;
          showTooltip(event, point);
        })
        .on("keydown", (event, point) => {
          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); showTooltip(event, point); }
          if (event.key === "Escape") hideTooltip();
        });
      markers.append("circle").attr("class", "pareto-hit").attr("r", 12);
      markers.append("path").attr("class", "pareto-symbol")
        .attr("d", (point) => d3.symbol().type(familySymbol.get(familyName(point.run))).size(width < 500 ? 105 : 85)());
      if (runs.length <= 12) {
        const labels = markers.append("text").attr("class", "pareto-point-label")
          .attr("font-size", 10).attr("fill", "#344552").attr("text-anchor", "start")
          .text(point => compactPointLabel(point.run));
        const boxes = [];
        labels.each(function(point) {
          const box = this.getBBox();
          boxes.push({ id: point.id, x: x(point.x), y: y(point.y), width: box.width, height: box.height, bearingY: box.y });
        });
        const positions = new Map(placePointLabels(boxes, { left: margin.left + 2, right: width - margin.right - 2,
          top: margin.top, bottom: panelHeight - margin.bottom - 3 }).map(box => [box.id, box]));
        labels.each(function(point) {
          const box = positions.get(point.id), measured = boxes.find(item => item.id === point.id);
          d3.select(this).attr("x", box.left - measured.x).attr("y", box.top - measured.y - measured.bearingY);
          const targetX = Math.max(box.left, Math.min(box.left + box.width, measured.x));
          const targetY = Math.max(box.top, Math.min(box.top + box.height, measured.y));
          if (Math.hypot(targetX - measured.x, targetY - measured.y) > 18) {
            panel.insert("line", ".pareto-point").attr("class", "pareto-label-leader")
              .attr("x1", measured.x).attr("y1", measured.y).attr("x2", targetX).attr("y2", targetY)
              .attr("stroke", "#94a3b8").attr("stroke-width", .7).attr("pointer-events", "none");
          }
        });
      }
      if (!panelPoints.length) panel.append("text").attr("class", "pareto-empty")
        .attr("x", (width + margin.left - margin.right) / 2).attr("y", panelHeight / 2)
        .attr("text-anchor", "middle").text("No available measurements for this selection");
    });
    const plottedRuns = new Set(points.map((point) => point.run.id)).size;

    const missing = runs.length - plottedRuns;
    status.textContent = `${plottedRuns}/${runs.length} configurations · ${points.length} score observations`
      + (missing ? ` · ${missing} without measurements for this axis` : "");
    root.dataset.axis = metricId;
    root.dataset.score = selectedScore;
    root.dataset.points = String(points.length);
    if (isSnapshot) controls.forEach((control) => { control.disabled = true; });
  }

  scoreButtons.forEach((button) => button.addEventListener("click", () => {
    selectedScore = button.dataset.paretoScore;
    scoreButtons.forEach((candidate) => {
      const active = candidate === button;
      candidate.setAttribute("aria-pressed", String(active));
      candidate.classList.toggle("is-dark", active);
    });
    renderChart();
  }));
  [xSelect, separate].forEach((control) => control.addEventListener("change", renderChart));
  root.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTooltip(); });
  root.addEventListener("click", (event) => {
    if (selectedPoint && !tooltip.contains(event.target) && !event.target.closest(".pareto-point")) hideTooltip();
  });
  root.addEventListener("focusout", (event) => {
    if (!root.contains(event.relatedTarget)) hideTooltip();
  });
  let observedWidth = Math.round(plot.getBoundingClientRect().width);
  const resize = new ResizeObserver((entries) => {
    const width = Math.round(entries[0].contentRect.width);
    if (width === observedWidth) return;
    observedWidth = width;
    if (resizeFrame) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderChart);
  });
  resize.observe(plot);
  renderTable();
  renderLegend();
  const provenance = [data.generated_at ? `Measurements captured ${data.generated_at}` : "",
    typeof data.methodology === "string" ? data.methodology : "",
    typeof data.normalisation?.description === "string" ? data.normalisation.description : ""].filter(Boolean).join(". ");
  document.getElementById("pareto-provenance").textContent = "";
  renderChart();
  document.fonts?.ready.then(renderChart);
  root.dataset.loadState = "ready";
}

initialise();
