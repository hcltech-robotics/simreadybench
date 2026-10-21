"use strict";

const leaderboardBody = document.getElementById("leaderboard-body");
const leaderboardStatus = document.getElementById("leaderboard-status");
const leaderboardTable = document.getElementById("leaderboard-table-region");
const leaderboardArchive = document.getElementById("leaderboard-archive");
const leaderboardSource = "https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/main/leaderboard.json";
const fidelityComponents = ["shape", "rigid_parts", "joints", "materials"];
const compositeNames = ["formal", "visual", "kinematic"];
let leaderboardEntries = [];

function textCell(value, className) {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) cell.className = className;
  return cell;
}

function metricDetail(cell, value) {
  const detail = document.createElement("span");
  detail.className = "metric-coverage";
  detail.textContent = value;
  cell.append(detail);
}

function countLabel(value) {
  return value.toLocaleString("en-GB", { maximumSignificantDigits: 7 });
}

function metricCell(metric = {}) {
  const inapplicable = metric.status === "not_applicable";
  const cell = textCell(isScore(metric.score) ? metric.score.toFixed(2) : inapplicable ? "N/A" : "Not measured", "metric-value");
  const interval = metric.confidence_interval;
  if (isScore(metric.score) && interval?.level === 0.95
      && isScore(interval.lower) && isScore(interval.upper) && interval.lower <= interval.upper) {
    metricDetail(cell, `95% CI [${interval.lower.toFixed(2)}, ${interval.upper.toFixed(2)}]`);
  }
  if (inapplicable) {
    metricDetail(cell, "No eligible reference criteria");
    return cell;
  }
  const credit = metric.earned;
  if (Number.isFinite(credit) && Number.isFinite(metric.eligible)) {
    metricDetail(cell, `${countLabel(credit)}/${countLabel(metric.eligible)} credit`);
  }
  if (Number.isFinite(metric.measured) && Number.isFinite(metric.eligible)) {
    metricDetail(cell, `${countLabel(metric.measured)}/${countLabel(metric.eligible)} measured`);
  }
  metricDetail(cell, isScore(metric.coverage) ? `${metric.coverage.toFixed(1)}% mean reference coverage` : "Reference coverage not measured");
  if (Number.isInteger(metric.eligible_attempts) && Number.isInteger(metric.measured_attempts)) {
    metricDetail(cell, `${metric.measured_attempts}/${metric.eligible_attempts} eligible reconstructions scored`);
  }
  if (metric.unknown_eligibility_attempts) {
    metricDetail(cell, `${metric.unknown_eligibility_attempts} with eligibility not measured`);
  }
  return cell;
}

function usageMean(metric) {
  return metric && Number.isFinite(metric.mean) && metric.mean >= 0
    && Number.isInteger(metric.total_attempts) && metric.total_attempts > 0
    && metric.measured_attempts === metric.total_attempts;
}

function usageCount(value) {
  return value.toLocaleString("en-GB", { maximumFractionDigits: 2 });
}

function usageSeconds(value) {
  if (value < 60) return `${usageCount(value)} s`;
  return `${usageCount(value / 60)} min`;
}

function usageCost(value) {
  if (value > 0 && value < 0.0001) return "<USD 0.0001";
  return `USD ${value.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;
}

function usageLabel(metric, format) {
  return usageMean(metric) ? format(metric.mean) : "Not measured";
}

function usageCoverage(cell, label, metric) {
  if (!usageMean(metric) && Number.isInteger(metric?.measured_attempts)
      && Number.isInteger(metric?.total_attempts) && metric.total_attempts > 0) {
    metricDetail(cell, `${label}: ${metric.measured_attempts}/${metric.total_attempts} reconstructions measured`);
  }
}

function usageCell(usage = {}, primary, details, format) {
  const [name, label] = primary;
  const metric = usage?.[name];
  const cell = textCell(usageLabel(metric, format), "metric-value");
  if (usageMean(metric)) metricDetail(cell, label);
  usageCoverage(cell, label, metric);
  details.forEach(([detailName, detailLabel, detailFormat = format]) => {
    const detailMetric = usage?.[detailName];
    metricDetail(cell, `${detailLabel}: ${usageLabel(detailMetric, detailFormat)}`);
    usageCoverage(cell, detailLabel, detailMetric);
  });
  return cell;
}

function renderLeaderboard() {
  const entries = leaderboardEntries;
  leaderboardBody.replaceChildren();
  leaderboardTable.hidden = !entries.length;
  leaderboardStatus.hidden = !entries.length;
  leaderboardArchive.hidden = Boolean(entries.length);
  entries.forEach((entry) => {
    const row = document.createElement("tr");
    row.append(textCell(entry.model), textCell(entry.reasoning_effort || "Default"), textCell(entry.harness), textCell(entry.skills.length ? entry.skills.join(", ") : "None"));
    compositeNames.forEach((name) => {
      row.append(metricCell(entry.composite_scores?.[name]));
    });
    row.append(
      usageCell(entry.usage, ["inference_calls_used", "model requests"], [
        ["validation_calls_used", "Validation calls"], ["model_request_errors", "Model request errors"],
      ], usageCount),
      usageCell(entry.usage, ["elapsed_seconds", "elapsed time"], [
        ["model_request_seconds", "Model request time"],
      ], usageSeconds),
      usageCell(entry.usage, ["total_tokens", "tokens"], [
        ["input_tokens", "Input tokens"], ["output_tokens", "Output tokens"],
      ], usageCount),
      usageCell(entry.usage, ["total_cost_usd", "total cost"], [
        ["inference_cost_usd", "Inference"], ["compute_cost_usd", "Compute"],
      ], usageCost),
    );
    const resultCell = document.createElement("td");
    const link = document.createElement("a");
    link.href = entry.results_url;
    link.textContent = "View";
    resultCell.append(link);
    row.append(resultCell);
    leaderboardBody.append(row);
  });
  leaderboardStatus.textContent = entries.length ? `${entries.length} ${entries.length === 1 ? "entry" : "entries"}. Each NV25 entry covers 100 reconstructions of the 25 public references.` : "";
}

function isScore(value) {
  return Number.isFinite(value) && value >= 0 && value <= 100;
}

function isCompleteEntry(entry) {
  if (!entry || typeof entry !== "object") return false;
  if (![entry.model, entry.harness].every((value) => typeof value === "string" && value.length)) return false;
  if (!Array.isArray(entry.skills) || !entry.skills.every((value) => typeof value === "string")) return false;
  if (entry.reference_set !== "NV25" || entry.reference_count !== 25) return false;
  if (entry.repetitions !== 4 || entry.completed_attempts !== 100) return false;
  if (entry.profile?.id !== "Robotics-Prop" || entry.profile?.version !== "4.0.0") return false;
  if (!isScore(entry.compliance) || !entry.fidelity) return false;
  if (!compositeNames.every((name) => {
    const metric = entry.composite_scores?.[name];
    return metric && (metric.score === null || isScore(metric.score));
  })) return false;
  if ([entry.mandatory, entry.optional].some((metric) => metric && (
    !(metric.score === null || isScore(metric.score)) || !(metric.coverage === null || isScore(metric.coverage))
  ))) return false;
  if (!fidelityComponents.every((component) => {
    const metric = entry.fidelity[component];
    return metric && (metric.score === null || isScore(metric.score)) && (metric.coverage === null || isScore(metric.coverage));
  })) return false;
  try {
    return new URL(entry.results_url).protocol === "https:";
  } catch {
    return false;
  }
}

fetch(leaderboardSource, { mode: "cors", credentials: "omit", cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("Leaderboard unavailable");
    return response.json();
  })
  .then((data) => {
    if (data.schema !== "simreadybench-leaderboard/v1" || !Array.isArray(data.entries)) throw new Error("Invalid leaderboard data");
    leaderboardEntries = data.entries.filter(isCompleteEntry);
    renderLeaderboard();
  })
  .catch(() => {
    leaderboardEntries = [];
    renderLeaderboard();
  });
