"use strict";
const region = document.getElementById("leaderboard-table-region");
const body = document.getElementById("leaderboard-body");
const status = document.getElementById("leaderboard-status");
const archive = document.getElementById("leaderboard-archive");
const format = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString("en-GB", {maximumFractionDigits: digits}) : "—";
region.dataset.loadState = "loading";
fetch("assets/pareto.json", {credentials: "omit"}).then(response => {
  if (!response.ok) throw new Error("Results unavailable");
  return response.json();
}).then(data => {
  if (!Array.isArray(data.runs) || !data.runs.length) throw new Error("Results unavailable");
  body.replaceChildren();
  for (const run of data.runs) {
    const row = document.createElement("tr");
    const add = (text, title) => { const cell = document.createElement("td"); cell.textContent = text; if (title) cell.title = title; row.append(cell); return cell; };
    add(run.label);
    for (const name of ["formal", "visual", "kinematic"]) {
      const metric = run.scores[name];
      const cell = add(format(metric?.value), `${metric.measured_attempts}/${metric.eligible_attempts} scores`);
      if (name === "formal") cell.style.fontWeight = "700";
    }
    add("$" + format(run.metrics.cost_usd_per_asset.value, 4));
    add(format(run.metrics.elapsed_seconds_per_asset.value / 60));
    add(format(run.metrics.tokens_per_asset.value, 0));
    const cell = add(""); const link = document.createElement("a"); link.href = run.source_url; link.textContent = "View"; cell.append(link);
    body.append(row);
  }
  document.getElementById("leaderboard-caption").textContent = `${data.reference_count} NV100 objects · three reconstructions per configuration · evaluator ${data.evaluator_version}`;
  region.hidden = false; status.hidden = true; archive.hidden = true; region.dataset.loadState = "ready";
}).catch(() => { region.dataset.loadState = "error"; status.hidden = false; status.textContent = "View the results on Hugging Face."; archive.hidden = false; });
