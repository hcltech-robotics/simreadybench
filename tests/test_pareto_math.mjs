import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../site/pareto-math.js", import.meta.url), "utf8");
const { availableScore, paretoFrontier } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("frontier minimises usage and maximises fidelity independently", () => {
  const points = [
    { id: "cheap", x: 1, y: 40 }, { id: "middle", x: 2, y: 70 },
    { id: "best", x: 3, y: 90 }, { id: "dominated", x: 4, y: 80 },
  ];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["cheap", "middle", "best"]);
});

test("one strict improvement dominates; identical configurations remain tied", () => {
  const points = [
    { id: "equal1", x: 2, y: 80 }, { id: "equal2", x: 2, y: 80 },
    { id: "same-score-costlier", x: 3, y: 80 }, { id: "same-cost-weaker", x: 2, y: 70 },
  ];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["equal1", "equal2"]);
});

test("partial mean measurements participate in the frontier; missing or negative resources do not", () => {
  const points = [
    { id: "measured", x: 2, y: 80 }, { id: "partial", x: 1, y: 100, complete: false },
    { x: null, y: 90 }, { x: -1, y: 100 }, { x: 1, y: NaN },
  ];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["partial"]);
});

test("standardised cumulative values may be negative", () => {
  const points = [{ id: "cheap", x: 1, y: -2 }, { id: "stronger", x: 2, y: -1 }, { id: "dominated", x: 3, y: -2 }];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["cheap", "stronger"]);
});

test("zero is a measured value and input ordering is not mutated", () => {
  const points = [{ id: "later", x: 3, y: 90 }, { id: "zero", x: 0, y: 0 }];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["zero", "later"]);
  assert.deepEqual(points.map((point) => point.id), ["later", "zero"]);
});

test("partial usage uses its available mean and the configuration's available score mean", () => {
  const run = { scores: { formal: { value: 90 } } };
  const metric = { value: 2, complete: false, scores: { formal: { value: 70, x_value: 3 } } };
  assert.deepEqual(availableScore(run, metric, "formal"), { value: 90, x_value: 2 });
});

test("missing means are never zero-filled, but a measured zero remains valid", () => {
  const run = { scores: { formal: { value: 0 }, visual: { value: null } } };
  assert.deepEqual(availableScore(run, { value: 0 }, "formal"), { value: 0, x_value: 0 });
  assert.equal(availableScore(run, { value: null }, "formal"), null);
  assert.equal(availableScore(run, { value: 2 }, "visual"), null);
  assert.equal(availableScore(run, { value: -1 }, "formal"), null);
});

test("cumulative means may be negative; invalid fidelity scores are rejected", () => {
  const run = { scores: { cumulative: { value: -1.5 }, formal: { value: -1 }, visual: { value: 101 } } };
  assert.deepEqual(availableScore(run, { value: 2 }, "cumulative"), { value: -1.5, x_value: 2 });
  assert.equal(availableScore(run, { value: 2 }, "formal"), null);
  assert.equal(availableScore(run, { value: 2 }, "visual"), null);
});

test("matched chart means take precedence while preserving missing scores", () => {
  const run = {scores: {formal: {value: 99}}};
  const metric = {value: 2, paired_scores: {formal: {value: 90, x_value: 2, measured_attempts: 250}}};
  assert.deepEqual(availableScore(run, metric, "formal"), {value: 90, x_value: 2, measured_attempts: 250});
  assert.equal(availableScore(run, {...metric, paired_scores: {formal: {value: null, x_value: 2}}}, "formal"), null);
});
