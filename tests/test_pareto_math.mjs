import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../site/pareto-math.js", import.meta.url), "utf8");
const { pairedScore, paretoFrontier } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

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

test("partial usage and missing, negative or out-of-range measurements cannot define a frontier", () => {
  const points = [
    { id: "measured", x: 2, y: 80 }, { id: "partial", x: 1, y: 100, complete: false },
    { x: null, y: 90 }, { x: -1, y: 100 }, { x: 1, y: 101 }, { x: 1, y: NaN },
  ];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["measured"]);
});

test("zero is a measured value and input ordering is not mutated", () => {
  const points = [{ id: "later", x: 3, y: 90 }, { id: "zero", x: 0, y: 0 }];
  assert.deepEqual(paretoFrontier(points).map((point) => point.id), ["zero", "later"]);
  assert.deepEqual(points.map((point) => point.id), ["later", "zero"]);
});

test("partial usage requires a paired score cohort instead of the full-run score", () => {
  const run = { scores: { formal: { value: 90 } } };
  assert.equal(pairedScore(run, { value: 2, complete: false }, "formal"), null);
  const paired = { value: 70, x_value: 3 };
  assert.equal(pairedScore(run, { value: 2, complete: false, scores: { formal: paired } }, "formal"), paired);
});

test("paired eligible score cohorts override full-run usage even with complete usage", () => {
  const run = { scores: { kinematic: { value: 90 } } };
  const paired = { value: 90, x_value: 4 };
  assert.equal(pairedScore(run, { value: 2, complete: true, scores: { kinematic: paired } }, "kinematic"), paired);
  assert.equal(pairedScore(run, { value: null, complete: true }, "kinematic"), null);
  assert.equal(pairedScore(run, { value: 2, complete: false, scores: { kinematic: { value: null, lower: 0, upper: 100 } } }, "kinematic"), null);
});

test("usage cannot be paired with any score cohort without explicit matching usage", () => {
  const run = { scores: {
    formal: { value: 90, measured_attempts: 100, eligible_attempts: 100 },
    kinematic: { value: 80, measured_attempts: 55, eligible_attempts: 55 },
  } };
  const metric = { value: 2, complete: true, total_attempts: 100 };
  assert.equal(pairedScore(run, metric, "formal"), null);
  assert.equal(pairedScore(run, metric, "kinematic"), null);
  assert.equal(pairedScore(run, { ...metric, scores: { formal: { value: 90 } } }, "formal"), null);
  assert.equal(pairedScore(run, { ...metric, scores: { formal: { value: 90, x_value: -1 } } }, "formal"), null);
});
