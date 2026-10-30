// Agent-owned image receipts and enforced model-turn budget; no CAD tools.
import {appendFileSync, existsSync, readFileSync, writeFileSync, renameSync} from "node:fs";
import {createHash} from "node:crypto";
import {dirname, join} from "node:path";

export function makeNotice(config, turn, candidateExists) {
  const remaining = config.model_turn_limit - turn + 1;
  const phase = remaining <= config.final_validation_turns ? "final_revalidation"
    : candidateExists ? "validate_or_repair" : "build_first_candidate";
  const overdue = !candidateExists && turn > config.first_candidate_target_turn;
  const action = phase === "final_revalidation"
    ? "FINAL REVALIDATION WINDOW: stop exploration and optional edits. Use an available genuine feedback call on the changed final candidate now, inspect the result and leave its bytes stable."
    : candidateExists ? "An entrypoint exists; existence alone is not validity. Validate this first complete candidate before further polishing, then repair actionable failures."
    : `${overdue ? "FIRST CANDIDATE TARGET MISSED: " : ""}Write the first complete candidate by turn ${config.first_candidate_target_turn}, after only essential instruction/image/API checks; stop broad package or source exploration.`;
  const message = `Model-turn budget: turn ${turn} of ${config.model_turn_limit}. `
    + `${remaining} turns remain including this turn; ${remaining - 1} remain after this turn. `
    + "One turn is one assistant generation plus its whole tool-result batch, regardless of the number of tools; notices and validator calls are not extra turns. "
    + `The hard cap covers this session and any bounded length continuation. Reserve the final ${config.final_validation_turns} turns for final validation and reporting. `
    + (remaining === 1 ? "This is the last turn: perform any remaining final validation in this turn's tool batch; there is no extra summary turn. " : "")
    + action + " The existing five admitted-validator-call allowance is unchanged; consult genuine feedback health for used/remaining. A passing candidate can finish early; do not exhaust calls or revalidate unchanged bytes merely to consume the allowance.";
  return {turn, limit: config.model_turn_limit, remaining_including_current: remaining,
    remaining_after_current: remaining - 1, phase, candidate_target_overdue: overdue, message};
}

function imageDigests(value, result = []) {
  if (Array.isArray(value)) for (const item of value) imageDigests(item, result);
  else if (value && typeof value === "object") {
    if (value.type === "image_url") {
      const uri = typeof value.image_url === "string" ? value.image_url : value.image_url?.url;
      if (typeof uri === "string" && uri.startsWith("data:image/png;base64,"))
        result.push(createHash("sha256").update(Buffer.from(uri.split(",", 2)[1], "base64")).digest("hex"));
    } else for (const item of Object.values(value)) imageDigests(item, result);
  }
  return result;
}

export default function (pi) {
  const file = process.env.PI_NVIDIA_RUNTIME_FILE;
  if (!file) throw new Error("Agent runtime file is required");
  const logs = dirname(dirname(dirname(file)));
  const receiptFile = join(logs, "pi-turn-budget.json");
  const config = JSON.parse(readFileSync(file, "utf8"));
  if (config.model_turn_limit !== 64) throw new Error("This runtime requires the explicit 64-model-turn policy");
  let state = existsSync(receiptFile) ? JSON.parse(readFileSync(receiptFile, "utf8"))
    : {limit: 64, started: 0, completed: 0, exhausted: false};
  if (state.limit !== 64 || !Number.isInteger(state.started) || !Number.isInteger(state.completed)
    || state.started < 0 || state.started > 64 || state.completed < 0 || state.completed > state.started)
    throw new Error("Invalid persisted turn-budget receipt");
  function persist() {
    const temporary = receiptFile + ".tmp";
    writeFileSync(temporary, JSON.stringify({...state,
      remaining: 64 - state.started, candidate_exists: existsSync(config.submission_entrypoint),
      definition: "one_assistant_generation_plus_tool_result_batch"}) + "\n");
    renameSync(temporary, receiptFile);
  }
  persist();
  let sequence = 0, verifiedInitialImage = false, blocked = false;
  pi.on("turn_start", (_event, ctx) => {
    if (state.started >= 64) { blocked = true; ctx.abort(); return; }
    state.started++;
    persist();
  });
  pi.on("turn_end", (_event, ctx) => {
    // Pi dispatches this boundary only after every tool result was persisted.
    if (blocked) return;
    state.completed = state.started;
    state.exhausted = state.completed === 64;
    persist();
    // BoundaryResult.continue=false cannot suppress Pi's natural tool continuation.
    // Supported abort stops after this complete batch; ACP reports max_turn_requests.
    if (state.exhausted) ctx.abort();
  });
  pi.on("context", (event) => {
    if (blocked || state.exhausted) return;
    const notice = makeNotice(config, state.started, existsSync(config.submission_entrypoint));
    appendFileSync(join(logs, "pi-budget-notices.jsonl"), JSON.stringify(notice) + "\n");
    return {messages: [...event.messages, {role: "user", content: [{type: "text", text: notice.message}], timestamp: Date.now()}]};
  });
  pi.on("before_provider_request", (event, ctx) => {
    if (blocked || state.exhausted) { ctx.abort(); return; }
    const config = JSON.parse(readFileSync(file, "utf8"));
    const digests = imageDigests(event.payload?.messages);
    const expected = config.expected_image_sha256;
    const matched = expected ? digests.includes(expected) : null;
    appendFileSync(join(logs, "pi-image-wire.jsonl"), JSON.stringify({request: ++sequence,
      expected_sha256: expected, png_sha256: digests, matched, initial_image_verified: verifiedInitialImage}) + "\n");
    if (expected && !verifiedInitialImage && !matched) {
      ctx.abort(); // Abort the existing request signal; throwing alone is swallowed by Pi's hook runner.
      throw new Error("Initial frozen observation missing from the provider payload; inference aborted");
    }
    if (matched) verifiedInitialImage = true;
  });
}
