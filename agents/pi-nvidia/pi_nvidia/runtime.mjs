// Agent-owned image receipts and advisory deadline notices; no CAD tools.
import {appendFileSync, existsSync, readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {dirname, join} from "node:path";

export function makeNotice(config, now, candidateExists) {
  const elapsed = Math.max(0, Number((now - BigInt(config.started_monotonic_ns)) / 1000000000n));
  const remaining = Math.max(0, Number((BigInt(config.working_deadline_monotonic_ns) - now) / 1000000000n));
  const phase = remaining === 0 ? "finish_now" : remaining <= config.final_reserve_sec
    ? "final_revalidation" : candidateExists ? "validate_or_repair" : "build_first_candidate";
  const overdue = !candidateExists && elapsed >= config.first_candidate_target_sec;
  const action = phase === "finish_now" ? "The conservative working deadline has passed. Finish now; report validation status and unresolved findings."
    : phase === "final_revalidation" ? "FINAL REVALIDATION WINDOW: stop exploration and optional edits. Validate the changed final candidate now using the genuine client and an available call; keep its bytes stable afterward."
    : candidateExists ? "An entrypoint exists; existence alone is not validity. If this complete candidate is not yet validated, validate it now before further polishing, then repair actionable failures."
    : `${overdue ? "FIRST CANDIDATE TARGET MISSED: " : ""}Write the first complete candidate now, after only essential instruction/image/API checks; do not continue broad package or source exploration.`;
  const message = `Runtime budget notice: configured Harbor authoring limit ${config.configured_limit_sec}s. `
    + `Conservative working time remaining ${remaining}s; adapter elapsed ${elapsed}s; working deadline ${config.working_deadline_utc}. `
    + `This is a conservative adapter clock with startup headroom, not Harbor's unavailable absolute deadline; Harbor may stop earlier. `
    + `Reserve the final ${config.final_reserve_sec}s for final validation and reporting. ${action} `
    + "The existing five admitted-validator-call allowance is unchanged; consult genuine feedback health for used/remaining. A passing candidate can finish early; do not exhaust calls or revalidate unchanged bytes just because of this notice.";
  return {phase, elapsed_sec: elapsed, remaining_sec: remaining, candidate_target_overdue: overdue, message};
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
  let sequence = 0, verifiedInitialImage = false;
  pi.on("context", (event) => {
    const config = JSON.parse(readFileSync(file, "utf8"));
    const notice = makeNotice(config, process.hrtime.bigint(), existsSync(config.submission_entrypoint));
    appendFileSync(join(logs, "pi-budget-notices.jsonl"), JSON.stringify({...notice, deadline_basis: config.deadline_basis}) + "\n");
    return {messages: [...event.messages, {role: "user", content: [{type: "text", text: notice.message}], timestamp: Date.now()}]};
  });
  pi.on("before_provider_request", (event, ctx) => {
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
