export function compactPointLabel(run) {
  const model = run.requested_model?.split("/").at(-1) || run.model || run.id;
  return `${model}/${run.reasoning_display || run.reasoning || "default"}`;
}

const overlap = (a, b, gap = 3) => a.left < b.left + b.width + gap
  && a.left + a.width + gap > b.left && a.top < b.top + b.height + gap
  && a.top + a.height + gap > b.top;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

/** Place annotation boxes without moving any measured point or chart axis. */
export function placePointLabels(items, bounds) {
  const placed = [];
  const markers = items.map(item => ({ left: item.x - 6, top: item.y - 6, width: 12, height: 12 }));
  const order = [...items].sort((a, b) => a.y - b.y || a.x - b.x || a.id.localeCompare(b.id));
  for (const item of order) {
    const width = Math.min(item.width, bounds.right - bounds.left);
    const height = item.height;
    const candidates = [];
    const add = (left, top) => candidates.push({ id: item.id, width, height,
      left: clamp(left, bounds.left, bounds.right - width),
      top: clamp(top, bounds.top, bounds.bottom - height) });
    for (let row = 0; row < 8; row++) {
      for (const top of [item.y - height - 9 - row * (height + 5), item.y + 9 + row * (height + 5)]) {
        add(item.x + 9, top); add(item.x - width - 9, top);
      }
    }
    // Dense ties may need a nearby free row rather than an overlapping label.
    for (let top = bounds.top; top + height <= bounds.bottom; top += height + 5) {
      add(item.x - width / 2, top); add(bounds.left, top); add(bounds.right - width, top);
    }
    const penalty = box => placed.filter(other => overlap(box, other)).length * 1e9
      + markers.filter(marker => overlap(box, marker, 1)).length * 1e6
      + (clamp(item.x, box.left, box.left + width) - item.x) ** 2
      + (clamp(item.y, box.top, box.top + height) - item.y) ** 2;
    candidates.sort((a, b) => penalty(a) - penalty(b));
    placed.push(candidates[0]);
  }
  return placed;
}
