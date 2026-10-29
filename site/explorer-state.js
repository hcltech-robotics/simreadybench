/** A saved HTML document cannot preserve a live WebGL drawing buffer. */
export function resetExplorerPreview(root, { interactive = true } = {}) {
  root.dataset.state = "idle";
  const byId = name => root.querySelector(`#explorer-${name}`);
  byId("canvas").replaceChildren();
  for (const name of ["object", "variant", "mode-swipe", "mode-delta", "reset", "swipe", "threshold"]) {
    byId(name).disabled = true;
  }
  byId("stage").setAttribute("aria-busy", "false");
  byId("stage").dataset.mode = "swipe";
  byId("swipe").hidden = true;
  byId("divider").hidden = true;
  byId("delta-controls").hidden = true;
  byId("readout").hidden = true;
  const message = interactive ? "3D comparison loads when visible." : "Enable JavaScript to use the 3D comparison.";
  byId("status").textContent = message;
  byId("preview-message").textContent = message;
  byId("preview-message").hidden = false;
}
