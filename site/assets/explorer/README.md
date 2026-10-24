# NV25 interactive geometry comparison

This viewer contains five reference objects and their first reconstruction from
four recorded benchmark conditions. Every condition uses zero-based repetition
`0`; examples are not selected by score. The manifest pins the public result
dataset and NVIDIA reference revisions, each original results-file hash, the
actual trial condition identifier, source USD hashes and exported GLB hashes.

The three displayed scores are the original formal, visual and kinematic fidelity
measurements. Values retain their recorded precision in the manifest. A measured
zero remains zero; `not_applicable` remains distinct from an unavailable value.
Reference objects are not assigned reconstruction scores.

Cost is recorded inference cost in USD, with its source classification retained.
Compute and total cost are unavailable in this selection. Token counts retain
the recorded input, output and total usage, including repeated context. Time is
elapsed reconstruction time in seconds; model-request time is also retained when
available. Missing usage is `null`, never an inferred zero. Model reasoning,
harness and output limits are shown because these experimental settings differ.

The GLBs are static, neutral-shaded geometry projections of the recorded USD
assets. They preserve authored geometry scale after unit conversion and apply
the measured whole-object alignment when available. They omit source textures,
materials, physics schemas, animation and authoring metadata. They therefore
support shape inspection; the visual score still evaluates the original asset,
including its appearance. An invalid source mesh produces an unavailable preview
while its recorded scores and usage remain visible. Analytic primitives are tessellated with 64 angular segments; subdivision meshes show their authored control surfaces. Nonplanar polygons use deterministic fan triangulation. Per-file counts, alignment and conversion notes are recorded in [geometry-provenance.json](geometry-provenance.json). Browser conversion used OpenUSD 26.11; the benchmark evaluator was unchanged.

Orichalcum attribution is limited to its source pin. Its contents and authoring
traces are not part of this publication.

## Mesh delta point cloud

`delta-manifest.json` describes the optional mesh-difference display. Each valid
reference/reconstruction pair has 12,000 deterministic samples from each mesh,
drawn uniformly by triangle surface area. A point's colour represents its
unsigned Euclidean distance to the closest triangle surface of the other mesh.
Both directions are included so missing and additional surfaces remain visible.
Distances are calculated against complete triangle surfaces, not against the
other point samples or vertices. Reference sample positions are reused across
all reconstructions of each object.

The clouds retain the GLBs' common metre/Y-up coordinates, including any recorded
rigid evaluation alignment. No additional centring, registration or scale fitting
is applied. Each binary file contains little-endian float32 rows
`x, y, z, distance_m`, with reference rows first. The manifest records their count,
split index, hashes, source GLB hashes, reference bounding-box diagonal and sampled
maximum/95th-percentile distance. These are geometry display diagnostics, not new
benchmark fidelity scores or continuous Hausdorff measurements. Finite sampling
can miss small features; the GLBs' tessellation and omitted-material limitations
also apply. The unavailable reconstruction has no fabricated cloud.

## Attribution and licence

Reference geometry is derived from NVIDIA SimReady Assets, revision
`f4860aa0c75607adb50a78bf96e77510bffc040f`, under the accompanying
[Creative Commons Attribution 4.0 licence](NVIDIA-ASSET-LICENSE.txt). NVIDIA retains ownership of the
reference assets. The source entrypoint and revision for each object are linked
in `manifest.json`. The licence text is copied byte-for-byte from the pinned
reference package.

Generated candidates and their measurements are attributed to the named model,
harness and run in the
[SimReadyBench results dataset](https://huggingface.co/datasets/hcltech-robotics/simreadybench).
Repository code remains under the repository's MIT licence.

## Rebuild and verify

Prepare the allowlisted discovery snapshot, public/private export job manifests
and export receipts in a build-data directory. These inputs are not served by the
site. Run from the repository root:

```bash
python3 scripts/build_viewer_manifest.py /path/to/build-data site/assets/explorer --require-geometry
python3 scripts/build_viewer_manifest.py /path/to/build-data site/assets/explorer --check --require-geometry
```

The checker compares all selected score and usage values against their exact
source trial and verifies each available GLB against its export receipt. It
rejects missing, duplicated or changed trial identities and changed geometry.

To rebuild the delta clouds from the already published GLBs, use an isolated
environment with `numpy==2.4.1`, `scipy==1.17.1`, `trimesh==4.11.5` and
`rtree==1.4.1`:

```bash
python3 scripts/build_viewer_delta.py site
python3 scripts/build_viewer_delta.py site --check
PYTHONPATH=src python3 -m pytest tests/test_viewer_delta.py -q
```

The check recomputes every cloud and compares the exact bytes and metadata. The
tests also cover identical meshes, translation, scale differences, both distance
directions, triangle-interior projections and the published sample distances.
