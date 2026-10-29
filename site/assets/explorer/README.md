# Interactive geometry comparison

The gallery displays reference objects and matched reconstruction examples from
the benchmark results. `manifest.json` selects each object, model configuration
and repetition explicitly. Examples use the same repetition across conditions;
they are not selected by score. Each candidate links to its source trial, render
and USD package.

The displayed formal, visual and kinematic scores are recorded benchmark
measurements. The manifest preserves their precision and distinguishes measured
zero, unavailable values and `not_applicable`. Reference objects have no
reconstruction score. Usage fields retain their recorded units and missing
values; preview generation does not recalculate scores or costs.

## Geometry previews

GLBs are neutral-shaded geometry projections of the submitted and reference USDs.
They preserve physical scale after unit conversion and apply the recorded
whole-object alignment when available. Textures, materials, physics schemas,
animation and authoring metadata are omitted. The preview supports shape
inspection; it is not a replacement for the original USD or its visual score.

Analytic primitives use 64 angular segments. Subdivision meshes show their
authored control surfaces, and nonplanar polygons use deterministic fan
triangulation. Conversion counts, alignment and source/GLB checksums are recorded
in [geometry-provenance.json](geometry-provenance.json). An unavailable preview
does not change the recorded score.

## Mesh delta point cloud

`delta-manifest.json` describes the mesh-difference display. Each available pair
has 12,000 deterministic area-weighted surface samples from each mesh. A point's
colour represents its unsigned Euclidean distance to the closest triangle
surface of the other mesh. Both directions are included to show missing and
additional surfaces. Reference sample positions are shared across the
reconstructions of each object.

Clouds retain the GLBs' common metre/Y-up coordinates and recorded rigid
alignment. There is no additional registration or scale fitting. Binary rows are
little-endian float32 `x, y, z, distance_m`, with reference samples first. The
manifest records counts, the split index, checksums, source GLB hashes, reference
diagonal and sampled maximum/95th-percentile distances.

These are finite-sample display diagnostics, not new fidelity scores or
continuous Hausdorff measurements. Small features can be missed, and the GLB
tessellation and omitted-material limitations still apply.

## Build and check

Build tools use local source files and matching export records. Copy the gallery
manifests and `geometry-provenance.json` into a separate build directory. Download
the selected hosted GLBs, verify their checksums and provide local relative paths
in that directory's `assets/explorer/manifest.json`. Keep the original scale,
alignment and source identities.

The delta builder uses `numpy==2.4.1`, `scipy==1.17.1`, `trimesh==4.11.5` and
`rtree==1.4.1`:

```sh
python scripts/build_viewer_delta.py ./viewer-build
python scripts/build_viewer_delta.py ./viewer-build --check
PYTHONPATH=src python -m pytest tests/test_viewer_delta.py -q
```

The check recomputes clouds and compares bytes and metadata. The tests cover
translation, scale, both distance directions and triangle-interior projections.
See the [site guide](../../README.md) for browser rendering checks.

## Attribution and licence

Reference geometry derives from NVIDIA SimReady Assets revision
`f4860aa0c75607adb50a78bf96e77510bffc040f`, under the accompanying
[Creative Commons Attribution 4.0 licence](NVIDIA-ASSET-LICENSE.txt). NVIDIA
retains ownership of the reference assets. `manifest.json` links each source
entry point and revision; the licence text is preserved from the source package.

Generated candidates and measurements are attributed to their model, harness
and run in the [results dataset](https://huggingface.co/datasets/hcltech-robotics/simreadybench).
Repository code remains under the repository's MIT licence.
