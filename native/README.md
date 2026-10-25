# Native SimReadyBench evaluator

This independently installable package contains no Harbor harness, private runner,
model client or Orichalcum application dependency. Python 3.12, NumPy 2.5.4,
Pillow 13.0.0 and OpenUSD 26.8 are pinned. The optional fetch and validator
dependencies are separate extras. Code retains the accompanying MIT license;
NVIDIA assets retain their own licenses and provenance.

Blender 4.3.2 is used **only by the verifier renderer**. Blender authoring remains
prohibited. The actor recipe installs neither Blender nor NVIDIA validation SDKs.

## Reference preparation and rendering

Install the clean wheel in an isolated Python 3.12 environment. Use the delivered
dependency lock for an exact resolved environment. Install public Blender 4.3.2
on the rendering host; it is a system dependency, not part of the wheel. For a
nested entrypoint, pass `--package-root /qualified/closure` so sibling textures
remain constrained to the caller-qualified closure. Preparation propagates the
qualified copied reference root automatically.

```sh
python -m simready_eval.render --source /closure/asset.usd \
  --cameras cameras.json --output /fresh/render-output --timeout 180
```

The camera JSON schema is `simready-cameras-v1`, with `resolution: [512,512]`,
a separate `observation` camera, `observed_camera_id` naming it, and exactly four
comparison `views`. Each camera provides `id`,
`projection`, `camera_to_world`, `clip_start`, `clip_end`, and either
`orthographic_scale` or `focal_length_mm` plus `sensor_width_mm`.
Matrices are rows acting on homogeneous column vectors, in metres/Z-up. Their
columns are camera right/up/backward/position; the camera looks along local -Z.
All framing is frozen from the reference, never fitted to the candidate.

The default output is four comparison `<id>.png` RGB files and four
`<id>-mask.png` coverage masks. Run the same command with `--role observation`
for the separate fifth camera, and use that genuinely rendered RGB as
`public/observation.png`. `--role all` renders all five genuinely distinct frozen
cameras in one extraction and Blender job. Masks are independent of material opacity.
The command prints a JSON receipt and exits nonzero for unsupported or failed
measurements. Unsupported bound materials remain unknown. Unbound surfaces use
the explicitly recorded display/default policy below.

To generate the complete local task payload and eligibility/checksum inventory:

```sh
python -m simready_eval.prepare --source /closure/asset.usd \
  --package-root /closure --output /fresh/task-payload --task-id example \
  --cameras cameras.json --timeout 180
```

`package-root` must be an already qualified single-task dependency closure.
Nested original entrypoints are supported. Preserve relative dependencies and licenses; never
point it at a source collection or working repository. Omitting `--cameras`
freezes a separate observation plus isometric/front/side/top comparison rig from reference bounds.
The helper writes public observation/instructions, reference files, four
comparison RGBs/masks, cameras, renderer identity and frozen eligibility.
It retains the existing public nominal dimensions in canonical metres XYZ,
using `metadata.nominal_dimensions_m` when supplied and otherwise the full
visible source bounds. The one-image rule governs image access; it does not
remove this existing text input. Submission limits are 256 MiB and 10000
files/directories combined, with only USD and its confined data dependencies.
Final scoring verifies the recipe, cameras, bridge code and Blender binary
against that eligibility before touching the candidate. Changing the renderer
requires a newly qualified binding even when existing pixels remain identical.
It writes no HF revision into self-hashed content: after the dataset is frozen,
the release owner binds the output file checksums to that immutable HF commit.

## Scores and support

Formal, visual and kinematic fidelity are independent percentages. Formal uses
the genuine pinned NVIDIA 46 mandatory requirements with the original 95/5
completion rule; 51 optional requirements remain diagnostic. Kinematics retains
the existing joint-motion and moving/fixed group metric.
Visual is `sqrt(G * mean(R1,R2,R3,R4))`, where G is full 3D surface correspondence
and each R is foreground-only RGB SSIM at a frozen camera. Authored material
correspondence remains diagnostic with property-level measurement coverage.
Opacity-only introspection cannot certify complete material appearance.

Qualified rendering supports effective universal Preview Surface colour,
roughness, metallic, opacity, emissive colour, positive literal IOR, tangent
normal textures and UVTexture/PrimvarReader/Transform2d networks. UVTexture uses
authored output channels, sRGB/raw colour spaces, scale/bias and fallback values.
A fileless constant fallback requires neither texture coordinates nor a UV
primvar.
UDIM inputs use native tiled Blender images with their original tile coordinates;
missing tiles use the authored/default texture fallback. Both indexed and
unindexed face-varying UVs retain their original corners. An unused MDL graph does not
invalidate a supported effective universal material. MDL-only materials,
displacement, subdivision surfaces and unsupported active advanced Preview
Surface features remain explicit unknowns. Unequal-radius
Capsule_1 is also unqualified. Do not count an unknown receipt as a clean render.

An unbound surface uses authored `displayColor` and `displayOpacity`, retaining
per-corner interpolation. If absent, the published renderer defaults are linear
RGB `(0.18,0.18,0.18)`, roughness `0.5`, metallic `0`, opacity `1` and IOR `1.5`.
Receipts retain `source_material_status=unbound` and material correspondence
`not_applicable`; these renderer defaults are not attributed to the source
material. A bound unsupported shader never receives this fallback.

The renderer bounds source extraction at four million triangles and 1 GiB of
surface storage, and decoded textures at 1 GiB total, 16 megapixels per image.
Exceeding a bound returns an explicit unknown resource receipt. These limits
do not establish a whole-dataset runtime or memory qualification.

The final verifier bounds the render worker at 12 GiB of address space inside
its 16 GiB container; other workers keep their own bounds and the overall
600-second deadline remains unchanged. Run large preparation jobs one at a time
on a 16 GiB host. This draft is usable for qualified smaller assets. The
1,931,457-triangle locker produced only partial comparison images before the
180-second Blender limit and is not a qualified workload for this draft.
Oversized assets must be filtered during roster qualification, without
decimation or relaxed scoring. A four-million-triangle extraction cap is not a
claim that every selected asset has passed runtime qualification.

Finite nonzero authored normals retain their original corners and are normalized.
Zero authored normal vectors are preserved as zero, invoking Blender
`Mesh.normals_split_custom_set` automatic normals for those corners only; the
receipt records their count. Nonfinite normals remain unknown. Repeated-position
quads retain the observer's original-corner fan and its approximation diagnostic;
zero-area triangles have no surface sampling weight. No vertices are welded or
snapped.
Missing PhysX native bindings can make specific validator checks unknown.

## Native Harbor export and migration

`python -m simready_eval.export --release release.json --wheel evaluator.whl
--output /fresh/export` creates only an allowlisted standalone task distribution.
It requires exactly 100 tasks unless `--fixture` is explicit, and never uploads.
HF `benchmark/` assets are fetched role-selectively at a full commit plus file
checksums during setup. Offline scoring reads only the declared submission and
frozen fixtures. The feedback sidecar persists five calls and runs candidate
parsing under an unprivileged UID separated from its control ledger.
Exported recipes consume role-specific hash-enforced dependency locks for Linux
x86_64 / CPython 3.12, then install the checksum-inventoried native wheel without
resolving more dependencies. Actor and verifier defaults remain 231000 and 600
seconds respectively; repetitions and model turn/token limits remain runner
settings. A Docker host supporting Harbor's NFT FIB network enforcement is
required for the declared allowlist and offline phase policies.

Keep results in the same HF dataset under `results/`. The local migration tool
is `python -m simready_bench.results_migration`; its default is dry run, and
`--apply` only makes local additive copies with checksum/rollback evidence.
No live migration, bucket creation or publication occurs in these commands.
`python -m simready_eval.mirror` produces a local per-set mirror manifest;
mutable bucket copies never replace the pinned dataset as version authority.
