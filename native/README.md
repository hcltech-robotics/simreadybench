# Native SimReadyBench evaluator

This independently installable package provides formal validation, geometry,
appearance and kinematic evaluation, reference rendering and Harbor task export.
Python 3.12, NumPy 2.5.4, Pillow 13.0.0 and OpenUSD 26.8 are pinned. Fetch and
validator dependencies are separate extras. Code uses the accompanying MIT
license; NVIDIA assets retain their own licenses and attribution.

Blender 4.3.2 is used **only by the verifier renderer**. Blender authoring remains
prohibited. The actor recipe installs neither Blender nor NVIDIA validation SDKs.

NV100 v2 evaluates candidates with Cycles CPU, 256 fixed samples, seed 1729,
four CPU threads, no adaptive sampling and no denoising. Its reference images
use the qualified Blender 4.5.11 LTS OptiX producer. The reference receipt retains
the actual producer's recipe, binary and bridge identities; eligibility also
pins the CPU evaluator and shared scientific settings. Formal, visual and
kinematic scores are reported separately.

## Import existing NV100 v2 reference captures

Run this command in the pinned CPU verifier environment (Blender 4.3.2 and the
2.0.0 evaluator wheel). It imports five completed RGB/mask pairs without rendering:

```sh
python -m simready_eval.capture --task /frozen-v1/tasks/TASK_ID \
  --receipt /captures/TASK_ID/diagnostic-receipt.json \
  --capture-root /captures/TASK_ID --output /new-v2/tasks/TASK_ID
```

The input task must contain its original `asset-inventory.json`, `evaluator.json`,
public inputs and complete reference/comparison closure. Only the frozen v1
producer identity is accepted as the baseline. New captures must match its
complete source scene and five frozen cameras, the approved GPU producer, and
their recorded image hashes. Original USD, textures, licenses, profile and
instructions are preserved. The raw producer receipt is retained unchanged;
derived eligibility binds it to the CPU evaluator. Failed imports leave no
new task output.
An existing `role-inventory.json` is validated against both original role trees;
the new sidecar is rebuilt with v2 paths and current image/eligibility hashes.

Store the new tasks under `benchmark/nv100/v2/tasks/` in a new immutable dataset
revision. Build their Harbor release using `simready-harbor-release-v2` and the
2.0.0 wheel. V1 remains under its existing revision and uses the 1.0.2 wheel.
Role manifests select only their own version, task and role.

The task observation uses a reproducible task/index-seeded viewing angle,
world Z-up without roll. The selector samples azimuth 0–360° and elevation
15–70°, projects all eight authoritative reference-bounds corners, and fits
both image dimensions with 15% margin and valid clipping. Its seed is separate
from the fixed Cycles noise seed. Nonzero candidate indices require a recorded
pixel-review reason; once chosen, the camera remains fixed for the task.

```sh
python -m simready_eval.observation --cameras original-cameras.json \
  --bounds original-provenance.json --task-id TASK_ID --candidate-index 0 \
  --output selected-cameras.json
```

To reuse an existing full GPU batch's four comparison pairs and independently
render only the new observation, preserve both actual raw receipts and import:

```sh
python -m simready_eval.capture --task /frozen-v1/tasks/TASK_ID \
  --receipt /comparison-batch/TASK_ID/diagnostic-receipt.json \
  --capture-root /comparison-batch/TASK_ID \
  --observation-receipt /observation-batch/TASK_ID/diagnostic-receipt.json \
  --observation-root /observation-batch/TASK_ID \
  --cameras /selected/TASK_ID/cameras.json --object-name 'reviewed object name' \
  --output /new-v2/tasks/TASK_ID
```

The importer proves the four comparison cameras are unchanged, verifies both
producer receipts against the original source scene, and records their distinct
full camera hashes. It retains the original public brief separately and writes
the reviewed object name into the new brief without dropping its constraints.
The final CPU evaluation continues to render only the four comparison views.

## Reference preparation and rendering

Install the evaluator wheel in an isolated Python 3.12 environment. Use the release's
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
remove this existing text input. Submission limits are 1 GiB and 10000
files/directories combined, with only USD and its confined data dependencies.
Final scoring verifies the recipe, cameras, bridge code and Blender binary
against that eligibility before touching the candidate. Changing the renderer
requires a newly qualified binding even when existing pixels remain identical.
It writes no HF revision into self-hashed content: after the dataset is frozen,
the release manifest binds the output file checksums to that immutable HF commit.

For a complete original package, `runtime_closure(source, package_root)` in
`simready_eval.prepare` returns an exact path/size/SHA-256 inventory of authored
dependencies and notices, plus excluded archive/thumbnail paths. It copies
nothing. Map those paths directly from the verified acquisition store when
publishing, and selectively materialize one bounded task before normal package
validation/preparation. A referenced archive is retained; an unrelated one is
not fetched into the verifier. The canonical HF source archive can retain all
original files. `public_instruction(dimensions)` emits the same public text
without another render.

Trusted reference preparation and final reference preflight allow a separately
bounded 2 GiB dependency closure. This permits inspecting original source
packages; it does not establish reference eligibility. Candidate files and
expanded archives are bounded at 1 GiB. A selected reference still needs a faithful candidate representation
within the declared submission limit and successful evaluation with headroom.
Texture reduction changes the candidate and can change its visual score.
Package admission and complete runtime qualification are separate checks.

Feedback accepts a bounded plain TAR envelope with separate metadata overhead;
the extracted package stays within 1 GiB. Its 2 GiB scratch and one in-flight
upload/evaluation bound memory use before body reading. Busy requests receive
retryable HTTP 503 with `Retry-After` and consume no feedback slot. Retry the
same request ID and candidate digest. The service permits five calls, retains
replay state across restarts and bounds responses at 8 MiB. Its ledger is
inaccessible to the actor.

## Scores and support

Formal, visual and kinematic fidelity are independent percentages. Formal uses
the pinned NVIDIA 46 mandatory requirements with the 95/5 completion rule;
51 optional requirements are scored separately. Kinematics measures joint motion
and recovery of moving and fixed constraint groups.
Visual is `sqrt(G * mean(R1,R2,R3,R4))`, where G is full 3D surface correspondence
and each R is foreground-only RGB SSIM at a frozen camera. Authored material
correspondence remains diagnostic with property-level measurement coverage.
Opacity-only introspection cannot certify complete material appearance.
The separate material-property sampler does not certify PNG16 values through
Pillow's 8-bit RGB conversion; those properties remain explicitly unmeasured.

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
surface storage. The texture bridge streams one lossless file at a time and
retains shared content-addressed descriptors instead of decoded texture banks.
Its 1 GiB pixel-buffer bound applies to peak decoding/export allocation, with
a separate 16-megapixel and 128 MiB encoded-file bound per image. Eight-bit
images retain the existing canonical RGBA PNG bytes; 16-bit PNGs retain their
original encoded bytes and native sample precision for Blender decoding. Raw
and sRGB interpretations remain separate even when content is shared. Receipts
report source bit depths, content sharing and bridge pixel-buffer estimates;
actual Blender allocations are measured separately under its address-space
bound. Exceeding a resource bound returns typed unknown. These limits do not
establish a whole-dataset runtime or memory qualification.

The final verifier bounds the render worker at 12 GiB of address space inside
its 16 GiB container. Conversion and rendering share the 600-second final
deadline, and Blender has a separate 180-second subprocess limit. Large textures,
complex geometry and concurrent workers can exceed these bounds even when the
package passes inspection. Qualify references with complete evaluation under the
release's resource settings, leaving headroom for candidate work.

A triangle-count limit or a successful render-only probe does not establish
full-evaluation eligibility. Preserve source geometry and textures during
reference preparation; choose compatible references rather than simplifying
them or relaxing scoring. A reference can match itself geometrically while
failing formal profile requirements, so retain its measured formal result.

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
The exporter acquires the actor observation and raw public brief against the
frozen HF manifest. It preserves the raw brief as `source-public-instruction.txt`
and creates named root/linked/runtime instructions containing the full original
body. The actor build first verifies the original frozen pair, removes that
staging tree, then installs exactly the observation and named instruction in
`/task`, verifying the derived wrapper against its separate `actor-runtime.json`
identity without HF access. Canonical tasks require an owner-reviewed `object_name` and public
`nominal_dimensions_m` (XYZ metres), separate from stable `id`. The root task
instruction opens "Reconstruct this <name> ..." and shows these dimensions;
task descriptions and the dataset index use the same label. The raw public
instruction bytes and original HF manifest/hash remain unchanged; the named
wrapper has its own explicit derived-content hash. Root instructions link to the packaged
image and the pinned public HF preview/download. An optional `--public-assets`
root mirrors the manifest HF paths for offline export; unrelated local files are
never included. Verifier `benchmark/` assets are fetched role-selectively at a full commit plus file
checksums during setup. Offline scoring reads only the declared submission and
frozen fixtures. The feedback sidecar persists five calls and runs candidate
parsing under an unprivileged UID separated from its control ledger.
Exported recipes consume role-specific hash-enforced dependency locks for Linux
x86_64 / CPython 3.12, then install the checksum-inventoried native wheel without
resolving more dependencies. Actor and verifier defaults remain 231000 and 600
seconds respectively; repetitions and model turn/token limits remain runner
settings. A Docker host supporting Harbor's NFT FIB network enforcement is
required for the declared allowlist and offline phase policies.

The publishing procedure is in [Publish NV100 task packages](tools/README.md).

`python -m simready_bench.results_migration` creates additive local copies of
historical result snapshots. It defaults to a dry run; `--apply` writes local
copies with source/destination checksums and rollback records. It does not upload
results to a live dataset.
`python -m simready_eval.mirror` produces a local per-set mirror manifest;
mutable bucket copies never replace the pinned dataset as version authority.
