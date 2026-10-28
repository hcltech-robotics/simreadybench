Reconstruct this oily-waste can from the single frozen observation as a self-contained USD package.

Nominal dimensions in canonical metres XYZ: [0.4926615161821246, 0.48329901695251465, 0.6464274525665711]
Stable task ID: `nv100_04ed00317ccc5a90`

![Frozen task observation](https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/c56694fe51cda7833bea4a0e4b6e6ae255f72182/benchmark/nv100/v2/tasks/nv100_04ed00317ccc5a90/public/observation.png)

[Download the frozen observation](https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/c56694fe51cda7833bea4a0e4b6e6ae255f72182/benchmark/nv100/v2/tasks/nv100_04ed00317ccc5a90/public/observation.png) · [Packaged image](observation.png)

The packaged observation is /task/observation.png.
Read /task/instruction.md for the task-specific public instructions.
Save the package under /output/submission, with its entrypoint at asset.usd.
Do not use Blender for authoring; Blender is used only by the verifier renderer.

Up to five formal feedback calls are available. Use a fresh request ID for each
changed candidate; retry the same request ID after a transport interruption:

    python -m simready_eval.feedback validate_candidate --package /output/submission --request-id attempt-1 --url http://feedback:8080

HTTP 503 feedback_busy is retryable after Retry-After with the same request ID;
the busy request has not been admitted and does not consume a feedback call.

Final evaluation independently measures formal, visual, and kinematic fidelity.

## Trajectory and asset provenance

Each attempt must include a complete, reviewable runner-recorded trajectory of
agent tool calls and results. Do not remove or conceal calls or results.

Retrieving or importing any external OpenUSD asset is an automatic protocol
failure, whether or not it is a benchmark reference. This includes USD, USDA,
USDC and USDZ files, and USD assets contained in downloaded archives.

You may author, load, render and validate your own USD locally and use the
task-provided inputs. Ordinary software-tool installation is allowed; it does
not permit importing externally supplied USD assets into the reconstruction.

Protocol validity is recorded separately from reconstruction scores. A missing
or unreviewable trajectory is unverified, not a clean integrity pass. Ambiguous
cases require review.

## Frozen public instructions

Reconstruct this oily-waste can from the input image as a self-contained USD package.

Nominal dimensions in canonical metres XYZ: [0.4926615161821246, 0.48329901695251465, 0.6464274525665711]

Preserve the visible geometry, materials, rigid bodies, joints and articulation. Write a top-level asset.usd and include all relative dependencies in your submission.

Submit only USD and its data dependencies: at most 1 GiB and 10000 files/directories combined. Keep authoring programs outside the submission; links, executable files and external dependencies are not accepted.

Do not use Blender for authoring; Blender is used only by the verifier renderer.
