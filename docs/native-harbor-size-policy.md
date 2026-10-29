# Native package and resource limits

NV100 submissions have a **1 GiB expanded-package limit** and at most **10,000
files and directories**. The limit includes dependencies and expanded archives.
Individual candidate files have the same byte bound. The evaluator checks paths,
links, archive expansion and dependencies before composing the USD.

The native evaluator shares this limit across package inspection, final
submission capture, formal validation and feedback uploads. The constants are
defined in [`native/src/simready_eval/limits.py`](../native/src/simready_eval/limits.py).

## Transport and reference acquisition

Feedback uploads use a separately bounded **1.25 GiB plain-TAR envelope**. Its
additional space covers headers and metadata; it does not increase the expanded
candidate allowance. Feedback uses **2 GiB of scratch space** and admits one
upload/evaluation at a time. Busy requests receive HTTP 503 with `Retry-After`
before consuming a feedback call.

Trusted reference acquisition has a separate **2 GiB dependency-closure limit**.
Reference packages may retain original textures and geometry that would exceed
a smaller candidate allowance. Reference and candidate limits serve different
roles and must be recorded with the evaluator version.

## Runtime limits

Package admission does not establish that evaluation will complete. Geometry,
decoded-image, worker-memory and time limits apply independently. The final
verifier has a **600-second total deadline**; conversion and rendering share the
time remaining in that deadline. Blender has a **180-second subprocess limit**.
The renderer's address-space bound is **12 GiB** within the **16 GiB verifier
container**.

Qualify reference workloads with the renderer and resource settings used by the
release. Texture reduction or geometry simplification changes the asset and can
change its visual score. Resource exhaustion remains an unavailable measurement,
rather than a model-quality zero. See the [native evaluator guide](../native/README.md)
for material support and allocation limits.
