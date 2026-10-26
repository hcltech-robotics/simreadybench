"""Shared bounded artifact sizes; independent of renderer/time score semantics."""

CANDIDATE_BYTES = 1024**3
REFERENCE_BYTES = 2 * 1024**3
MAX_ENTRIES = 10_000
# Plain TAR/PAX metadata is not candidate data. This conservative bound covers
# 10,000 names of at most 4096 Unicode characters (up to four UTF-8 bytes each),
# headers, padding and the ordinary client-generated PAX metadata.
TAR_OVERHEAD_BYTES = 256 * 1024**2
TAR_BYTES = CANDIDATE_BYTES + TAR_OVERHEAD_BYTES
FEEDBACK_SCRATCH_BYTES = 2 * 1024**3
CANDIDATE_LABEL = "1 GiB"
