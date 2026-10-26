"""Name a public instruction from a caller-supplied reviewed object label.

The capture importer must retain the original raw bytes before calling this pure
text helper. Its result is a derived public brief and needs its own file hash;
it must never be labeled with the original instruction's frozen checksum.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata


_GENERIC_OPENING = "Reconstruct the asset shown in observation.png as a self-contained USD package."
_NAMED_OPENING = re.compile(r"Reconstruct this [^\r\n]+ from the input image as a self-contained USD package\.")
_DIMENSIONS = re.compile(r"^Nominal dimensions in canonical metres XYZ:[ \t]*(.+?)[ \t]*\r?$", re.MULTILINE)


def _checked_dimensions(dimensions):
    if (not isinstance(dimensions, list) or len(dimensions) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 1e-9 <= value <= 1e6 for value in dimensions)):
        raise ValueError("dimensions must contain three positive finite canonical metre XYZ values")
    return dimensions


def named_instruction(original: str, object_name: str, dimensions: list | None = None) -> str:
    """Replace only a known generic/generated opening and preserve its full body.

    Unknown opening paragraphs remain intact after the new named opening. The
    reviewed name is mandatory; no asset name is inferred from the original.
    Existing dimensions and line endings are retained. Optional dimensions fill
    a missing canonical XYZ line; conflicting existing values are rejected.
    Reapplying the same reviewed name and dimensions is idempotent.
    """
    if not isinstance(original, str) or not original.strip():
        raise ValueError("original public instruction must be nonempty text")
    if (not isinstance(object_name, str) or not 1 <= len(object_name) <= 160
            or object_name != object_name.strip() or not any(char.isalpha() for char in object_name)
            or any(unicodedata.category(char).startswith("C") or char in "<>[]`\\*#|_"
                   for char in object_name)):
        raise ValueError("object_name must be a reviewed readable single-line label without markup or controls")
    insert_dimensions = False
    if dimensions is not None:
        _checked_dimensions(dimensions)
        existing = list(_DIMENSIONS.finditer(original))
        for match in existing:
            try:
                declared = _checked_dimensions(json.loads(match.group(1)))
            except (ValueError, TypeError) as error:
                raise ValueError("existing dimensions cannot be verified without rewriting the public brief") from error
            if declared != dimensions:
                raise ValueError("provided dimensions conflict with the existing public dimensions")
        insert_dimensions = not existing
    newline = "\r\n" if "\r\n" in original else "\n"
    opening = f"Reconstruct this {object_name} from the input image as a self-contained USD package."
    first = re.match(r"([^\r\n]+)(\r?\n\r?\n|$)", original)
    if first and (first.group(1) == _GENERIC_OPENING or _NAMED_OPENING.fullmatch(first.group(1))):
        separator, body = first.group(2), original[first.end():]
    else:
        separator, body = newline * 2, original
    if insert_dimensions:
        body = "Nominal dimensions in canonical metres XYZ: " + json.dumps(dimensions) + newline * 2 + body
        separator = separator or newline * 2
    return opening + separator + body
