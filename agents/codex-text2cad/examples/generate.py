"""No-model integration fixture: real cadgen/build123d → STEP + static/hinged USD."""

import argparse
from pathlib import Path

import build123d as bd
from cadgen import model

from cad_agent.usd_export import Part, Hinge, export_asset
from cad_agent.workflow import verify_generation


@model
def base_plate():
    shape = bd.Box(100, 60, 10) - bd.Cylinder(5, 10)
    shape.label = "base"
    return shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    # Actual engine-generated shape, not a synthetic receipt or import-only test.
    base = base_plate().shape
    for kind in ("static", "hinge"):
        output = args.directory.resolve() / kind / "submission/asset.usd"
        evidence = args.directory.resolve() / kind / "evidence"
        articulated = kind == "hinge"
        parts = [Part("base", base, color=(0.8, 0.1, 0.2), rigid=articulated,
                      kinematic=articulated, mass_kg=1),
                 Part("lid", bd.Box(80, 40, 4), translation_mm=(0, 20, 30),
                      rotation_wxyz=(0.7071067811865476, 0, 0, 0.7071067811865476),
                      color=(0.1, 0.2, 0.9), rigid=articulated, mass_kg=0.2)]
        hinges = [Hinge("lid_hinge", "base", "lid", (0, 20, 5), "X", (-10, 110))] if articulated else []
        export_asset(parts, output, source=__file__, evidence_dir=evidence, hinges=hinges)
        audit = verify_generation(evidence / "generation.json", output)
        assert audit["status"] == "artifact_chain_verified"
        print(kind, audit)


if __name__ == "__main__":
    main()
