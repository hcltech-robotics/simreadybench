"""Pinned Linux x86_64 / CPython 3.12 runtime wheel locks, without the native wheel.

Hashes cover the public wheels resolved for the supported native container.
The actor uses core+fetch; feedback uses core+validator; verifier uses their union.
Build tooling is separately pinned by pyproject.toml. This module performs no I/O
except when its explicit local-file CLI is invoked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

LOCKED_REQUIREMENTS = """anyio==4.15.1 --hash=sha256:6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101
certifi==2026.10.23 --hash=sha256:e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1
click==8.5.0 --hash=sha256:255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360
filelock==4.2.4 --hash=sha256:301241438c57936ef93f4243a94f2559022268bfcf9b375a9a72eb0f8db8358e
fsspec==2026.9.0 --hash=sha256:8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f
h11==0.16.0 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86
hf-xet==1.7.0 --hash=sha256:2814a6e999d13464c4d679b788cc5d784eb5a4edfc638a31f10e9a11ab531ef8
httpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55
httpx==0.28.1 --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad
huggingface_hub==1.33.0 --hash=sha256:04e434b06e100eddbce9a6e817d72693a7884b10a79bd67ab48080d5c07eb899
idna==3.20 --hash=sha256:ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c
Jinja2==3.1.6 --hash=sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67
markdown-it-py==4.2.0 --hash=sha256:9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a
MarkupSafe==3.0.4 --hash=sha256:8e124f974786f831d6043728e38296969d3579db8896fe004682f5758e613581
MaterialX==1.39.5 --hash=sha256:897c5a5df9068cc50b966e4bcda63fb71d03b79c453dcf5b62b3d16a5b3369ce
mdurl==0.1.2 --hash=sha256:84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8
numpy==2.5.4 --hash=sha256:fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a
packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
physx-usd-schemas==25.11.2.72696984 --hash=sha256:44a8671f8d4f0fbdf07c9436b3827f96156e4b094b646d0903937f275eb84f0e
pillow==13.0.0 --hash=sha256:42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca
PyYAML==6.0.3 --hash=sha256:ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc
simready-foundation-tier-core==2026.8.0 --hash=sha256:f3bfd5905fdf1b8315ee2f7104404c7baa56f970812695d92aed636839477987
simready-validate==2026.8.0 --hash=sha256:6313d463e537246ddccbd05d8568cc0d1df71df9cd91b29b80d7d4b3026808c2
tqdm==4.70.1 --hash=sha256:c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73
typing_extensions==4.16.0 --hash=sha256:481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8
usd-core==26.8 --hash=sha256:204fb1ef1fc977b827c491f4e9fa39b57518504ed0cacfe12e026e95123cacf1
usd-profiles-nvidia==1.22.0 --hash=sha256:2aafc38c7d77d4a03578972cf44b0e42f768d3403a6a449cd393c46cda724131
usd-validation-nvidia==1.22.0 --hash=sha256:5bf8756a1642c627561859f46ba04b898f79ff0b09a494aa468d027821a2b5c4
"""

ACTOR = frozenset("anyio certifi click filelock fsspec h11 hf-xet httpcore httpx huggingface-hub idna numpy packaging pillow pyyaml tqdm typing-extensions usd-core".split())
FEEDBACK = frozenset("jinja2 markdown-it-py markupsafe materialx mdurl numpy physx-usd-schemas pillow simready-foundation-tier-core simready-validate usd-core usd-profiles-nvidia usd-validation-nvidia".split())


def runtime_lock(role):
    """Return a complete hash-enforced runtime lock for one supported role."""
    roles = {"actor": ACTOR, "feedback": FEEDBACK, "verifier": ACTOR | FEEDBACK}
    if role not in roles:
        raise ValueError("role must be actor, feedback or verifier")
    lines = [line for line in LOCKED_REQUIREMENTS.splitlines()
             if line.split("==", 1)[0].lower().replace("_", "-") in roles[role]]
    return "# Linux x86_64 / CPython 3.12; native wheel installed separately with --no-deps.\n" + "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("actor", "feedback", "verifier"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(runtime_lock(args.role))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
