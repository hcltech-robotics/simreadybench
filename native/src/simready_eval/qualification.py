"""One sealed reference-producer / candidate-evaluator pair for NV100 v2."""
from __future__ import annotations

from pathlib import Path
import shutil

from . import render

QUALIFICATION = 'simready-nv100-cycles256-optix4511-cpu432-adapter-v2'
HISTORICAL_QUALIFICATION = 'simready-nv100-cycles256-optix4511-cpu432-v1'
SCHEMA = 'simready-render-compatibility-v1'
GPU = {
    'recipe_sha256': '514dbf82a3fbf2ba24f8fb1462a54df6132eb895361f7424389b303034ff9d64',
    'bridge_code_sha256': '2e515052b919ac7eff777d2119cf9fc41e29d99aae0b922f70712572d580b419',
    'renderer_binary_sha256': 'dc72290ee8651c93c4a946c012c5f2a034946fd320e6c3ab214fa23181427428',
}
CPU_BINARY = '0dfe9af0f3e67643ecf15327fa5e14e7315f61d1a81afdcafe76a71b1918e2b4'
CPU_RECIPE = '6a5b4165228f56b4f07301ba91dcace7640f73d6bd938fa282834b8f762ff973'
CPU_BRIDGE = '808c59cea9f938d9b449f972186cc286846bb725a2164819877ee07d7100c924'
HISTORICAL_CPU_BRIDGE = 'f9b0e89cfa2ea149a11758f1bd6c84633821173b769d988bf9fffa44a867b4da'
EVIDENCE = {
    'producer_bundle_sha256': 'fa2047176e3cc2545cab8a74d5bbe8467877a9e70d98b6a28551b7c8a8798f83',
    'paired_calibration_sha256': 'd473a9bb9ad91eeb961d145bb20911b4d83e5d708832eb37f83dfed8fb21ca83',
    'cpu_native_qualification_sha256': '6c28b5a150f7ce72832ab473708e45e76f5da390eca80b884ad02683a328b523',
}
HISTORICAL_EVIDENCE = dict(EVIDENCE)
EVIDENCE.update({
    'adapter_cpu_parity_sha256': 'a170d9f2ec3c357f59aefb28bb307583359e6d06b24ec8cbe2c1bf15faf4dcd8',
    'bounded_subdivision_cpu_checks_sha256': '2577e63509eb65efacef9ea2b5426c5f08f126f08e3a65841013f30b8f8a23ea',
})
LEGACY = {
    'render_recipe_sha256': '0bd3c8bbdf84136880ec5d0efdda59938bea8fa9f7561cf8f811f2567b17fb9d',
    'render_bridge_code_sha256': '62a265f1f70a1106499a38d3d2a38acfbcfce8ccc98cdf391cca21b9c4ba47eb',
    'renderer_binary_sha256': CPU_BINARY,
}


def scientific_recipe(recipe):
    """Execution identities remain separate; every other setting must agree."""
    return {key: value for key, value in recipe.items()
            if key not in ('blender_version', 'device', 'threads')}


def evaluator_identity():
    if render._identity(render.RECIPE) != CPU_RECIPE:
        raise ValueError('Unqualified CPU evaluator recipe')
    binary = shutil.which('blender')
    if binary is None or render._hash(binary) != CPU_BINARY:
        raise ValueError('Unqualified CPU evaluator binary')
    bridge = render._identity([
                render._hash(render.__file__),
                render._hash(Path(render.__file__).with_name('render_worker.py'))])
    if bridge != CPU_BRIDGE:
        raise ValueError('Unqualified CPU evaluator bridge')
    return {'recipe_sha256': CPU_RECIPE, 'bridge_code_sha256': bridge,
            'renderer_binary_sha256': CPU_BINARY}


def validate_producer(receipt):
    if (receipt.get('schema') != 'simready-render-receipt-v1'
            or receipt.get('status') != 'measured'
            or receipt.get('variant') != 'raw'
            or receipt.get('alignment') != [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]):
        raise ValueError('Qualified raw GPU producer receipt required')
    if any(receipt.get(key) != value for key, value in GPU.items()):
        raise ValueError('Unqualified GPU producer identity')
    recipe = receipt.get('recipe', {})
    if (render._identity(recipe) != GPU['recipe_sha256']
            or scientific_recipe(recipe) != scientific_recipe(render.RECIPE)):
        raise ValueError('GPU producer scientific recipe mismatch')
    actual = receipt.get('renderer', {})
    if (actual.get('recipe') != recipe or actual.get('version') != '4.5.11 LTS'
            or actual.get('device') != 'OPTIX' or actual.get('threads') != 4):
        raise ValueError('GPU producer worker receipt mismatch')
    return dict(GPU)


def compatibility(receipt):
    return {'schema': SCHEMA, 'qualification': QUALIFICATION,
            'qualification_evidence': dict(EVIDENCE),
            'scientific_recipe_sha256': render._identity(scientific_recipe(render.RECIPE)),
            'reference_producer': validate_producer(receipt),
            'candidate_evaluator': evaluator_identity()}


def validate_compatibility(value, receipt):
    current = compatibility(receipt)  # Checks the running binary and bridge.
    if value == current:
        return current
    # Admit only the exact existing sealed producer pair. Its frozen pixels,
    # recipe and closure still pass the original checks. The running adapter
    # must be the separately qualified v2 bridge.
    historical = {**current, 'qualification': HISTORICAL_QUALIFICATION,
                  'qualification_evidence': dict(HISTORICAL_EVIDENCE),
                  'candidate_evaluator': {**current['candidate_evaluator'],
                                          'bridge_code_sha256': HISTORICAL_CPU_BRIDGE}}
    if value != historical:
        raise ValueError('Frozen producer/evaluator compatibility mismatch')
    return current
