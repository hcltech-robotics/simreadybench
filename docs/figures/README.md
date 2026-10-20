# Score diagrams

The three diagrams show how the implemented subscores combine. Their Mermaid sources embed the shared Geist Sans styling and use LaTeX for equations.

| Diagram | Rendered files | Editable source |
| --- | --- | --- |
| Formal fidelity | [PNG](../../site/assets/formal-fidelity.png)/[SVG](../../site/assets/formal-fidelity.svg) | [Mermaid](formal-fidelity.mmd) |
| Visual fidelity | [PNG](../../site/assets/visual-fidelity.png)/[SVG](../../site/assets/visual-fidelity.svg) | [Mermaid](visual-fidelity.mmd) |
| Kinematic fidelity | [PNG](../../site/assets/kinematic-fidelity.png)/[SVG](../../site/assets/kinematic-fidelity.svg) | [Mermaid](kinematic-fidelity.mmd) |

Formal fidelity follows the [mandatory requirement scorer](../../src/simready_bench/engine/scoring.py#L419-L494). Visual fidelity follows the [surface score](../../src/simready_bench/fidelity.py#L217-L233), [per-sample appearance aggregation](../../src/simready_bench/fidelity.py#L400-L432) and [composite geometric mean](../../src/simready_bench/composites.py#L102-L151). Kinematic fidelity follows the [joint quality factors](../../src/simready_bench/kinematics.py#L165-L225), [group recovery](../../src/simready_bench/kinematics.py#L256-L300) and [composite geometric mean](../../src/simready_bench/kinematics.py#L417-L464).

The [evaluation guide](../evaluation.md) defines the equations and treatment of unmeasured and inapplicable criteria.
