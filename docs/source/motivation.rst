Why MLFF Robustness Matters
===========================

The opportunity
---------------

Density functional theory provides a physically grounded route to calculating
energies, forces, and other atomic properties, but its computational cost
limits the length and scale of many simulations. MLFFs learn from reference
electronic-structure calculations and can evaluate atomic configurations much
more quickly.

This speed makes MLFFs attractive for relaxation, molecular dynamics,
materials screening, and deformation studies.

The risk
--------

An MLFF is most trustworthy near configurations represented by its training
distribution. During a simulation, atoms may enter unfamiliar arrangements
because of thermal motion, strain, defects, interfaces, optimization steps, or
numerical instability.

A small force error can alter the next atomic update. That update changes the
next prediction, allowing an initially small error to redirect an entire
relaxation trajectory.

Research question
-----------------

This work tests whether small, deliberately challenging coordinate
perturbations expose model-specific failure modes.

The central question is not simply whether a model predicts a large force.
It is whether the model can guide the perturbed structure back toward the
same physically meaningful configuration reached before the attack.

Why this helps other researchers
--------------------------------

A reproducible robustness benchmark can help researchers:

* identify perturbation magnitudes at which predictions become unstable;
* compare architectures and training distributions under matched conditions;
* distinguish recoverable perturbations from persistent structural changes;
* detect catastrophic outliers hidden by average-error metrics;
* select models and numerical precision appropriate for a simulation;
* establish validation checks before deploying an MLFF in a new domain.

The benchmark therefore complements conventional test-set accuracy. It asks
how prediction errors behave inside an iterative physical workflow, where each
prediction influences the next configuration.
