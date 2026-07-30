Concepts and Terminology
========================

This page provides concise definitions of the main physical, computational,
and adversarial concepts used throughout the project.

Core concepts
-------------

Density functional theory
~~~~~~~~~~~~~~~~~~~~~~~~~

**Density functional theory (DFT)** is an electronic-structure method used to
calculate properties such as atomic energies and forces. DFT provides much of
the reference data used to train MLFFs, but its computational cost limits the
size and duration of practical simulations.

Machine learning force field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **machine learning force field (MLFF)** learns energies and forces from
reference electronic-structure calculations. Once trained, it can evaluate
atomic structures much faster than running DFT directly. An MLFF remains an
approximation and may become unreliable for configurations outside its
training distribution.

Training distribution
~~~~~~~~~~~~~~~~~~~~~

The **training distribution** is the range of chemistries, geometries,
energies, and forces represented in a model's training data. Predictions are
generally most trustworthy near this distribution and less certain for
unfamiliar atomic environments.

Relaxation
----------

Atomic relaxation
~~~~~~~~~~~~~~~~~

A **relaxation** repeatedly evaluates a structure's energy and forces and uses
the L-BFGS optimizer to update the atomic positions. The process stops when the
maximum atomic force satisfies the convergence threshold or when the
configured step limit is reached.

Relaxation finds a local minimum under the selected MLFF. It does not, by
itself, prove that the resulting structure is physically correct or agrees
with DFT.

First relaxation
~~~~~~~~~~~~~~~~

The **first relaxation** converts the input structure into a model-specific
reference configuration. Forces, energy, geometry, symmetry, and topology are
recorded from this reference before an attack is applied.

Second relaxation
~~~~~~~~~~~~~~~~~

The **second relaxation** begins from the perturbed structure and tests
recovery. Its final configuration is compared with the first-relaxation
reference. A run reaching the 600-step limit without satisfying the force
criterion is recorded as non-converged, not fully recovered.

Relaxation trajectory
~~~~~~~~~~~~~~~~~~~~~

A **trajectory** stores the sequence of atomic configurations produced during
relaxation. It can be used to examine how energy, forces, volume, and atomic
positions evolve at every optimizer step.

Models
------

MACE
~~~~

**Message Passing Atomic Cluster Expansion (MACE)** uses higher-order
equivariant message passing to predict atomic energies and forces. Atoms are
represented as graph nodes, neighbour relationships as edges, and geometric
information is combined through equivariant tensor products. Equivariance
ensures that rotating the input produces the corresponding rotation of
vector-valued predictions.

The project-specific MACE model is trained using LiCOHPF data.

See :ref:`MACE <method-ref-mace>`.

MACE-MH
~~~~~~~

**MACE-MH** is a pretrained MACE-family model intended to cover a broader
range of materials and atomic environments than the project-specific MACE
model.

See :ref:`MACE-MH <method-ref-mace-mh>`.

UMA
~~~

**Meta's Universal Model for Atoms (UMA)** is a general-purpose pretrained
model developed from a large and chemically diverse collection of atomic
structures. Its predictions can be conditioned by task, charge, and spin
settings.

See :ref:`UMA <method-ref-uma>`.

CHGNet
~~~~~~

**Crystal Hamiltonian Graph Neural Network (CHGNet)** is a graph-based MLFF
designed for crystalline materials. It uses message passing between atoms and
their neighbours to predict energies, forces, stresses, and related material
properties.

See :ref:`CHGNet <method-ref-chgnet>`.

MTP
~~~

A **Moment Tensor Potential (MTP)** represents atomic environments using
moment-tensor descriptors and fits an efficient parameterized energy model.
MTP calculations are typically computationally compact. The MTP used in this
project is trained using LiCOHPF data.

See :ref:`MTP <method-ref-mtp>`.

Random seeds
------------

A **random seed** initializes stochastic operations so that an experiment can
be repeated. In this project, seeds control random components where present;
they do not retrain or randomly reinitialize the fixed pretrained model
weights. Comparing seeds 42--46 tests whether reported conclusions are stable
to stochastic variation.

Adversarial methods
-------------------

Adversarial attack
~~~~~~~~~~~~~~~~~~

An **adversarial attack** deliberately changes atomic coordinates using the
gradient of a selected objective. The goal is to find small, controlled
perturbations that expose unstable or unreliable model behaviour.

These are **white-box attacks** because they require access to gradients from
the evaluated model.

Attack budget
~~~~~~~~~~~~~

The attack budget, :math:`\epsilon`, limits the permitted coordinate
perturbation. This project additionally reports epsilon as a percentage of
the structure's minimum lattice-vector length to support comparison between
structures of different sizes.

FGSM
~~~~

The **Fast Gradient Sign Method (FGSM)** applies one coordinate update in the
sign direction of the objective gradient. It provides a computationally
inexpensive single-step attack.

See :ref:`FGSM <method-ref-fgsm>`.

I-FGSM
~~~~~~

The **Iterative Fast Gradient Sign Method (I-FGSM)** divides the attack into
multiple smaller gradient-sign updates. Intermediate configurations remain
restricted by the selected attack budget.

See :ref:`I-FGSM <method-ref-ifgsm>`.

PGD
~~~

**Projected Gradient Descent (PGD)** applies repeated gradient-based updates
and projects the result back into the allowed perturbation region after each
step. It is a strong first-order white-box attack, but it should not be
described as universally strongest for every model and objective.

See :ref:`PGD <method-ref-pgd>`.

Non-adversarial baseline
------------------------

Contour exploration
~~~~~~~~~~~~~~~~~~~

**Contour exploration** follows an isoenergy path, providing a
non-adversarial baseline that moves through a targeted approximately
constant-energy region instead of following an adversarial gradient.
It helps determine whether failures are caused specifically by
adversarial direction selection or merely by moving atoms away from
equilibrium.

See :ref:`Contour exploration <method-ref-contour>`.
