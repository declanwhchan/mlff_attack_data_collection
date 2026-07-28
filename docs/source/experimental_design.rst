Experimental Design
===================

Workflow
--------

Every adversarial experiment follows the same conceptual sequence:

#. Load an initial atomic structure.
#. Relax the structure with the selected MLFF.
#. Record the pre-attack forces, geometry, symmetry, and topology.
#. Apply a controlled adversarial coordinate perturbation.
#. Measure the immediate response before the second relaxation.
#. Relax the perturbed structure using the same model and settings.
#. Compare the final configuration with the pre-attack relaxed reference.

The workflow can be summarized as:

``Initial structure -> relaxation -> perturbation -> relaxation -> recovery analysis``

A relaxation repeatedly predicts the current energy and forces, then uses an
optimizer to update atomic positions. It ends when the force convergence
criterion is satisfied or when the 300-step limit is reached. A trajectory
stores the sequence of configurations produced during this process.

Datasets
--------

The benchmark contains two complementary chemistry domains:

* **Two-dimensional Materials Project set:** 20 structures selected to test
  diverse low-dimensional inorganic materials.
* **LiCOHPF set:** 20 Li-C-O configurations associated with
  solid-electrolyte-interphase modelling.

Using two domains helps separate general failure patterns from behaviour tied
to a particular chemistry or training distribution.

Models
------

Five MLFFs are evaluated:

* **CHGNet:** a graph neural network designed for crystalline materials.
* **MACE:** the project-specific MACE model trained using LiCOHPF data.
* **MACE-MH:** a pretrained general-purpose MACE-family model.
* **UMA:** Meta's pretrained Universal Model for Atoms.
* **MTP:** a moment tensor potential trained using LiCOHPF data.

MACE-family models use equivariant geometric message passing to represent
many-body atomic environments. UMA is designed to cover broad chemical and
task domains. CHGNet represents crystals through graph-based message passing.
MTP uses a moment-tensor descriptor expansion and is computationally compact.

Attacks
-------

Three first-order white-box attacks are considered:

* **FGSM:** one update in the sign direction of the selected gradient.
* **I-FGSM:** repeated smaller FGSM-style updates.
* **PGD:** iterative gradient updates projected back into an allowed
  perturbation region.

The reported perturbation magnitude is the nominal attack epsilon normalized
by the corresponding structure's minimum lattice-vector length:

``normalized epsilon (%) = 100 * epsilon / min(|a|, |b|, |c|)``

This quantity describes the attack budget. It should not be interpreted as a
guarantee that the final post-relaxation atomic displacement has the same
magnitude.

Contour exploration
~~~~~~~~~~~~~~~~~~~

Contour exploration is the non-adversarial baseline. It moves through a
targeted constant-energy region rather than following an adversarial
gradient. This provides a structured comparison for determining whether
gradient-selected directions are unusually damaging.

Additional comparisons
~~~~~~~~~~~~~~~~~~~~~~

The broader experiment includes:

* random seeds 42 through 46;
* float32 and float64 calculations where supported;
* epsilon sweeps;
* iterative-step sweeps;
* primitive structures and supercells;
* immediate and post-relaxation force changes;
* atomic displacement and convergence;
* neighbour, coordination, RDF, symmetry, and topology metrics.
