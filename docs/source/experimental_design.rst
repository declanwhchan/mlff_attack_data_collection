.. _experimental-design:

Experimental Design
===================

This study tests whether machine-learning force fields recover after their
input atomic coordinates are deliberately perturbed. Mathematical definitions
are provided separately in :doc:`metrics`.

Workflow
--------

Every adversarial experiment uses the same sequence:

#. Load an initial atomic structure.
#. Relax it using the selected MLFF.
#. Save the relaxed structure as the reference.
#. Apply an adversarial coordinate perturbation.
#. Measure the immediate response.
#. Relax the perturbed structure with the same MLFF.
#. Compare the final structure with the reference.

.. code-block:: text

   Initial structure
          ↓
   First relaxation
          ↓
   Reference structure
          ↓
   Adversarial perturbation
          ↓
   Immediate measurements
          ↓
   Second relaxation
          ↓
   Recovery measurements

The first relaxed structure is the reference. The original unrelaxed input is
not used as the recovery target.

Datasets
--------

Two structure sets are evaluated:

* **Two-dimensional Materials Project:** 20 low-dimensional structures.
* **LiCOHPF:** 20 Li-C-O configurations associated with
  solid-electrolyte-interphase modelling.

The two datasets test whether observed behaviour is specific to one chemistry
or structural domain.

Models
------

Five MLFFs are compared:

* **MACE**, trained using LiCOHPF data
  (:ref:`MACE reference <method-ref-mace>`).
* **MTP**, trained using LiCOHPF data
  (:ref:`MTP reference <method-ref-mtp>`).
* **MACE-MH**, a pretrained MACE-family model
  (:ref:`MACE-MH reference <method-ref-mace-mh>`).
* **UMA**, Meta's pretrained Universal Model for Atoms
  (:ref:`UMA reference <method-ref-uma>`).
* **CHGNet**, a pretrained crystal graph neural network
  (:ref:`CHGNet reference <method-ref-chgnet>`).

The experiment compares model behaviour, not only computational speed or
ordinary test-set accuracy.

Adversarial attacks
-------------------

Three coordinate attacks are evaluated:

* **FGSM:** one sign-gradient update
  (:ref:`FGSM reference <method-ref-fgsm>`).
* **I-FGSM:** repeated sign-gradient updates
  (:ref:`I-FGSM reference <method-ref-ifgsm>`).
* **PGD:** repeated updates projected into the allowed perturbation region
  (:ref:`PGD reference <method-ref-pgd>`).

The attacks modify atomic coordinates. Model parameters remain fixed.

Attack variables include:

* epsilon;
* normalized epsilon;
* number of attack steps;
* attack step size;
* random seed.

The attack equations are defined in :ref:`metric-attack-definitions`.

Relaxation
----------

The structure is relaxed before and after the attack using the same model and
relaxation settings.

The configured convergence threshold is:

.. code-block:: text

   fmax = 0.01 eV/Å

The maximum relaxation length is:

.. code-block:: text

   300 optimizer steps

A run reaching 300 steps reached the configured limit and should not
automatically be described as converged.

Relaxation uses the Atomic Simulation Environment optimization interface
(:ref:`ASE optimization reference <method-ref-ase-optimize>`).

Evaluation stages
-----------------

Metrics are recorded at two stages:

**Immediate response**
   The attacked structure is compared with the pre-attack relaxed reference
   before the second relaxation.

**Post-relaxation response**
   The final structure is compared with the same reference after the second
   relaxation.

This separates attack magnitude from recovery behaviour.

Baseline
--------

Contour exploration is the non-adversarial baseline. It explores a targeted
energy contour instead of selecting an adversarial gradient direction.

The implementation uses ASE contour exploration
(:ref:`contour exploration reference <method-ref-contour>`).

Contour exploration is a structured baseline, not a random-noise baseline.

Experiment coverage
-------------------

The benchmark evaluates two sets of 20 structures: LiCOHPF Li-C-O
configurations and two-dimensional Materials Project structures. Each
structure is tested using MACE, MACE-MH, UMA, CHGNet, and MTP.

The adversarial evaluation includes FGSM, I-FGSM, and PGD. Attack
configurations span nominal :math:`\epsilon` values from 0.001 to 10 and
1 to 100 attack steps where applicable. Contour exploration provides a
500-step non-adversarial constant-energy baseline.

Experiments use random seeds 42 through 46. MACE, MACE-MH, UMA, and CHGNet
are evaluated in float32 and float64, while MTP is evaluated in float64.
Reported epsilon values may additionally be normalized by each structure's
minimum lattice-vector length for cross-structure comparison.

Experimental comparisons
------------------------

The broader experiment compares:

* random seeds 42 through 46;
* float32 and float64 where supported;
* epsilon sweeps;
* attack-step sweeps;
* attack step-size sweeps;
* immediate and post-relaxation outcomes;
* primitive structures and supercells;
* adversarial attacks and contour exploration.

The experiment seeds control stochastic experimental operations. They do not
retrain the downloaded model artifacts.

Supercell stress test
---------------------

The optional supercell workflow repeats the experiment using expanded periodic
structures. It tests whether observed behaviour persists when the represented
atomic system becomes larger.

Supercells are constructed using ASE structure-building operations
(:ref:`ASE supercell reference <method-ref-supercell>`).

Measured outcomes
-----------------

The analysis includes:

* displacement;
* delta force and force angle;
* relaxation steps;
* final energy;
* neighbour-edge changes;
* neighbour-set Jaccard distance;
* coordination changes;
* RDF L1 distance;
* space-group change;
* symmetry-operation retention;
* unique-site change;
* float32/float64 agreement;
* cross-seed variability.

Every mathematical definition is listed on the :doc:`metrics` page.

Interpretation
--------------

The experiment evaluates robustness under deliberately challenging coordinate
perturbations. It does not directly measure error against DFT unless an
explicit DFT reference is included.

A persistent post-relaxation difference indicates failure to recover the
pre-attack relaxed configuration under the selected model and settings. It
does not, by itself, prove that the final structure is physically impossible.

The validation context is discussed in the
:ref:`MLFF validation reference <method-ref-validation>`.

