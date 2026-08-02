Results Index
=============

This page summarizes the measured outcomes and principal findings. Blank
fields should be completed only after the corresponding summary files have been
validated.

Figures
----------------------

The following figures summarize the principal trends discussed in the tables
below. They provide a visual overview of model behaviour across the evaluated
metrics.

.. list-table::
   :widths: 50 50
   :class: borderless

   * - .. figure:: ../images/delta_force.png
          :width: 100%
          :align: center

          **Delta force.** Force response immediately following attack.

     - .. figure:: ../images/displacement.png
          :width: 100%
          :align: center

          **Atomic displacement.** Mean displacement versus perturbation strength.

   * - .. figure:: ../images/relaxation_steps.png
          :width: 100%
          :align: center

          **Relaxation steps.** Number of optimization steps required for convergence.

     - .. figure:: ../images/rdf_distance.png
          :width: 100%
          :align: center

          **RDF distance.** Structural similarity after relaxation.

   * - .. figure:: ../images/jaccard.png
          :width: 100%
          :align: center

          **Neighbour Jaccard distance.** Local topology changes.

     - .. figure:: ../images/coordination.png
          :width: 100%
          :align: center

          **Coordination change.** Maximum coordination-number variation.

Outcomes
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 28

   * - Outcome
     - Stage
     - Result
   * - Delta force
     - Post-attack
     - MACE and MTP break off to 10\ :sup:`6` -- 10\ :sup:`9` after 1 min lattice
   * - Delta force
     - Post-attack + relaxation
     - MACE and MTP remain at 10\ :sup:`8`eV/Å after perturbation of ~1% min. lattice, while other models < 1eV/Å
   * - Displacement
     - Post-attack
     - Smooth exponential increase as epsilon sizes increase
   * - Displacement
     - Post-attack + relaxation
     - Models begin to diverge, MACE and MTP experience a greater initial spike at ~1% min lattice
   * - Relaxation steps
     - Initial relaxation
     - CHGNet takes most steps ~250, MACE and MTP take the least
   * - Relaxation steps
     - Post-attack + relaxation
     - *Fill in*
   * - Neighbour Jaccard distance
     - Post-attack
     - *Fill in*
   * - Neighbour Jaccard distance
     - Post-attack + relaxation
     - *Fill in*
   * - Maximum coordination change
     - Post-attack
     - *Fill in*
   * - Maximum coordination change
     - Post-attack + relaxation
     - *Fill in*
   * - RDF L1 distance
     - Post-attack
     - *Fill in*
   * - RDF L1 distance
     - Post-attack + relaxation
     - *Fill in*
   * - Space-group change
     - Post-attack
     - *Fill in*
   * - Space-group change
     - Post-attack + relaxation
     - *Fill in*
   * - Symmetry-operation retention
     - Post-attack
     - *Fill in*
   * - Symmetry-operation retention
     - Post-attack + relaxation
     - *Fill in*
   * - Unique-site change
     - Post-attack
     - *Fill in*
   * - Unique-site change
     - Post-attack + relaxation
     - *Fill in*

Key takeaways
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Finding
     - Question
     - Result
   * - Recovery threshold
     - At what normalized epsilon does recovery begin to fail?
     - 1% of min lattice parameter
   * - Largest force response
     - Which model/attack/material produces the largest delta force?
     - MACE and MTP pretrained by LiCOHPF dataset
   * - Slowest convergence
     - Which model most frequently reaches the 600-step limit?
     - CHGNet
   * - Most recoverable model
     - Which model has the smallest post-relaxation changes?
     - *Fill in*
   * - Least recoverable model
     - Which model has the largest persistent changes?
     - *Fill in*
   * - Attack comparison
     - How do FGSM, I-FGSM, and PGD differ under matched budgets?
     - *Fill in*
   * - Chemistry dependence
     - Do LiCOHPF and 2D materials exhibit different failure patterns?
     - *Fill in*
   * - Topology transition
     - When do connectivity or coordination changes become persistent?
     - *Fill in*
   * - Precision dependence
     - Do float32 and float64 support the same conclusions?
     - *Fill in*
   * - Seed dependence
     - Are conclusions stable across seeds 42 through 46?
     - No abnormalities which suggest results are stable and reliable.
   * - Contour exploration baseline
     - Are adversarial directions more damaging than contour motion?
     - Slight increase but many magnitudes less than deliberate attacks,
       suggesting that atoms merely moving away from equilibrium do not lead
       to failure modes compared to adversarial perturbations that maximize
       model loss.
   * - Supercell dependence
     - Do conclusions persist for expanded periodic systems?
     - *Fill in*
