Results Index
=============

This page summarizes the measured outcomes and principal findings. Blank
fields should be completed only after the corresponding summary files have been
validated.

Outcomes
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 38 28

   * - Outcome
     - Stage
     - Question answered
     - Result
   * - Delta force
     - After first relaxation
     - Median force change before attack
     - *Fill in*
   * - Delta force
     - Immediately after attack
     - Median force change before recovery
     - MACE and MTP break off to 10:sup:\6-10:sup:\9 after ~1% min lattice
   * - Delta force
     - After second relaxation
     - Whether force differences persist after recovery
     - MACE and MTP remain at ~10:sup:\8 after ~1% min lattice, while other models < 1
   * - Displacement
     - After first relaxation
     - Median atomic movement before attack
     - *Fill in*
   * - Displacement
     - Immediately after attack
     - Median atomic movement before recovery
     - Smooth increase as epsilon sizes increase
   * - Displacement
     - After second relaxation
     - Median atomic movement after recovery
     - Models begin to diverge, MACE and MTP experience a greater initial spike at ~1% min lattice
   * - Relaxation steps
     - After first relaxation
     - Steps until convergence before attack
     - CHGNet takes most steps ~250, MACE and MTP take the least
   * - Relaxation steps
     - Steps until convergence before recovery
     - Optimization difficulty and frequency of reaching 600 steps
     - *Fill in*
   * - Relaxation steps
     - Steps until convergence after recovery
     - Optimization difficulty and frequency of reaching 600 steps
     - *Fill in*
   * - Neighbour Jaccard distance
     - Pre and post-relaxation
     - Fractional change in neighbour connectivity
     - *Fill in*
   * - Maximum coordination change
     - Pre and post-relaxation
     - Largest site-level coordination change
     - *Fill in*
   * - RDF L1 distance
     - Pre and post-relaxation
     - Change in the distribution of interatomic distances
     - *Fill in*
   * - Space-group change
     - Pre and post-relaxation
     - Whether crystallographic symmetry classification changes
     - *Fill in*
   * - Symmetry-operation retention
     - Pre and post-relaxation
     - Fraction of original symmetry operations retained
     - *Fill in*
   * - Unique-site change
     - Pre and post-relaxation
     - Change in crystallographically unique sites
     - *Fill in*
   * - float32 & float64 agreement
     - Matched post-relaxation runs
     - Sensitivity of outcomes to numerical precision
     - *Fill in*
   * - Cross-seed variability
     - Matched seeds 42-46
     - Sensitivity to stochastic experimental components
     - *Fill in*
   * - Contour comparison
     - Matched model/material groups
     - Difference between adversarial and non-adversarial motion
     - *Fill in*
   * - Supercell comparison
     - Post-relaxation
     - Whether failure modes persist with expanded periodic cells
     - *Fill in*

Key takeaways
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Finding
     - Research question
     - Result
   * - Recovery threshold
     - At what normalized epsilon does recovery begin to fail?
     - *Fill in*
   * - Largest force response
     - Which model/attack/material produces the largest delta force?
     - *Fill in*
   * - Slowest convergence
     - Which model most frequently reaches the 600-step limit?
     - *Fill in*
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
     - *Fill in*
   * - Contour baseline
     - Are adversarial directions more damaging than contour motion?
     - *Fill in*
   * - Supercell dependence
     - Do conclusions persist for expanded periodic systems?
     - *Fill in*
