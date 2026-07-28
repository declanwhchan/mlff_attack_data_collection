Results Index
=============

This page indexes the structure sets, chemical systems, models, attacks,
experimental coverage, and observed results. Blank fields should be completed
only after the corresponding summary files have been validated.

Chemistry and structure register
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 25 25

   * - Dataset
     - Chemical system
     - Structures
     - Structure identifiers
     - Coverage notes
   * - LiCOHPF
     - Li-C-O
     - 20
     - ``licohpf_001`` through ``licohpf_020``
     - *Fill in missing or excluded structures*
   * - 2D Materials Project
     - Multiple chemistries
     - 20
     - *Fill in MP identifiers*
     - *Fill in formulas and missing or excluded structures*

Experiment matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 14 10 14 22 18 12 16 14

   * - Dataset
     - Model
     - Attack/baseline
     - Parameter coverage
     - Evaluation stage
     - Seeds
     - Precision
     - Result
   * - LiCOHPF
     - MACE
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE-MH
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE-MH
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE-MH
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MACE-MH
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - UMA
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - UMA
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - UMA
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - UMA
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - CHGNet
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - CHGNet
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - CHGNet
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - CHGNet
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - LiCOHPF
     - MTP
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - LiCOHPF
     - MTP
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - LiCOHPF
     - MTP
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - LiCOHPF
     - MTP
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - 2D Materials Project
     - MACE
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE-MH
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE-MH
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE-MH
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MACE-MH
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - UMA
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - UMA
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - UMA
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - UMA
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - CHGNet
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - CHGNet
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - CHGNet
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - CHGNet
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float32 & float64
     - *Fill in*
   * - 2D Materials Project
     - MTP
     - FGSM
     - Epsilon sweep; single gradient-sign update
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - 2D Materials Project
     - MTP
     - I-FGSM
     - Epsilon sweep and iterative-step sweep
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - 2D Materials Project
     - MTP
     - PGD
     - Epsilon, iterative-step, and step-size sweep
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*
   * - 2D Materials Project
     - MTP
     - Contour exploration
     - Non-adversarial constant-energy baseline
     - Pre and post-relaxation
     - 42-46
     - float64
     - *Fill in*

Outcome and metric register
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 19 19 34 18 30

   * - Outcome
     - Stage
     - Question answered
     - Units/summary
     - Result
   * - Delta force
     - Immediately after attack
     - Median and distribution of force change before recovery
     - eV/Angstrom
     - *Fill in*
   * - Delta force
     - After second relaxation
     - Whether force differences persist after recovery
     - eV/Angstrom
     - *Fill in*
   * - Mean displacement
     - After second relaxation
     - Typical atomic movement relative to the relaxed reference
     - Angstrom
     - *Fill in*
   * - Maximum displacement
     - After second relaxation
     - Largest individual atomic movement
     - Angstrom
     - *Fill in*
   * - Relaxation steps
     - Before and after attack
     - Optimization difficulty and frequency of reaching 300 steps
     - steps
     - *Fill in*
   * - Final energy
     - After second relaxation
     - Energy of the recovered or altered configuration
     - eV
     - *Fill in*
   * - Neighbour Jaccard distance
     - Pre and post-relaxation
     - Fractional change in neighbour connectivity
     - dimensionless
     - *Fill in*
   * - Edges added/removed
     - Pre and post-relaxation
     - Number of changed neighbour relationships
     - count
     - *Fill in*
   * - Mean coordination change
     - Pre and post-relaxation
     - Average change in atomic coordination
     - neighbours
     - *Fill in*
   * - Maximum coordination change
     - Pre and post-relaxation
     - Largest site-level coordination change
     - neighbours
     - *Fill in*
   * - RDF L1 distance
     - Pre and post-relaxation
     - Change in the distribution of interatomic distances
     - method-dependent
     - *Fill in*
   * - Space-group change
     - Pre and post-relaxation
     - Whether crystallographic symmetry classification changes
     - fraction or category
     - *Fill in*
   * - Symmetry-operation retention
     - Pre and post-relaxation
     - Fraction of original symmetry operations retained
     - fraction
     - *Fill in*
   * - Unique-site change
     - Pre and post-relaxation
     - Change in crystallographically unique sites
     - count or fraction
     - *Fill in*
   * - float32 & float64 agreement
     - Matched post-relaxation runs
     - Sensitivity of outcomes to numerical precision
     - :math:`R^2` and paired differences
     - *Fill in*
   * - Cross-seed variability
     - Matched seeds 42-46
     - Sensitivity to stochastic experimental components
     - median and IQR
     - *Fill in*
   * - Contour comparison
     - Matched model/material groups
     - Difference between adversarial and non-adversarial motion
     - metric-dependent
     - *Fill in*
   * - Supercell comparison
     - Post-relaxation
     - Whether failure modes persist with expanded periodic cells
     - metric-dependent
     - *Fill in*

Key findings to complete
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
     - Which model most frequently reaches the 300-step limit?
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
