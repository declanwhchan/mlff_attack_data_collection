HPC Outputs After Full Workflow
===============================

The experiment separates large run-level artifacts from compact summaries,
figures, and documentation.

* **Scratch storage** contains trajectories, forces, structures, scheduler
  summaries, and other large intermediate files.
* **Project storage** contains combined summary tables, final figures, and
  cross-trial comparisons from `plot.sh`.

Storage roots
-------------

Unless overridden by environment variables, the LiCOHPF workflows use:

.. code-block:: text

   Repository:
   /home/$USER/project/mlff_attack_data_collection/

   Scratch:
   /scratch/$USER/mlff_attack_data_collection/licohpf_database/

   Project results:
   /home/$USER/project/mlff_attack_data_collection/
   └── licohpf_database_results/

Trial organization
------------------

Five trials are used for the random-seed comparison:

.. code-block:: text

   trial1_seed42
   trial2_seed43
   trial3_seed44
   trial4_seed45
   trial5_seed46

Every trial uses the same model, material, attack, and precision organization.
The seed changes stochastic experimental components without retraining the
fixed pretrained model checkpoints.

Scratch directory
-----------------

After the CPU, GPU, and contour workflows have generated their data, the
scratch root should follow this structure:

.. code-block:: text

   /scratch/$USER/mlff_attack_data_collection/licohpf_database/
   │
   ├── trial1_seed42/
   │   │
   │   ├── outputs_float32/
   │   │   ├── mace_mh/
   │   │   │   ├── licohpf_001/
   │   │   │   ├── licohpf_002/
   │   │   │   ├── ...
   │   │   │   └── licohpf_020/
   │   │   │
   │   │   ├── uma/
   │   │   │   ├── licohpf_001/
   │   │   │   ├── ...
   │   │   │   └── licohpf_020/
   │   │   │
   │   │   ├── chgnet/
   │   │   │   ├── licohpf_001/
   │   │   │   ├── ...
   │   │   │   └── licohpf_020/
   │   │   │
   │   │   └── mace_model/
   │   │       ├── licohpf_001/
   │   │       ├── ...
   │   │       └── licohpf_020/
   │   │
   │   ├── outputs_float64/
   │   │   ├── mace_mh/
   │   │   ├── uma/
   │   │   ├── mtp/
   │   │   ├── chgnet/
   │   │   └── mace_model/
   │   │
   │   ├── array_summaries/
   │   │   ├── float32_mace_mh_licohpf_001_summary.csv
   │   │   ├── float32_uma_licohpf_001_summary.csv
   │   │   ├── float32_chgnet_licohpf_001_summary.csv
   │   │   ├── float32_mace_model_licohpf_001_summary.csv
   │   │   ├── float64_mace_mh_licohpf_001_summary.csv
   │   │   ├── float64_uma_licohpf_001_summary.csv
   │   │   ├── float64_mtp_licohpf_001_summary.csv
   │   │   ├── float64_chgnet_licohpf_001_summary.csv
   │   │   ├── float64_mace_model_licohpf_001_summary.csv
   │   │   └── ... one summary per model, precision, and structure
   │   │
   │   └── contour_array_summaries/
   │       ├── float32_mace_mh_licohpf_001_summary.csv
   │       ├── float32_uma_licohpf_001_summary.csv
   │       ├── ...
   │       └── float64_mace_model_licohpf_020_summary.csv
   │
   ├── trial2_seed43/
   │   └── same structure as trial1_seed42
   │
   ├── trial3_seed44/
   │   └── same structure as trial1_seed42
   │
   ├── trial4_seed45/
   │   └── same structure as trial1_seed42
   │
   ├── trial5_seed46/
   │   └── same structure as trial1_seed42
   │
   └── supercell/
       └── supercell run-level outputs and summaries

A representative run directory contains the files created for one material,
model, precision, attack, and parameter combination:

.. code-block:: text

   outputs_float32/
   └── uma/
       └── licohpf_004/
           └── fgsm_eps04/
               ├── before_attack_relaxation.traj
               ├── after_attack_relaxation.traj
               ├── before_force.csv
               ├── perturbed_force.csv
               ├── after_force.csv
               ├── perturbed structure output
               ├── final relaxed structure output
               ├── topology edge-change data
               └── run metadata and status

Exact run-folder names encode the attack and its parameters. Iterative cases
may additionally include step-count and step-size tokens.

Summary expectations
--------------------

For each trial, the main summaries should include 20 per-structure files for
every supported model and precision.

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Precision
     - Expected models
     - Structures per model
   * - float32
     - MACE-MH, UMA, CHGNet, MACE Model
     - 20
   * - float64
     - MACE-MH, UMA, MTP, CHGNet, MACE Model
     - 20

Each complete per-model main summary represents 20 structures and the
configured attack rows for those structures. The validation script should be
used to determine whether the expected rows are actually present and
successful.

Project results directory
-------------------------

After ``plot.sh`` combines the scratch summaries, the repository should
contain:

.. code-block:: text

   mlff_attack_data_collection/
   │
   └── licohpf_database_results/
       │
       ├── trial1_seed42/
       │   └── outputs_comprehensive/
       │       │
       │       ├── float32/
       │       │   ├── combined_dataset.csv
       │       │   ├── mace_mh/
       │       │   │   ├── summary.csv
       │       │   │   └── contour/
       │       │   │       └── summary.csv
       │       │   ├── uma/
       │       │   │   ├── summary.csv
       │       │   │   └── contour/
       │       │   │       └── summary.csv
       │       │   ├── chgnet/
       │       │   │   ├── summary.csv
       │       │   │   └── contour/
       │       │   │       └── summary.csv
       │       │   ├── mace_model/
       │       │   │   ├── summary.csv
       │       │   │   └── contour/
       │       │   │       └── summary.csv
       │       │   ├── contour/
       │       │   └── generated float32 figures
       │       │
       │       ├── float64/
       │       │   ├── combined_dataset.csv
       │       │   ├── mace_mh/
       │       │   ├── uma/
       │       │   ├── mtp/
       │       │   ├── chgnet/
       │       │   ├── mace_model/
       │       │   ├── contour/
       │       │   └── generated float64 figures
       │       │
       │       └── comparison/
       │           └── float32-versus-float64 figures
       │
       ├── trial2_seed43/
       │   └── same project structure as trial1_seed42
       │
       ├── trial3_seed44/
       │   └── same project structure as trial1_seed42
       │
       ├── trial4_seed45/
       │   └── same project structure as trial1_seed42
       │
       ├── trial5_seed46/
       │   └── same project structure as trial1_seed42
       │
       ├── random_seed/
       │   ├── cross-seed summary tables
       │   └── random-seed comparison figures
       │
       └── supercell/
           ├── combined supercell summaries
           └── supercell comparison figures

File lifecycle
--------------

The files move through four stages:

.. code-block:: text

   Scheduler scripts
          |
          v
   Scratch run directories
          |
          v
   Per-structure summary CSVs
          |
          v
   Project combined summaries and figures

``main.sh`` and ``main_gpu.sh``
   Generate adversarial and relaxation run-level outputs and write main array
   summaries.

``contour.sh`` and ``contour_gpu.sh``
   Generate contour exploration outputs and write contour array summaries.

``supercell.sh`` and ``supercell_gpu.sh``
   Generate expanded-cell calculations under the supercell output root.

``plot.sh``
   Combines available scratch summaries into project-level model summaries,
   builds comprehensive datasets, generates figures, and initiates cross-seed
   plotting.

Live directory checks
---------------------

Inspect the scratch layout:

.. code-block:: bash

   bash run_licohpf_database/completion_check.sh
