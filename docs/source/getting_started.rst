Getting Started
===============

MLFF Attack Data Collection runs HPC experiments using the
`mlff_attack package
<https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack>`_.
The workflow relaxes atomic structures, applies adversarial perturbations,
relaxes the perturbed structures, and collects measurements describing model
recovery and robustness.

Dataset workflows
-----------------

The repository contains two dataset workflows:

* ``run_2d_structures/``: 20 two-dimensional Materials Project structures.
* ``run_licohpf_database/``: 20 LiCOHPF structures.

Commands on this page use ``run_<dataset>`` as a placeholder. Replace it with
either:

.. code-block:: text

   run_2d_structures
   run_licohpf_database

For example:

.. code-block:: bash

   bash run_licohpf_database/setup.sh

Experiment
----------

The experiments evaluate five machine-learning force fields:

* MACE-MH
* UMA
* CHGNet
* MACE
* MTP

Three gradient-based adversarial attacks are evaluated:

* Fast Gradient Sign Method (FGSM)
* Iterative Fast Gradient Sign Method (I-FGSM)
* Projected Gradient Descent (PGD)

Contour exploration is available as a non-adversarial baseline.

The principal workflow is:

#. Load an initial atomic structure.
#. Relax the structure using the selected MLFF.
#. Apply a controlled adversarial perturbation.
#. Relax the perturbed structure using the same MLFF.
#. Compare the original, perturbed, and final structures.
#. Save force, displacement, convergence, symmetry, and topology measurements.
#. Aggregate the run-level results and generate plots.

Setup for HPC
-------------

The MLFF backends require separate dependency environments because their
dependency stacks are not fully compatible. Use the supplied dependency
scripts instead of installing every backend into one environment.

Expected folder layout
~~~~~~~~~~~~~~~~~~~~~~

After installation, the relevant files and environments should resemble:

.. code-block:: text

   ~/project/
   ├── mlff_attack/
   ├── mlff_attack_data_collection/
   │   ├── .env
   │   ├── mace-mh-1.model
   │   ├── uma-s-1p1.pt
   │   ├── MACE_model.model
   │   ├── pot.almtp
   │   └── pot.almtp.elements
   ├── .venv-mace/
   ├── .venv-uma/
   ├── .venv-chgnet/
   ├── mlip-3/
   │   └── bin/
   │       └── mlp
   └── .venv-mtp/
       └── bin/
           └── mlp

The exact environment contents are managed by the dependency scripts and job
launchers.

1. Clone the repositories
~~~~~~~~~~~~~~~~~~~~~~~~~

On the HPC:

.. code-block:: bash

   cd ~/project

   git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
   git clone https://github.com/declanwhchan/mlff_attack_data_collection.git

Enter the data-collection repository:

.. code-block:: bash

   cd ~/project/mlff_attack_data_collection

2. Create the model environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the dependency installer for the selected dataset:

.. code-block:: bash

   bash mlff_venvs_for_hpc.sh

The installer prepares the model-specific Python environments and MTP
executables expected by the Slurm scripts.

3. Configure credentials
~~~~~~~~~~~~~~~~~~~~~~~~

Create ``.env`` in the repository root:

.. code-block:: text

   MP_API_KEY=your_materials_project_key
   HF_TOKEN=hf_your_huggingface_token_here

Protect the file:

.. code-block:: bash

   cd ~/project/mlff_attack_data_collection
   chmod 600 .env

Do not commit ``.env`` or expose its contents in Slurm logs.

4. Install the model artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the externally distributed model artifacts into the root of
``mlff_attack_data_collection``.

.. list-table:: Model artifacts
   :header-rows: 1
   :widths: 20 35 45

   * - Model
     - Required artifact
     - Description
   * - MACE-MH
     - ``mace-mh-1.model``
     - Pretrained MACE-MH model
   * - UMA
     - ``uma-s-1p1.pt``
     - Pretrained UMA checkpoint
   * - MACE
     - ``MACE_model.model``
     - MACE model used by the experiment
   * - MTP
     - ``pot.almtp``
     - MTP potential
   * - MTP
     - ``pot.almtp.elements``
     - Element mapping for the MTP potential
   * - CHGNet
     - No external artifact
     - Loaded through the installed CHGNet package

Verify that the required files exist and are not empty:

.. code-block:: bash

   cd ~/project/mlff_attack_data_collection

   for artifact in \
       mace-mh-1.model \
       uma-s-1p1.pt \
       MACE_model.model \
       pot.almtp \
       pot.almtp.elements
   do
       if [ -s "$artifact" ]; then
           echo "OK: $artifact"
       else
           echo "MISSING OR EMPTY: $artifact"
       fi
   done

Do not submit the full experiment until every required artifact reports
``OK``.

Quick start
-----------

The scripts inside ``run_<dataset>/sample_1/`` are small, single-test versions
of the main workflow. Run these first to check the environments, model
artifacts, credentials, and output paths before launching the complete Slurm
arrays.

Before submitting HPC jobs, synchronize the latest local repository changes
to the HPC copy.

Run the following stages in order. Start each stage only after the required
previous stage has completed successfully.

Step 1: Prepare the experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate the test configurations and required workflow inputs:

.. code-block:: bash

   cd ~/project/mlff_attack_data_collection
   bash run_<dataset>/setup.sh

Optionally generate initial and perturbed CIF files:

.. code-block:: bash

   bash run_<dataset>/cifs.sh

Step 2: Submit the main jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit the relaxation and adversarial-attack jobs:

.. code-block:: bash

   sbatch run_<dataset>/main.sh
   sbatch run_<dataset>/main_gpu.sh

If the dataset provides separate CPU and GPU launchers, submit the required
launcher according to the model allocation documented by that workflow.

Optionally submit the contour exploration baseline:

.. code-block:: bash

   sbatch run_<dataset>/contour.sh
   sbatch run_<dataset>/contour_gpu.sh

Wait for all required main and contour jobs to finish before generating their
combined plots.

Step 3: Generate plots
~~~~~~~~~~~~~~~~~~~~~~

Submit the plotting workflow:

.. code-block:: bash

   sbatch run_<dataset>/plot.sh

The plotting stage reads the completed summary files, combines the available
model results, and writes the generated tables and figures to the project
results directory.

Optionally visualize the initial atomic structures:

.. code-block:: bash

   sbatch run_<dataset>/visualize.sh

Optional supercell stress test
------------------------------

The supercell workflow creates larger structures and repeats the MLFF attack
experiment as a stress test.

The controller script generates supercell CIF files, submits the attack job
array, and schedules a dependent plotting job:

.. code-block:: bash

   sbatch run_<dataset>/supercell.sh
   sbatch run_<dataset>/supercell_gpu.sh

The supercell workflow is separate from the standard main and contour
experiments. It is not required to generate the standard results.

Monitoring jobs
---------------

Refresh the Slurm queue once per second:

.. code-block:: bash

   watch -n 1 sq

Follow the output from a running job:

.. code-block:: bash

   tail -f slurm-<jobid>.out

Replace ``<jobid>`` with the identifier returned by ``sbatch``.

A job disappearing from the queue does not guarantee that every experimental
row succeeded. Inspect its Slurm output and generated summary files before
starting dependent stages.

Viewing results without downloading
-----------------------------------

Create an SSH tunnel
~~~~~~~~~~~~~~~~~~~~

On the local Windows computer, open PowerShell and run:

.. code-block:: powershell

   ssh -L 8000:localhost:8000 <username>@fir.alliancecan.ca

Keep this session open.

Start the HTTP server
~~~~~~~~~~~~~~~~~~~~~

In another terminal, connect to the same HPC login node used by the tunnel.
Then run:

.. code-block:: bash

   cd ~/project/mlff_attack_data_collection
   python -m http.server 8000

Open the following address in the local browser:

`http://localhost:8000 <http://localhost:8000>`_

Downloading selected results
----------------------------

Connect to the HPC using SFTP, navigate to the repository, and download only
the required output directory:

.. code-block:: text

   cd mlff_attack_data_collection
   get -r <target_directory>

Replace ``<target_directory>`` with the desired results, summary, or figure
directory.
