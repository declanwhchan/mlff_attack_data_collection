# MLFF Attack Data Collection

This repo runs an HPC data-collection workflow from the `mlff_attack` package:

> https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack

This repo has two dataset workflows:

- `run_2d_structures/`: [`20 Materials Project structures`](research/2d_structures.md).
- `run_licohpf_database/`: [`LiCOHPF database`](https://github.com/gitliwq/LiCOHPF_database_1).

---

# Setup for HPC

This repo expects three separate Python environments because MACE, UMA, and CHGNet have different dependency stacks.

## Expected Folder Layout

After completing the following 3 steps:

```text
~/project/
├── mlff_attack/
├── mlff_attack_data_collection/
│   └── .env
├── .venv-mace/
└── .venv-uma/
└── .venv-chgnet/
```

## 1. Clone Repos

```bash
git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
git clone https://github.com/declanwhchan/mlff_attack_data_collection.git
```

## 2. Create Python Environments

```bash
cd ~/project/mlff_attack_data_collection
bash run_<dataset>/dependencies.sh
```

## 3. Create `.env`

Create `.env` in `mlff_attack_data_collection` and enter the following:

```bash
MP_API_KEY=your_materials_project_key
HF_TOKEN=hf_your_huggingface_token_here
```

Then protect it:

```bash
chmod 600 .env
```

# Workflow

Scripts in `run_<dataset>/sample_1/` are the single-test versions of the same workflow, so you can run them for a quick check before running the full collection.

> **Reminder to sync local changes to HPC:** Before running or submitting jobs on the HPC, in SFTP, push/sync all local changes from this computer so the HPC copy is up to date.

## Data Collection Jobs

Run the next step only after the previous step is fully completed.

All steps are to be executed in SSH.

### Step 1 — Run Setup

```bash
sbatch run_<dataset>/setup.sh
```

### Step 2 — Run Main Jobs

```bash
sbatch run_<dataset>/main.sh

# Run contour exploration (OPTIONAL)
sbatch run_<dataset>/contour.sh
```

### Step 3 — Generate Plots

```bash
sbatch run_<dataset>/plot.sh

# Visualize initial atomic structures (OPTIONAL)
sbatch run_<dataset>/visualize.sh
```

### Optional — Run Supercell Stress Test

This submits a controller job that generates supercell CIFs, launches the full MACE/UMA/CHGNet attack array, and then runs a dependent plotting job.

```bash
sbatch run_<dataset>/supercell.sh
```

### Check Status of Jobs

```bash
# Refresh checking the queue every second:
watch -n 1 sq

# Follow live Slurm output:
tail -f slurm-<jobid>.out
```

---

# Export Outputs

## View Plots Without Downloading

### PowerShell

```bash
ssh -L 8000:localhost:8000 <username>@fir.alliancecan.ca
```

### HPC

Then, open a separate terminal and ensure you are in the same `login<#>`. If not, then `ssh login<#>` and continue on HPC:

```bash
cd mlff_attack_data_collection
python -m http.server 8000
```

### Open

```text
http://localhost:8000
```

## Or Download Target Results

Login using SFTP:

```bash
cd /mlff_attack_data_collection

get -r <target_directory>
```
