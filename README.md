# MLFF Attack Data Collection

This repo runs an HPC data-collection workflow from the `mlff_attack` package:

> https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack

The 20 Materials Project structures used in the comprehensive workflow are listed in `tests_materials.csv`.

Practical-use context and literature sources are summarized in [`materials_practical_uses.md`](research/materials_practical_uses.md).

---

# Setup for HPC

This repo expects two separate Python environments because MACE and UMA have different dependency stacks.

## Expected Folder Layout

After completing the following 3 steps:

```text
~/project/
├── mlff_attack/
├── mlff_attack_data_collection/
├── .venv-mace/
└── .venv-uma/
```

## 1. Clone Repos

```bash
cd ~/project

git clone https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git
git clone https://github.com/declanwhchan/mlff_attack_data_collection.git
```

## 2. Create Python Environments

```bash
cd ~/project/mlff_attack_data_collection
bash scripts_bash/dependencies.sh
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

Scripts in `scripts_bash/sample_1/` are the single-test versions of the same workflow, so you can run them for a quick check before running the full collection.

> **Reminder to sync local changes to HPC:** Before running or submitting jobs on the HPC, in SFTP, push/sync all local changes from this computer so the HPC copy is up to date.

## Data Collection Jobs

Run the next step only after the previous step is fully completed.

All steps are to be executed in SSH.

### Step 1 — Run Setup

```bash
sbatch scripts_bash/setup.sh
```

### Step 2 — Run Main Jobs

```bash
sbatch scripts_bash/main.sh

# Run contour exploration (OPTIONAL)
sbatch scripts_bash/contour.sh
```

### Step 3 — Generate Plots

```bash
sbatch scripts_bash/plot.sh

# Visualize initial atomic structures (OPTIONAL)
sbatch scripts_bash/visualize.sh
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

## Or Download Main Results

Login using SFTP:

```bash
cd mlff_attack_data_collection/trial<#>
get -r outputs_comprehensive
```
