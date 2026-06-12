## MLFF Attack Data Collection

This repo runs the HPC data-collection workflow for the main `mlff_attack` package:

[TRustworthy-AI-Tools-for-Science/mlff_attack](https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack)

**HPC setup reminder:** before running the workflow, make sure the HPC has `~/project/.venv-mace` and `~/project/.venv-uma` created on the cluster, with the main `mlff_attack` repo installed into both.

## Reminders:

### Sync Local Changes To HPC

Before running or submitting jobs on the HPC, in SFTP, push/sync all local changes from this computer so the HPC copy is up to date.

### Create `.env` On HPC

In the HPC repo directory, create `.env` and enter the following:

```bash
MP_API_KEY=your_materials_project_key
HF_TOKEN=hf_your_huggingface_token_here
```

Then protect it:

```bash
chmod 600 .env
```

---

# Workflow

Scripts ending in `_1.sh` are the single-test versions of the same workflow, so you can run `setup_1.sh` and `main_1.sh` for a quick check before running the full collection.

## Data Collection Jobs

Run the next step only after the previous step is fully completed. All steps are to be executed in SSH.

### 1. Run Setup

```bash
sbatch scripts_bash/setup.sh
```

### 2. Run Main Jobs

```bash
sbatch scripts_bash/main.sh
```

### 3. Generate Plots

```bash
sbatch scripts_bash/plot.sh
```

### Check Status of Jobs

```bash
# Refresh checking the queue every second:
watch -n 1 sq

# Follow live Slurm output:
tail -f slurm-<jobid>.out
```

---

## Export Outputs

### View Plots Without Downloading

On the HPC, in SSH from the repo directory:

```bash
cd mlff_attack_data_collection
python -m http.server 8888
```

Then, open a separate terminal and ensure you are in the same login#:

```bash
ssh -L 8888:localhost:8888 <username>@fir.alliancecan.ca
```

Open http://localhost:8888


### Or Download Results

Login using SFTP:

```bash
cd mlff_attack_data_collection
get -r comprehensive_outputs
get -r outputs_mace
get -r outputs_uma
```