## Main Repository

This data-collection workflow depends on the main `mlff_attack` Python package:

[TRustworthy-AI-Tools-for-Science/mlff_attack](https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack)

**HPC setup reminder:** before running the workflow, make sure the HPC has `~/project/.venv-mace` and `~/project/.venv-uma` created on the cluster, with the main `mlff_attack` repo installed into both.

# Workflow

## Reminder: Sync Local Changes To HPC

Before running or submitting jobs on the HPC, push/sync all local changes from this computer so the HPC copy is up to date.

## SSH

### 1. Run Setup
```bash
sbatch scripts_bash/setup.sh
```

### 2. Run Main Jobs

**Single-sample test**
```bash
sbatch --array=1-1 scripts_bash/main.sh
```

**Full run**
```bash
sbatch scripts_bash/main.sh
```

### 3. Generate Plots
```bash
sbatch scripts_bash/plot.sh
```

---

## SFTP

### Download Results

```bash
get -r comprehensive_outputs
get -r array_summaries
get generated_material_tests.csv
get -r outputs_mace
get -r outputs_uma
```