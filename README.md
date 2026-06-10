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