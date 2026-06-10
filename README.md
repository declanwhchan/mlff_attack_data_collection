# Workflow

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
```

### Optional Downloads

```bash
get outputs_mace
get outputs_uma
```