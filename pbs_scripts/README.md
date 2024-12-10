Scripts for running replicates on [Imperial HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/).

Tested with HPC [miniforge/mamba](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/conda/).

Create env with:
```
eval "$(~/miniforge3/bin/conda shell.bash hook)"
mamba create --name bark python=3.10
```

Activate environment and install pip dependencies - note this is a fragile environment state as we are using both conda/mamba and pip at the same time. Luckily as we do everything using pip it should be ok.
```
# With conda/mamba env active
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then make sure gurobi has access to a license, by making a `~/gurobi.lic` file with the line:
```
TOKENSERVER=gurobi.cc.ic.ac.uk
```

Then simply edit the PBS script to select the correct config, and number of replicates and run!
```bash
...
```