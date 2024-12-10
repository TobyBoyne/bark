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

Then you can use the script `generate_and_submit.py` to create the correct PBS job and submit it. This is essentially a wrapper for PBS so you don't need to be familiar with it as well as enabling a command line interface with default values.