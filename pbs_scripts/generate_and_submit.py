"""
This script is used to build a PBS script using the template and submit it to the HPC.
Will also save a copy of the PBS script to this directory.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
from string import Template
import argparse


if __name__ == "__main__":
    ran_at = datetime.now().replace(microsecond=0).isoformat()
    parser = argparse.ArgumentParser(description='Submit PBS jobs with replicates')
    parser.add_argument('-c', '--config', type=str, required=True,
                    help='Path to configuration file')
    parser.add_argument('-n', '--n-replicates', type=int,
                    help='Number of replicate jobs to run in parallel')
    parser.add_argument('-j', '--job-name', type=str, default='bark_job',
                    help='Name for the PBS job')
    parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed for reproducibility')
    parser.add_argument('-o' , '--out-dir', type=str,
                    help='Directory for job output files - will be cwd if not specified')
    parser.add_argument('--nodes', type=int, default=1,
                    help='Number of nodes to request')
    parser.add_argument('--cpus', type=int, default=1,
                    help='Number of CPUs per node')
    parser.add_argument('--mem', type=int, default=32,
                    help='Memory in GB per node')
    parser.add_argument('--duration', type=str, default=7,
                    help='Duration of the job in hours')
    args = parser.parse_args()

    is_array_job = args.n_replicates is not None
    if is_array_job:
        n_replicates = args.n_replicates
        jobs_directive = f"#PBS -J 1-{n_replicates}"
    else:
        jobs_directive = ""
    JOB_ID_CUT = r"JOBID=`echo ${PBS_JOBID} | cut -d'[' -f1`"

    duration = f"{int(args.duration):02d}:00:00"

    if args.out_dir is None:
        out_dir = Path(os.getcwd()).absolute()
    else:
        out_dir = Path(args.out_dir).absolute()

    if is_array_job:
        logging.info(f"Ignoring seed {args.seed} for array job - using PBS_ARRAY_INDEX")
        seed = r"${PBS_ARRAY_INDEX}"
        array_id = r"$PBS_ARRAYID"
    else:
        seed = args.seed
        array_id = ""

    with open("pbs_job.template", "r") as file:
        template = file.read()

    config_path = Path(args.config).absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {args.config} does not exist")
    
    template = Template(template)
    template = template.substitute(
        define_jobid=JOB_ID_CUT if is_array_job else "",
        jobs_directive=jobs_directive,
        job_name=args.job_name,
        out_dir=out_dir,
        nodes=args.nodes,
        cpus=args.cpus,
        mem=args.mem,
        duration=duration,
        config_path=config_path,
        seed=seed,
        array_id=array_id,
    )

    tmp_save = f"{args.job_name}_{ran_at}.sh"
    with open(tmp_save, "w") as file:
        file.write(template)

    returned = subprocess.run(
        ["qsub", tmp_save],
        capture_output=True,
    )

    if returned.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {returned.stderr.decode('utf-8')} - check generated script {tmp_save}")
    
    # Get PBS job ID from qsub output
    stdout = returned.stdout.decode('utf-8').strip()
    job_id = stdout.removesuffix('.pbs')
    print(f"Submitted job {job_id}")
    # Rename the script to include the job ID as a prefix
    if job_id != "":
        new_filename = f"{job_id}_{tmp_save}"
        os.rename(tmp_save, new_filename)

