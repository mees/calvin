#!/bin/bash
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME";
echo "cpus per node: $SLURM_JOB_CPUS_PER_NODE";
echo "gres: $SLURM_GRES";
echo "mem: $SLURM_MEM_PER_NODE";
echo "ntasks: $SLURM_NTASKS";
echo "JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

# Job to perform
source ~/.bashrc
conda activate $1
timeout 23.8h srun python $3 slurm=true hydra.run.dir=$4 trainer.devices=$5 ${@:6}

if [[ $? -eq 124 ]]; then
echo "Time limit exceeded. Resubmit job.";
ssh ${USER}@$2 <<ENDSSH
sh $4/resume_training.sh
ENDSSH
fi

rm /dev/shm/train_*
rm /dev/shm/val_*

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
