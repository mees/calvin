#!/bin/bash
now=$(date +"%y-%m-%d/%H-%M-%S")
name=$(basename $1 experiment)
logpath="/home/hermannl/logs/play/$name/$now"
echo $logpath
mkdir -p $logpath
cd $logpath
echo "$(pwd)"

#####
##### USE FOR DEBUGGING PURPOSES
#####


#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 128000 # memory pool for each core (4GB)
#SBATCH -t 0-00:05 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
#SBATCH --gres=gpu:2
#SBATCH -o %x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e %x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
# #SBATCH -J slurm_test # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH --signal=SIGUSR1@90
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Job to perform
source ~/.bashrc
conda activate play_env
python /home/hermannl/repos/learning_from_play/lfp/training.py slurm_job_id=_$SLURM_JOB_ID log_dir=/home/hermannl/logs dataset.root_data_dir=/home/hermannl/data/banana dataset=play_table sampler=constant model.tsne_plot=false observation_space=rgb_only trainer.gpus=2

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
