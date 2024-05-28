#!/bin/bash -l
#SBATCH --job-name=internimage_train    # Kurzname des Jobs
#SBATCH --output=internimage_train-%j.out
#SBATCH --error=internimage_train-%j.err
#SBATCH --gres=gpu:a100:8
#SBATCH --constraint=a100_80
#SBATCH --partition=a100
#SBATCH --time=16:00:00         # Gesamtlimit für Laufzeit des Jobs (Format: HH:MM:SS)
#SBATCH --nodes=1               # Anzahl Knoten
#SBATCH --ntasks=8              # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL         # Art des Mailversands (gültige Werte z.B. ALL, BEGIN, END, FAIL oder REQUEUE)
#SBATCH --mail-user=kammerbauerro76348@th-nuernberg.de  # Emailadresse für Statusmails
#SBATCH --export=TORCH_HOME        # export only specified environment from submitting shell
                                   # first non-empty non-comment line ends SBATCH options
unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun

module purge
module load python/3.9-anaconda
module load cuda/11.5.1
module load cudnn/8.3.1.22-11.5.1
module load gcc/12.1.0

GPUS=8
CONFIG=$1
VAULT_DIR=$2
PY_ENV=internimage_env
PORT=${PORT:-29300}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

mkdir $VAULT_DIR/training_logs/internimage/$SLURM_JOB_ID

CACHE_DIR=$VAULT_DIR/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p $CACHE_DIR

export TORCH_HOME=$VAULT_DIR/models/torchhub
mkdir -p $TORCH_HOME

conda activate $PY_ENV

srun python -u $VAULT_DIR/software/InternImage/segmentation/train.py $CONFIG --work-dir=$VAULT_DIR/training_logs/internimage/$SLURM_JOB_ID --launcher='slurm' ${@:3}
