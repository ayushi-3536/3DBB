
SCRIPT=$1
CONFIG=$2
GPUS=$3
PORT=${PORT:-29509}

# CONDA_PATH="/home/as2114/ls"
source "$CONDA_PATH/etc/profile.d/conda.sh"
#source ~/.bashrc
conda init
conda activate /home/as2114/ls/envs/main
conda env list
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
     $SCRIPT --cfg $CONFIG ${@:4}
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$GPUS --master_port=$PORT \
#    $SCRIPT --cfg $CONFIG ${@:4}


