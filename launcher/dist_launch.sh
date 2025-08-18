
SCRIPT=$1
CONFIG=$2
GPUS=$3

# CONDA_PATH="/home/as2114/ls"
source "$CONDA_PATH/etc/profile.d/conda.sh"
#source ~/.bashrc
conda init
conda activate /home/as2114/ls/envs/main
conda env list
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
PYTHONPATH=$PYTHONPATH:/home/as2114/code/PANOPS/Depth-Anything-V2


TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
     $SCRIPT --cfg $CONFIG ${@:4} 


