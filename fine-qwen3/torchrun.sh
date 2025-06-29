export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --nproc_per_node=6 $1