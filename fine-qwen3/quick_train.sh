#!/bin/bash

# 快速训练脚本 - 常用配置

echo "请选择训练配置:"
echo "1) 单GPU训练 (最慢，内存占用最少)"
echo "2) 双GPU训练"
echo "3) 四GPU训练 (推荐)"
echo "4) 六GPU训练"
echo "5) 指定特定GPU (例如: 0,2,4)"
echo ""
read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "启动单GPU训练..."
        export CUDA_VISIBLE_DEVICES=0
        python fine-qwen3-torchrun.py
        ;;
    2)
        echo "启动双GPU训练..."
        export CUDA_VISIBLE_DEVICES=0,1
        torchrun --nproc_per_node=2 fine-qwen3-torchrun.py
        ;;
    3)
        echo "启动四GPU训练..."
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        torchrun --nproc_per_node=4 fine-qwen3-torchrun.py
        ;;
    4)
        echo "启动六GPU训练..."
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        torchrun --nproc_per_node=6 fine-qwen3-torchrun.py
        ;;
    5)
        read -p "请输入GPU ID (用逗号分隔，如 0,2,4): " gpu_ids
        gpu_count=$(echo $gpu_ids | tr ',' '\n' | wc -l)
        echo "使用GPU: $gpu_ids (共 $gpu_count 张)"
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        torchrun --nproc_per_node=$gpu_count fine-qwen3-torchrun.py
        ;;
    *)
        echo "无效选择，退出。"
        exit 1
        ;;
esac
