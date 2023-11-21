#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name xclip_vis
#SBATCH -o xclip_vis.out
#SBATCH -e xclip_vis.err
#SBATCH --exclude=worker-2
# debug info
hostname
which python3
nvidia-smi

TORCH_DISTRIBUTED_DEBUG=DETAIL

source ~/.bashrc
conda activate clip4clip
which python

export NCCL_P2P_DISABLE=1

python visual.py --do_eval --num_thread_reader=8 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/ \
    --features_path /nfs/data2/zhang/MovieGraph/clips_compressed \
    --output_dir /home/wiss/zhang/nfs/Mg_ckpt/xclip/ \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 1 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/mg_train1_seed42/pytorch_model.bin.6 \
    --manipulation temporal_int \
    --scale 0 \
    --dataset_ckpt seed42 \
    --train_file train1.csv \
    --val_file 'test_data_full/temporal_int.csv' \
    --test_file 'test_data_full/temporal_int.csv' \
    --seed 42 \