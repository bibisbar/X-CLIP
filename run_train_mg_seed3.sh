#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name mg_train2_seed3 
#SBATCH -o mg_train2_seed3.out
#SBATCH -e mg_train2_seed3.err
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
python -m torch.distributed.launch --nproc_per_node=2 --master_port 11304 \
    main_xclip.py --do_train --num_thread_reader=16 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/ \
    --features_path /nfs/data2/zhang/MovieGraph/clips_compressed \
    --output_dir /home/wiss/zhang/nfs/video_prober/xclip/mg_train2_seed3 \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 64 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --manipulation mg_train2_seed3  \
    --scale 0 \
    --dataset_ckpt mg_train2_seed3 \
    --train_file train2.csv \
    --val_file test_data_full/temporal_int.csv \
    --test_file test_data_full/temporal_int.csv \
    --seed 3 \