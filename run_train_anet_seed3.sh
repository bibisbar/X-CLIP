#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name anet_train3_seed3
#SBATCH -o anet_train3_seed3_continue.out
#SBATCH -e anet_train3_seed3_continue.err
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
python -m torch.distributed.launch --nproc_per_node=2 --master_port 11405 \
    main_xclip.py --do_train --num_thread_reader=16 \
    --epochs=5 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/ \
    --features_path /home/wiss/zhang/nfs/Anet-compressed \
    --output_dir /home/wiss/zhang/nfs/video_prober/xclip/anet_train3_seed3_continue \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 64 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/anet_train3_seed3/pytorch_model.bin.2 \
    --manipulation anet_train3_seed3  \
    --scale 0 \
    --dataset_ckpt anet_train3_seed3 \
    --train_file train_3.csv \
    --val_file temporal_contact_swap.csv \
    --test_file temporal_contact_swap.csv \
    --seed 3 \