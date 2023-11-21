#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name anet_eval_seed42
#SBATCH -o anet_eval_seed42.out
#SBATCH -e anet_eval_seed42.err

# debug info
hostname
which python3
nvidia-smi

TORCH_DISTRIBUTED_DEBUG=DETAIL

file_name=("temporal_contact_swap" "temporal_action_swap" "neighborhood_same_entity" "neighborhood_diff_entity" "counter_spatial" "counter_contact" "counter_action" "counter_attribute")


source ~/.bashrc
conda activate clip4clip
which python
export NCCL_P2P_DISABLE=1

for i in {0..7}
do
    manipulation=${file_name[$i]}
    echo $manipulation
    suffix="_mani"
    manipulation_neg=${file_name[$i]}$suffix
    
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 22222 \
    main_xclip.py --do_eval --num_thread_reader=8 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/ \
    --features_path /home/wiss/zhang/nfs/Anet-compressed \
    --output_dir /home/wiss/zhang/nfs/Mg_ckpt/anet_eval_seed42/ \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 1 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/anet_train1_seed42/pytorch_model.bin.1 \
    --manipulation $manipulation \
    --scale 0 \
    --dataset_ckpt seed42 \
    --train_file train_1.csv \
    --val_file ''$manipulation'.csv' \
    --test_file ''$manipulation'.csv' \
    --seed 42 \
    
    echo $manipulation_neg
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 22222 \
    main_xclip.py --do_eval --num_thread_reader=8 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/AnetQA/ \
    --features_path /home/wiss/zhang/nfs/Anet-compressed \
    --output_dir /home/wiss/zhang/nfs/Mg_ckpt/anet_eval_seed42/ \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 1 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/anet_train1_seed42/pytorch_model.bin.1 \
    --manipulation $manipulation_neg \
    --scale 1 \
    --dataset_ckpt seed42 \
    --train_file train_1.csv \
    --val_file ''$manipulation_neg'.csv' \
    --test_file ''$manipulation_neg'.csv' \
    --seed 42 \
    

done