#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name mg_eval_seed2_xclip
#SBATCH -o mg_eval_seed2.out
#SBATCH -e mg_eval_seed2.err

# debug info
hostname
which python3
nvidia-smi

TORCH_DISTRIBUTED_DEBUG=DETAIL

file_name=("temporal_int" "temporal_act" "neighborhood_same_entity" "neighborhood_diff_entity" "counter_rel" "counter_act" "counter_int" "counter_attr")

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
    
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 53679 \
    main_xclip.py --do_eval --num_thread_reader=8 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/ \
    --features_path /nfs/data2/zhang/MovieGraph/clips_compressed \
    --output_dir /home/wiss/zhang/nfs/Mg_ckpt/xclip/ \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 1 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/mg_train1_seed2/pytorch_model.bin.9 \
    --manipulation $manipulation \
    --scale 0 \
    --dataset_ckpt seed2 \
    --train_file train1.csv \
    --val_file 'test_data_full/'$manipulation'.csv' \
    --test_file 'test_data_full/'$manipulation'.csv' \
    --seed 2 \
    

    echo $manipulation_neg
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 53679 \
    main_xclip.py --do_eval --num_thread_reader=8 \
    --epochs=10 --batch_size=64  --n_display=1 \
    --data_path /home/wiss/zhang/Jinhe/video-attr-prober/Data/MovieGraph/ \
    --features_path /nfs/data2/zhang/MovieGraph/clips_compressed \
    --output_dir /home/wiss/zhang/nfs/Mg_ckpt/xclip/ \
    --lr 1e-4 --max_words 60 --max_frames 12 --batch_size_val 1 \
    --datatype moviegraph --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 \
    --init_model /home/wiss/zhang/nfs/video_prober/xclip/mg_train1_seed2/pytorch_model.bin.9 \
    --manipulation $manipulation_neg \
    --scale 1 \
    --dataset_ckpt seed2 \
    --train_file train1.csv \
    --val_file 'test_data_full/'$manipulation_neg'.csv' \
    --test_file 'test_data_full/'$manipulation_neg'.csv' \
    --seed 2 \
   

done