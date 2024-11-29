# #!/bin/bash
# CASE_NAME="ramen"

# # path to lerf_ovs/label
# gt_folder="/home/kuangshiai/Documents/Datasets/lerf_ovs/label"

# root_path="/home/kuangshiai/Documents/LangSplat-results"

# python evaluate_iou_loc.py \
#         --dataset_name ${CASE_NAME} \
#         --feat_dir ${root_path}/output \
#         --ae_ckpt_dir ${root_path}/autoencoder_ckpt \
#         --output_dir ${root_path}/eval_result \
#         --mask_thresh 0.4 \
#         --encoder_dims 256 128 64 32 3 \
#         --decoder_dims 16 32 64 128 256 256 512 \
#         --json_folder ${gt_folder}

#!/bin/bash
CASE_NAME="backpack"

# path to ground truth labels
gt_folder=""

root_path="/home/kuangshiai/Documents/LangSplat-results"

pos_query=("a water bottle" "a toothpaste")
neg_query=("a backpack")
test_idx=("00001" "00004" "00005" "00006" "00008" "00009" "00014")
level_idx=("1" "2" "3")

python evaluate_scivis.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output \
        --ae_ckpt_dir ${root_path}/autoencoder_ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder "" \
        --positive_queries "${pos_query[@]}" \
        --negative_queries "${neg_query[@]}" \
        --test_idx "${test_idx[@]}" \
        --level_idx "${level_idx[@]}"