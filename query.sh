# #!/bin/bash
# dataset_name="backpack"
# result_path="/home/kuangshiai/Documents/LangSplat-results/output/backpack"
# query_text="a water bottle"
# neg_query_text=("a backpack")
# neg_threshold=0.6
# threshold=0.6

# # 或许不应该简单用threshold来判断，而是应该利用原有的SAM分割来找到那个类，然后直接用那个类的mask

# for level in 1
# do
#     # Render RGB output highlighting the queried object
#     python query.py -m "${result_path}_${level}" --dataset_name "$dataset_name" --query_text "$query_text" --similarity_threshold $threshold --neg_query_text "${neg_query_text[@]}" --neg_threshold $neg_threshold
# done


#!/bin/bash
dataset_name="backpack"
feat_dir="/home/kuangshiai/Documents/LangSplat-results/output"
model_path="/home/kuangshiai/Documents/LangSplat-results/output/backpack_1"
query_text="a water bottle"
neg_query_text=("a backpack")
neg_threshold=0.6
pos_threshold=0.6

python query_smooth.py \
    --m $model_path \
    --dataset_name $dataset_name \
    --feat_dir $feat_dir \
    --query_text "$query_text" \
    --neg_query_text "${neg_query_text[@]}" \
    --pos_threshold $pos_threshold \
    --neg_threshold $neg_threshold \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --skip_test