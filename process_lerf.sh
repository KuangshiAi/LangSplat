#!/bin/bash

# Define the dataset path and output result path
dataset_path="/home/kuangshiai/Documents/Datasets/lerf_ovs/ramen"
dataset_name="ramen"
result_path="/home/kuangshiai/Documents/LangSplat-results/output/ramen"

# # Step 1: Extract the language features of the scene
# python preprocess.py --dataset_path "$dataset_path"

# # Step 2: Train the autoencoder model
# cd autoencoder
# python train.py --dataset_path "$dataset_path" --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name "$dataset_name"
# # Example usage:
# # python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

# # Step 3: Generate the 3-dimensional language features of the scene
# python test.py --dataset_path "$dataset_path" --dataset_name "$dataset_name"
# # Example usage:
# # python test.py --dataset_path ../data/sofa --dataset_name sofa

# # NOTE: Before training LangSplat, train the RGB 3D Gaussian Splatting model.
# # Refer to https://github.com/graphdeco-inria/gaussian-splatting for instructions.
# # Set the path of your RGB model as '--start_checkpoint'
# cd ..

for level in 1 2 3
do
    python train.py -s "$dataset_path" -m "$result_path" --start_checkpoint "${dataset_path}/output/chkpnt30000.pth" --feature_level "$level"
    # Example usage:
    # python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    # Render RGB output
    python render.py -m "${result_path}_${level}"
    # Render language feature output
    python render.py -m "${result_path}_${level}" --include_feature
    # Example usage:
    # python render.py -m output/sofa_3 --include_feature
done