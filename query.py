import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from autoencoder.model import Autoencoder
import open_clip  # Import OpenCLIP

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    # Load the autoencoder model
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = f"/home/kuangshiai/Documents/LangSplat-results/autoencoder_ckpt/{args.dataset_name}/best_ckpt.pth"

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load the OpenCLIP model
    clip_model_type = "ViT-B-16"
    clip_model_pretrained = "laion2b_s34b_b88k"
    clip_n_dims = 512  # Expected dimension of CLIP features

    # Load the OpenCLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_type,
        pretrained=clip_model_pretrained,
        device="cuda:0",
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model_type)

    # Encode the query text using OpenCLIP text encoder
    text = args.query_text
    text_tokens = tokenizer([text]).to("cuda:0")
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)  # [1, 512]
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize

    # Encode negative texts if any
    neg_texts = args.neg_query_text  # This is a list of strings
    print(f"Negative texts: {neg_texts}")
    if len(neg_texts) > 0:
        neg_text_tokens = tokenizer(neg_texts).to("cuda:0")  # [N_neg, token_length]
        with torch.no_grad():
            neg_text_embeddings = clip_model.encode_text(neg_text_tokens)  # [N_neg, 512]
        neg_text_embeddings = neg_text_embeddings / neg_text_embeddings.norm(dim=-1, keepdim=True)  # Normalize
    else:
        neg_text_embeddings = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        RGB_img = output["render"]  # [3, H, W]
        language_feature_img = output["language_feature_image"]  # [3, H, W]
        H, W = language_feature_img.shape[1], language_feature_img.shape[2]

        # Prepare the language_feature_img for decoding
        language_feature_img = language_feature_img.to("cuda:0")  # Move to GPU
        # Reshape to [N, 3] where N = H * W
        language_feature_img_flat = language_feature_img.view(3, -1).permute(1, 0)  # [N, 3]

        # Process in batches to avoid memory issues
        batch_size = 4096
        num_pixels = H * W
        decoded_features_list = []

        for i in range(0, num_pixels, batch_size):
            batch = language_feature_img_flat[i:i+batch_size]
            with torch.no_grad():
                decoded_batch = model.decode(batch)
            decoded_features_list.append(decoded_batch.to("cpu"))

        # Concatenate all decoded features
        decoded_features = torch.cat(decoded_features_list, dim=0)  # [N, 512]
        # Reshape back to [512, H, W]
        decoded_features = decoded_features.permute(1, 0).view(512, H, W)  # [512, H, W]

        # Now decoded_features is language feature map in CLIP space

        # Normalize decoded features along the feature dimension
        decoded_features_flat = decoded_features.view(512, -1).permute(1,0)  # [N, 512]
        decoded_features_flat = decoded_features_flat / decoded_features_flat.norm(dim=-1, keepdim=True)
        decoded_features_flat = decoded_features_flat.to("cuda:0")

        # Compute cosine similarity between decoded features and text embedding
        similarity = torch.matmul(decoded_features_flat, text_embedding.T) # [N, 1]

        # Normalize similarity to [0, 1]
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())

        similarity = similarity.view(H, W)

        # Generate mask where similarity > threshold
        threshold = args.similarity_threshold  # default=0.6
        mask = (similarity > threshold).float()  # [H, W]

        # Compute negative similarities if negative texts are provided
        if neg_text_embeddings is not None:
            # decoded_features_flat: [N, 512]
            # neg_text_embeddings: [N_neg, 512]
            # Compute similarity between each pixel feature and each negative text embedding
            similarity_neg = torch.matmul(decoded_features_flat, neg_text_embeddings.T)  # [N, N_neg]

            # Normalize similarity on each negative example to [0, 1]
            for i in range(similarity_neg.shape[1]):
                similarity_neg[:, i] = (similarity_neg[:, i] - similarity_neg[:, i].min()) / (similarity_neg[:, i].max() - similarity_neg[:, i].min())

            # For each pixel, find the maximum similarity across negative texts
            max_similarity_neg, _ = similarity_neg.max(dim=1)  # [N]
            max_similarity_neg = max_similarity_neg.view(H, W)

            # Apply negative threshold
            neg_threshold = args.neg_threshold  # default=0.6
            # Create negative mask where pixels with similarity <= neg_threshold are kept
            mask_neg = (max_similarity_neg <= neg_threshold).float()  # [H, W]
            # Update the mask by element-wise multiplication
            mask = mask * mask_neg

        # Apply mask to RGB image
        RGB_img = RGB_img.to("cpu")  # Move to CPU
        mask = mask.to("cpu")
        masked_RGB_img = RGB_img.clone()
        mask_3c = mask.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        white_background = torch.ones_like(RGB_img)
        masked_RGB_img = masked_RGB_img * mask_3c + white_background * (1 - mask_3c)

        # Save the masked image
        output_dir = os.path.join(model_path, 'masked_outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'masked_image_{idx:05d}.png')
        torchvision.utils.save_image(masked_RGB_img, output_path)

        # Optionally, save the mask
        mask_output_path = os.path.join(output_dir, f'mask_{idx:05d}.png')
        torchvision.utils.save_image(mask.unsqueeze(0), mask_output_path)

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--query_text", required=True, type=str)
    parser.add_argument("--include_feature", action="store_true", default=True)
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument('--similarity_threshold', type=float, default=0.6, help='Threshold for similarity mask')
    parser.add_argument("--neg_query_text", nargs='+', default=[], help="List of negative texts to mask out")
    parser.add_argument('--neg_threshold', type=float, default=0.6, help='Threshold for negative similarity mask')

    args = get_combined_args(parser)
    print("Querying " + args.model_path + " with " + args.query_text)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
