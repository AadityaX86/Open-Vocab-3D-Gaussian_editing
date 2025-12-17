import torch
import numpy as np
import faiss
import os
import clip
import viser
import viser.transforms as tf
import time
import matplotlib
import matplotlib.cm as cm
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

def get_colormap_safe(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)

def apply_heatmap(similarities, base_rgbs, threshold=0.20, colormap_name='turbo'):
    """
    Maps similarity scores to a heatmap gradient.
    """
    # 0. Handle NaNs
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

    # 1. Normalize [threshold, max] -> [0, 1]
    max_sim = similarities.max()
    
    # If nothing is relevant, return dimmed scene
    if max_sim <= threshold:
        return base_rgbs * 0.1

    # Stability epsilon
    norm_sim = (similarities - threshold) / (max_sim - threshold + 1e-8)
    norm_sim = torch.clamp(norm_sim, 0, 1)

    # 2. Apply Colormap
    cmap = get_colormap_safe(colormap_name)
    heatmap_colors_cpu = cmap(norm_sim.cpu().numpy())[:, :3] # (N, 3) RGB
    heatmap_colors = torch.tensor(heatmap_colors_cpu, device=base_rgbs.device, dtype=torch.float32)
    
    # 3. Blend
    mask = (similarities > threshold).float().unsqueeze(-1) 
    background_color = base_rgbs * 0.2
    final_colors = heatmap_colors * mask + background_color * (1 - mask)
    
    return final_colors

def main(args, dataset_args, pipeline_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    print(f"\n[1/5] Loading OpenAI CLIP Model...")
    
    # If you get a dimension mismatch error (512 vs 768), switch this string.
    model_name = "ViT-B/32" 
    
    print(f"       -> Loading {model_name}...")
    clip_model, preprocess = clip.load(model_name, device=device)
    
    print(f"[2/5] Loading Gaussian Model from {args.model_path}...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    checkpoint_path = os.path.join(args.model_path, "chkpnt0.pth") 
    
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, args, mode='test')
    
    print("[3/5] Loading FAISS Index and Decoding Features...")
    index = faiss.read_index(args.pq_index)
    
    language_features_idx = gaussians._language_feature.clone()
    check_valid = torch.sum(language_features_idx, 1)
    invalid_index = check_valid == 255 * (index.coarse_code_size() + index.code_size)
    
    decoded_features_cpu = np.zeros((language_features_idx.shape[0], 512), dtype=np.float32)
    valid_mask_cpu = (invalid_index.cpu() == False).numpy()
    
    decoded_features_cpu[valid_mask_cpu] = index.sa_decode(
        language_features_idx[valid_mask_cpu].cpu().numpy()
    )
    
    # Move to GPU
    gaussian_features = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float32)
    
    # Cleanup CPU memory
    del decoded_features_cpu
    del language_features_idx
    torch.cuda.empty_cache()
    
    # Normalize features
    norm = gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features = gaussian_features / (norm + 1e-9)

    print(f"\n[4/5] Processing Query: '{args.query}'")
    with torch.no_grad():
        # OpenAI CLIP Tokenization
        text_tokens = clip.tokenize([args.query]).to(device)
        
        # Encode Text
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Similarity
        similarity = (gaussian_features @ text_features.T).squeeze()
        similarity = torch.nan_to_num(similarity, nan=0.0)

        # Stats
        max_sim = similarity.max().item()
        mean_sim = similarity.mean().item()
        print(f"       -> Max Similarity: {max_sim:.4f}")
        print(f"       -> Avg Similarity: {mean_sim:.4f}")
        
        # Auto-adjust threshold
        target_threshold = args.threshold
        if target_threshold > max_sim:
            print(f"       [Info] Threshold {target_threshold} > Max Score {max_sim:.4f}.")
            target_threshold = max_sim * 0.75
            print(f"       -> Auto-lowered threshold to {target_threshold:.4f}")

        print("       -> Generating Heatmap Colors...")
        shs = gaussians.get_features
        base_rgbs_gpu = (shs[:, 0, :].detach() * 0.28209479177387814 + 0.5).clamp(0, 1)
        
        heatmap_rgbs_gpu = apply_heatmap(
            similarity, 
            base_rgbs_gpu, 
            threshold=target_threshold,
            colormap_name='turbo' 
        )
        
        heatmap_rgbs = heatmap_rgbs_gpu.cpu().numpy().astype(np.float32)

    print("[5/5] Preparing Geometry...")
    means = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    if opacities.ndim == 1: opacities = opacities[:, None]
    
    scales = gaussians.get_scaling.detach().cpu().numpy()
    quats = gaussians.get_rotation.detach().cpu().numpy()
    Rs = tf.SO3(quats).as_matrix()
    covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)

    print(f"\n------------------------------------------------")
    print(f"Starting Viser Server on Port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    server.scene.add_gaussian_splats(
        "/heatmap_scene",
        centers=means,
        rgbs=heatmap_rgbs, 
        opacities=opacities,
        covariances=covariances
    )
    
    print(f"SUCCESS! Heatmap for '{args.query}' is active.")
    print(f"Open your browser at: http://localhost:{args.port}")
    print(f"------------------------------------------------\n")
    
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Heatmap Search (OpenAI CLIP Backup)")
    lp = ModelParams(parser)
    op = PipelineParams(parser)
    
    parser.add_argument("--query", type=str, required=True, help="Text to search for")
    parser.add_argument("--threshold", type=float, default=0.20, help="Similarity threshold") 
    parser.add_argument("--pq_index", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))