import torch
import numpy as np
import faiss
import os
import open_clip
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

def apply_overlay_heatmap(similarities, base_rgbs, threshold=0.2, colormap_name='turbo'):
    """
    Overlays a heatmap ONLY on relevant objects. 
    Leaves everything else as the original photorealistic color.
    """
    # 0. Handle NaNs
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

    # 1. Normalize scores [threshold, max] -> [0, 1]
    max_sim = similarities.max()
    
    # If nothing matches, just return the original scene untouched
    if max_sim <= threshold:
        return base_rgbs 

    # Normalize: 0.0 = threshold, 1.0 = max_score
    norm_sim = (similarities - threshold) / (max_sim - threshold + 1e-8)
    norm_sim = torch.clamp(norm_sim, 0, 1)

    # 2. Get Heatmap Colors
    cmap = get_colormap_safe(colormap_name)
    heatmap_colors_cpu = cmap(norm_sim.cpu().numpy())[:, :3] # (N, 3)
    heatmap_colors = torch.tensor(heatmap_colors_cpu, device=base_rgbs.device, dtype=torch.float32)
    
    # 3. Blend Logic
    # mask = 1.0 if relevant, 0.0 if not
    mask = (similarities > threshold).float().unsqueeze(-1) 
    
    # Background is now just 'base_rgbs'
    background_color = base_rgbs 
    
    # Mix: Heatmap on matches + Original on non-matches
    final_colors = heatmap_colors * mask + background_color * (1 - mask)
    
    return final_colors

def main(args, dataset_args, pipeline_args):
    # 1. Setup & Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print(f"\n[1/5] Loading OpenCLIP Model (ViT-B-16)...")
    model_name = "ViT-B-16" 
    pretrained_source = "laion2b_s34b_b88k"
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained_source, 
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    print(f"[2/5] Loading Gaussian Model from {args.model_path}...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    checkpoint_path = os.path.join(args.model_path, "chkpnt0.pth") 
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, args, mode='test')
    
    # 2. Decode Language Features
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
    
    # Move to GPU for fast dot product
    gaussian_features = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float32)
    del decoded_features_cpu
    torch.cuda.empty_cache()
    
    # Normalize features once
    norm = gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features = gaussian_features / (norm + 1e-9)

    # 3. Pre-calculate Geometry (This doesn't change during interaction)
    print("[4/5] Pre-calculating Geometry and Base Colors...")
    means = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    if opacities.ndim == 1: opacities = opacities[:, None]
    
    scales = gaussians.get_scaling.detach().cpu().numpy()
    quats = gaussians.get_rotation.detach().cpu().numpy()
    Rs = tf.SO3(quats).as_matrix()
    covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)

    # Get Base RGBs (Photorealistic)
    shs = gaussians.get_features
    base_rgbs_gpu = (shs[:, 0, :].detach() * 0.28209479177387814 + 0.5).clamp(0, 1)

    # 4. Initialize Viser
    print(f"\n------------------------------------------------")
    print(f"Starting Viser Server on Port {args.port}...")
    print(f"Interactive Mode Enabled.")
    server = viser.ViserServer(port=args.port)

    # --- GUI ELEMENTS ---
    with server.gui.add_folder("Semantic Search"):
        gui_text = server.gui.add_text(
            "Query", 
            initial_value=args.query if args.query else ""
        )
        gui_threshold = server.gui.add_slider(
            "Threshold", 
            min=0.0, max=1.0, step=0.01, 
            initial_value=args.threshold
        )
        gui_status = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    # --- INTERACTIVE UPDATE FUNCTION ---
    def update_scene(_):
        """Callback to run whenever text or slider changes"""
        query_text = gui_text.value
        current_thresh = gui_threshold.value
        
        if not query_text:
            # If empty text, show original scene
            server.scene.add_gaussian_splats(
                "/scene",
                centers=means,
                rgbs=base_rgbs_gpu.cpu().numpy(),
                opacities=opacities,
                covariances=covariances
            )
            gui_status.value = "Showing Original"
            return

        gui_status.value = f"Processing '{query_text}'..."

        with torch.no_grad():
            # 1. Encode Text
            text_tokens = tokenizer([query_text]).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 2. Calculate Similarity
            similarity = (gaussian_features @ text_features.T).squeeze()
            similarity = torch.nan_to_num(similarity, nan=0.0)

            max_sim = similarity.max().item()
            
            # 3. Apply Heatmap
            final_rgbs_gpu = apply_overlay_heatmap(
                similarity, 
                base_rgbs_gpu, 
                threshold=current_thresh,
                colormap_name='turbo' 
            )
            
            final_rgbs = final_rgbs_gpu.cpu().numpy().astype(np.float32)

            # 4. Update Viser Scene
            server.scene.add_gaussian_splats(
                "/scene",
                centers=means,
                rgbs=final_rgbs, 
                opacities=opacities,
                covariances=covariances
            )
            
            gui_status.value = f"Max Sim: {max_sim:.4f}"

    # Bind the callback to the GUI elements
    gui_text.on_update(update_scene)
    gui_threshold.on_update(update_scene)

    # Initial trigger to render the scene
    update_scene(None)
    
    print(f"Open your browser at: http://localhost:{args.port}")
    print(f"------------------------------------------------\n")
    
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive Heatmap Overlay for Dr.Splat")
    lp = ModelParams(parser)
    op = PipelineParams(parser)
    
    parser.add_argument("--query", type=str, default="", help="Initial text to search for (optional)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Initial similarity threshold") 
    parser.add_argument("--pq_index", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))