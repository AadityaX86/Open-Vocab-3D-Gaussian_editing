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

# =========================================================================================
#  MATH & GEOMETRY HELPERS
# =========================================================================================

def delete_selection(gaussians, mask):
    """
    'Deletes' objects by making them effectively invisible.
    
    MATH: 
    Gaussian Splatting uses a Sigmoid activation for opacity: Opacity = Sigmoid(param).
    Sigmoid(-100) is approx 3.7e-44 (effectively zero).
    We don't actually remove the point from memory (which breaks the optimizer indices),
    we just hide it.
    """
    with torch.no_grad():
        # Set internal logit to -100 => Opacity becomes 0.0
        gaussians._opacity[mask] = -100.0

def color_selection(gaussians, mask, target_rgb):
    """
    Paints objects by overwriting the 0th-order Spherical Harmonic (SH).
    
    MATH:
    The Base Color (DC) in Gaussian Splatting is stored as the 0th SH coefficient.
    RGB = 0.5 + C0 * SH_0
    where C0 = 0.28209 (constant for 0th order SH).
    
    Therefore, to get a specific RGB:
    SH_0 = (Target_RGB - 0.5) / 0.28209
    """
    SH_C0 = 0.28209479177387814
    target_sh = (target_rgb - 0.5) / SH_C0
    
    with torch.no_grad():
        # Create tensor on correct device
        new_color = torch.tensor(target_sh, device=gaussians._features_dc.device).float()
        
        # Apply to Masked Gaussians
        # We only modify _features_dc (Base Color), leaving view-dependent effects alone (or resetting them)
        gaussians._features_dc[mask, 0, :] = new_color

def move_selection(gaussians, mask, offset_vector):
    """
    Translates selected objects in 3D space.
    """
    with torch.no_grad():
        offset = torch.tensor(offset_vector, device=gaussians._xyz.device).float()
        gaussians._xyz[mask] += offset

def get_colormap_safe(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)

def apply_overlay_heatmap(similarities, base_rgbs, threshold=0.2, colormap_name='turbo'):
    """
    Generates the visualization:
    - Matches > Threshold get colored by similarity (Red = High Match)
    - Matches <= Threshold stay original photo-realistic color
    """
    # 1. Clean Inputs
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
    max_sim = similarities.max()
    
    # Optimization: If no good matches, don't waste time computing colors
    if max_sim <= threshold:
        return base_rgbs 

    # 2. Normalize Scores [Threshold -> Max] maps to [0 -> 1]
    # We add 1e-8 to avoid division by zero
    norm_sim = similarities / (max_sim + 1e-8)
    norm_sim = torch.clamp(norm_sim, 0, 1)

    # 3. Generate Heatmap Colors (CPU work usually)
    cmap = get_colormap_safe(colormap_name)
    heatmap_colors_cpu = cmap(norm_sim.cpu().numpy())[:, :3]
    heatmap_colors = torch.tensor(heatmap_colors_cpu, device=base_rgbs.device, dtype=torch.float32)
    
    # 4. Blend Logic
    # Mask is 1.0 for matches, 0.0 for background
    mask = (similarities > threshold).float().unsqueeze(-1).to(base_rgbs.device)
    
    # Linear Interpolation: result = color1 * mask + color2 * (1-mask)
    final_colors = heatmap_colors * mask + base_rgbs * (1 - mask)
    return final_colors

# =========================================================================================
#  MAIN APPLICATION
# =========================================================================================

def main(args, dataset_args, pipeline_args):
    # --- 1. SETUP & LOAD MODELS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print(f"\n[1/5] Loading OpenCLIP Model (ViT-B-16)...")
    model_name = "ViT-B-16" 
    pretrained_source = "laion2b_s34b_b88k"
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_source, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    print(f"[2/5] Loading Gaussian Model from {args.model_path}...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    checkpoint_path = os.path.join(args.model_path, "chkpnt0.pth") 
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, args, mode='test')
    
    # --- 2. DECODE LANGUAGE FEATURES ---
    print("[3/5] Loading FAISS Index and Decoding Features...")
    index = faiss.read_index(args.pq_index)
    
    language_features_idx = gaussians._language_feature.clone()
    
    # Check for invalid feature indices (sentinels)
    check_valid = torch.sum(language_features_idx, 1)
    invalid_index = check_valid == 255 * (index.coarse_code_size() + index.code_size)
    
    # Decode on CPU (FAISS limitation) then move to GPU
    decoded_features_cpu = np.zeros((language_features_idx.shape[0], 512), dtype=np.float32)
    valid_mask_cpu = (invalid_index.cpu() == False).numpy()
    
    decoded_features_cpu[valid_mask_cpu] = index.sa_decode(
        language_features_idx[valid_mask_cpu].cpu().numpy()
    )
    
    gaussian_features = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float32)
    del decoded_features_cpu
    torch.cuda.empty_cache()
    
    # Normalize features for Cosine Similarity
    norm = gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features = gaussian_features / (norm + 1e-9)

    # --- 3. START VISER SERVER ---
    print("[4/5] Starting Viser Server...")
    server = viser.ViserServer(port=args.port)
    
    # State dictionary to persist data between callbacks
    state = {
        "current_mask": None,   # Boolean tensor of selected points
        "last_query": ""        # Last text searched
    }

    # --- GUI LAYOUT ---
    with server.gui.add_folder("Semantic Editor"):
        # Search Section
        gui_query = server.gui.add_text("Target Object", initial_value="bicycle")
        
        gui_threshold = server.gui.add_slider(
            "Selection Threshold", 
            min=0.0, 
            max=1.0, 
            step=0.01, 
            initial_value=0.22
        )
        
        # Actions Section
        gui_color_picker = server.gui.add_rgb("Paint Color", initial_value=(1.0, 0.0, 0.0))
        btn_paint = server.gui.add_button("Apply Paint")
        btn_delete = server.gui.add_button("Delete Selection")
        
        # Move Section
        with server.gui.add_folder("Transform"):
            slider_x = server.gui.add_slider("X Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
            slider_y = server.gui.add_slider("Y Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
            slider_z = server.gui.add_slider("Z Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
            btn_move = server.gui.add_button("Apply Move")

        gui_status = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    # --- CORE RENDERING LOGIC ---
    def render_scene(highlight_mask=None, similarity_scores=None):
        """
        Extracts current state of Gaussians and pushes to Viser.
        CRITICAL FIX: Filters out invisible points to prevent sorting artifacts.
        """
        
        # 1. Get raw properties from Gaussian Model
        # Note: These getters return the processed values (e.g., sigmoid applied to opacity)
        opacities = gaussians.get_opacity # Shape [N, 1]
        
        # --- CULLING LOGIC (Fixes Artifacts) ---
        # We create a CPU mask of points that are actually visible (opacity > 0.05).
        # If we send 0-opacity points to Viser, the depth sorter gets confused and creates "ghosts".
        vis_mask = (opacities.squeeze() > 0.05).cpu().numpy()
        
        # If everything is deleted, safeguard
        if not np.any(vis_mask):
            return 

        # 2. Extract Geometry (Apply Culling Mask)
        # We detach() to stop PyTorch gradients, cpu() to move to RAM, numpy() for Viser
        means = gaussians.get_xyz.detach().cpu().numpy()[vis_mask]
        opacities_vis = opacities.detach().cpu().numpy()[vis_mask]
        
        scales = gaussians.get_scaling.detach().cpu().numpy()[vis_mask]
        quats = gaussians.get_rotation.detach().cpu().numpy()[vis_mask]
        
        # 3. Calculate Covariances (Ellipsoid shapes)
        # This math converts Rotation/Scale -> 3x3 Covariance Matrix for rendering
        Rs = tf.SO3(quats).as_matrix()
        covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)

        # 4. Extract Colors & Apply Heatmap
        shs = gaussians.get_features
        # Convert SH -> RGB (Level 0 approximation)
        base_rgbs_gpu = (shs[:, 0, :].detach() * 0.28209479177387814 + 0.5).clamp(0, 1)

        final_rgbs_gpu = base_rgbs_gpu
        
        # Apply Heatmap Overlay if requested
        if similarity_scores is not None and highlight_mask is not None:
             final_rgbs_gpu = apply_overlay_heatmap(
                similarity_scores, 
                base_rgbs_gpu, 
                threshold=gui_threshold.value
            )
        
        # Apply Culling Mask to Colors
        rgbs_vis = final_rgbs_gpu.cpu().numpy()[vis_mask]

        # 5. Send to Viser
        # Using the same name "/scene" ensures we update the existing object, not create duplicates
        server.scene.add_gaussian_splats(
            "/scene",
            centers=means,
            rgbs=rgbs_vis,
            opacities=opacities_vis,
            covariances=covariances
        )

    # --- INTERACTION CALLBACKS ---
    
    def update_selection(_):
        """Logic: Text Input -> CLIP Embedding -> Cosine Sim -> Update Mask -> Render"""
        query_text = gui_query.value
        threshold = gui_threshold.value
        state["last_query"] = query_text

        # If text is empty, show original scene
        if not query_text:
            render_scene(highlight_mask=None, similarity_scores=None)
            gui_status.value = "Showing Original"
            return

        gui_status.value = f"Processing '{query_text}'..."
        
        with torch.no_grad():
            # 1. Encode Text
            text_tokens = tokenizer([query_text]).to(device)
            text_features = clip_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 2. Dot Product (Cosine Similarity)
            # Shapes: [N, 512] @ [512, 1] -> [N]
            similarity = (gaussian_features @ text_features.T).squeeze()
            similarity = torch.nan_to_num(similarity, nan=0.0)
            
            # 3. Update Global Selection Mask
            state["current_mask"] = (similarity > threshold)
            
            max_sim = similarity.max().item()
            gui_status.value = f"Max Score: {max_sim:.4f}"
            
            # 4. Render
            render_scene(highlight_mask=state["current_mask"], similarity_scores=similarity)

    def handle_delete(_):
        if state["current_mask"] is None: return
        print(f"Deleting selection for '{state['last_query']}'...")
        
        delete_selection(gaussians, state["current_mask"])
        
        # Force a re-calculation and render
        # We pass 'None' because the event handler expects an argument
        update_selection(None)

    def handle_paint(_):
        if state["current_mask"] is None: return
        rgb = gui_color_picker.value
        print(f"Painting selection {rgb}...")
        
        color_selection(gaussians, state["current_mask"], np.array(rgb))
        update_selection(None)

    def handle_move(_):
        if state["current_mask"] is None: return
        offset = [slider_x.value, slider_y.value, slider_z.value]
        print(f"Moving selection by {offset}...")
        
        move_selection(gaussians, state["current_mask"], offset)
        
        # Reset sliders to 0 so we don't apply the move again accidentally
        slider_x.value = 0.0
        slider_y.value = 0.0
        slider_z.value = 0.0
        
        update_selection(None)

    # --- BINDING ---
    # Connect GUI elements to functions
    gui_query.on_update(update_selection)
    gui_threshold.on_update(update_selection)
    btn_delete.on_click(handle_delete)
    btn_paint.on_click(handle_paint)
    btn_move.on_click(handle_move)

    # Initial Render
    print("Pre-loading scene...")
    render_scene()
    
    print(f"SUCCESS! Interactive Editor running on http://localhost:{args.port}")
    print(f"------------------------------------------------\n")
    
    # Keep main thread alive
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive Semantic Editor for Gaussian Splatting")
    lp = ModelParams(parser)
    op = PipelineParams(parser)
    
    parser.add_argument("--query", type=str, default="bicycle", help="Initial query")
    parser.add_argument("--threshold", type=float, default=0.22, help="Similarity threshold") 
    parser.add_argument("--pq_index", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))