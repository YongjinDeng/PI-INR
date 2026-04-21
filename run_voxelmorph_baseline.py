"""
VoxelMorph 3D Baseline for Deformable Image Registration
=========================================================
This script implements a VoxelMorph baseline for 3D CT-CBCT registration
on pancreatic cancer patients. It serves as a deep learning baseline
comparison for the PI-INR framework.

Key features:
- Patient-specific (zero-shot) optimization
- Downsampling for computational efficiency
- NCC loss + smoothness regularization
- Folding rate (negative Jacobian) computation

Author: Yongjin Deng
Affiliation: First Affiliated Hospital, Sun Yat-sen University
License: MIT
"""

import os
import gc
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')


# ================== Command Line Arguments ==================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='VoxelMorph Baseline for 3D CT-CBCT Registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_voxelmorph_baseline.py
  python run_voxelmorph_baseline.py --data_root /path/to/data --max_patients 20
  python run_voxelmorph_baseline.py --device cuda --epochs 50
        """
    )
    
    # Data paths
    parser.add_argument('--data_root', type=str,
                        default=r"X:\data\manifest-1661266724052\Pancreatic-CT-CBCT-SEG",
                        help='Root directory of the dataset')
    parser.add_argument('--result_dir', type=str,
                        default=r"X:\results\VoxelMorph_Baseline_Fast",
                        help='Directory to save results')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to save CSV results (default: result_dir/VoxelMorph_Results.csv)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'], help='Device to use for computation')
    
    # Image processing
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[2.0, 1.0, 1.0],
                        metavar=('Z', 'Y', 'X'), help='Target voxel spacing in mm')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Downsampling factor for memory efficiency')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs per patient')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--smooth_penalty', type=float, default=0.01,
                        help='Weight for smoothness regularization (low = allow folding)')
    
    # Evaluation
    parser.add_argument('--max_patients', type=int, default=40,
                        help='Maximum number of patients to process')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print progress every N epochs')
    parser.add_argument('--ssim_samples', type=int, default=20,
                        help='Number of slices to sample for SSIM computation')
    
    # Misc
    parser.add_argument('--force_rerun', action='store_true', default=True,
                        help='Delete existing results and rerun all patients')
    
    return parser.parse_args()


# ================== Configuration ==================

def get_config(args):
    """Convert parsed arguments to configuration dictionary."""
    config = {
        "DATA_ROOT": args.data_root,
        "RESULT_DIR": args.result_dir,
        "CSV_PATH": args.csv_path if args.csv_path else 
                    os.path.join(args.result_dir, "VoxelMorph_Results.csv"),
        "DEVICE": args.device,
        "TARGET_SPACING": tuple(args.target_spacing),
        "DOWNSAMPLE_FACTOR": args.downsample_factor,
        "EPOCHS": args.epochs,
        "LR": args.lr,
        "SMOOTH_PENALTY": args.smooth_penalty,
        "MAX_PATIENTS": args.max_patients,
        "PRINT_EVERY": args.print_every,
        "SSIM_SAMPLES": args.ssim_samples,
        "FORCE_RERUN": args.force_rerun,
    }
    return config


# ================== U-Net Architecture ==================

class EfficientUNet(nn.Module):
    """Memory-efficient 3D U-Net for deformable registration."""
    
    def __init__(self):
        super().__init__()
        # Encoder (downsampling path)
        self.e1 = nn.Sequential(nn.Conv3d(2, 8, 3, padding=1), nn.LeakyReLU(0.2))
        self.e2 = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.e3 = nn.Sequential(nn.Conv3d(16, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.e4 = nn.Sequential(nn.Conv3d(16, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        
        # Decoder (upsampling path with skip connections)
        self.d4 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.LeakyReLU(0.2))
        self.d3 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.LeakyReLU(0.2))
        self.d2 = nn.Sequential(nn.Conv3d(24, 8, 3, padding=1), nn.LeakyReLU(0.2))
        self.d1 = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1), nn.LeakyReLU(0.2))
        
        # Output layer (displacement field)
        self.flow = nn.Conv3d(8, 3, 3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 2, D, H, W) concatenating moving and fixed images
        
        Returns:
            flow: Displacement field of shape (B, 3, D, H, W)
        """
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        
        # Decoder with skip connections
        d4 = F.interpolate(e4, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.d4(torch.cat([d4, e3], 1))
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.d3(torch.cat([d3, e2], 1))
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.d2(torch.cat([d2, e1], 1))
        
        d1 = self.d1(d2)
        return self.flow(d1)


class SpatialTransformer(nn.Module):
    """Differentiable spatial transformer for 3D images."""
    
    def __init__(self, size):
        """Initialize spatial transformer.
        
        Args:
            size: Tuple of (D, H, W) specifying the output volume size
        """
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).float().unsqueeze(0)
        self.register_buffer('grid', grid)
    
    def forward(self, src, flow):
        """Apply displacement field to source image.
        
        Args:
            src: Source image tensor of shape (B, C, D, H, W)
            flow: Displacement field of shape (B, 3, D, H, W)
        
        Returns:
            warped: Warped image of shape (B, C, D, H, W)
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(3):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')


# ================== Loss Functions ==================

def fast_ncc_loss(I, J):
    """Simplified normalized cross-correlation loss.
    
    Args:
        I, J: Input tensors of the same shape
    
    Returns:
        Negative NCC value (to minimize)
    """
    I_mean = I.mean()
    J_mean = J.mean()
    I_norm = I - I_mean
    J_norm = J - J_mean
    numerator = (I_norm * J_norm).sum()
    denominator = torch.sqrt((I_norm * I_norm).sum() * (J_norm * J_norm).sum() + 1e-5)
    return -numerator / denominator


def gradient_loss(flow):
    """Smoothness penalty on deformation field.
    
    Args:
        flow: Displacement field of shape (B, 3, D, H, W)
    
    Returns:
        Mean squared gradient magnitude
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)) / 3.0


# ================== Data Loading ==================

def load_patient_data_fast(patient_path, target_spacing=(2.0, 1.0, 1.0)):
    """Load and preprocess a single patient's CT, CBCT, and dose data.
    
    Args:
        patient_path: Path to patient directory containing DICOM files
        target_spacing: Target voxel spacing (Z, Y, X) in mm
    
    Returns:
        fixed: Reference CT tensor (1, 1, D, H, W)
        moving: Moving CBCT tensor (1, 1, D, H, W)
        dose: Dose distribution tensor (1, 1, D, H, W)
        Returns None if loading fails
    """
    ct_series, dose_path = [], None
    
    # Find CT series and dose file
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
            if modality == 'CT' and len(dcm_files) > 20:
                ct_series.append(root)
            elif modality == 'RTDOSE':
                dose_path = os.path.join(root, dcm_files[0])
        except:
            continue
    
    ct_series.sort()
    if len(ct_series) < 2:
        return None
    
    # Read images using SimpleITK
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[1]))
    moving = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    
    # Rigid alignment (center-of-mass)
    fixed_center = np.array(fixed.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in fixed.GetSize()]))
    moving_center = np.array(moving.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in moving.GetSize()]))
    translation = (moving_center - fixed_center).tolist()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(sitk.TranslationTransform(3, translation))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    moving_aligned = resampler.Execute(moving)
    
    # Resample to target spacing
    def resample(img, default_val):
        orig_spacing = img.GetSpacing()
        orig_size = img.GetSize()
        target_size = [
            max(1, int(round(orig_size[0] * orig_spacing[0] / target_spacing[0]))),
            max(1, int(round(orig_size[1] * orig_spacing[1] / target_spacing[1]))),
            max(1, int(round(orig_size[2] * orig_spacing[2] / target_spacing[2])))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(default_val)
        return resampler.Execute(img)
    
    fixed = resample(fixed, -1000)
    moving_aligned = resample(moving_aligned, -1000)
    
    # Load dose distribution
    if dose_path and os.path.exists(dose_path):
        try:
            dose_img = sitk.ReadImage(dose_path)
            dose_img = resample(dose_img, 0)
        except:
            dose_img = sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32))
    else:
        dose_img = sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32))
    
    # Convert to tensors and normalize
    def to_tensor(img, is_dose=False):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if is_dose:
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        else:
            arr = np.clip((arr + 1000) / 2000.0, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    
    return to_tensor(fixed, False), to_tensor(moving_aligned, False), to_tensor(dose_img, True)


def pre_scan_patients(data_root):
    """Return list of patient IDs in the dataset."""
    all_patients = sorted([p for p in os.listdir(data_root) 
                          if p.startswith("Pancreas") and os.path.isdir(os.path.join(data_root, p))])
    return all_patients


# ================== Metrics ==================

def compute_folding_ratio(flow):
    """Compute percentage of voxels with negative Jacobian determinant.
    
    Args:
        flow: Displacement field tensor (1, 3, D, H, W)
    
    Returns:
        folding_ratio: Percentage of voxels with det(J) <= 0
    """
    flow_np = flow.squeeze(0).cpu().numpy()
    grad_x = np.gradient(flow_np[0], axis=(0, 1, 2))
    grad_y = np.gradient(flow_np[1], axis=(0, 1, 2))
    grad_z = np.gradient(flow_np[2], axis=(0, 1, 2))
    
    # Jacobian determinant: det(J) = det(I + ∇u)
    det = ((1 + grad_x[0]) * ((1 + grad_y[1]) * (1 + grad_z[2]) - grad_y[2] * grad_z[1]) -
           grad_x[1] * (grad_y[0] * (1 + grad_z[2]) - grad_y[2] * grad_z[0]) +
           grad_x[2] * (grad_y[0] * grad_z[1] - (1 + grad_y[1]) * grad_z[0]))
    
    return np.sum(det <= 0) / det.size * 100


def compute_ssim_uniform(fixed_np, warped_np, n_samples=20):
    """Compute SSIM by uniformly sampling slices throughout the volume.
    
    This avoids the bias of only sampling the first few slices
    (which are often empty air regions).
    
    Args:
        fixed_np: Fixed (reference) image as numpy array (D, H, W)
        warped_np: Warped image as numpy array (D, H, W)
        n_samples: Number of slices to sample
    
    Returns:
        mean_ssim: Average SSIM across sampled slices
    """
    depth = fixed_np.shape[0]
    if depth <= n_samples:
        indices = range(depth)
    else:
        step = depth // n_samples
        indices = range(step // 2, depth, step)  # start from middle to avoid air
    
    ssim_list = []
    for z in indices:
        if fixed_np[z].std() > 0.01:
            try:
                ssim_val = ssim(fixed_np[z], warped_np[z], data_range=1.0)
                ssim_list.append(ssim_val)
            except:
                continue
    
    return np.mean(ssim_list) if ssim_list else 0


# ================== Main Function ==================

def main():
    """Main entry point for VoxelMorph baseline."""
    
    # Parse command line arguments
    args = parse_args()
    config = get_config(args)
    
    print("=" * 60)
    print(" VoxelMorph Baseline for Deformable Image Registration")
    print(f" Device: {config['DEVICE']}")
    print(f" Epochs: {config['EPOCHS']} | LR: {config['LR']}")
    print(f" Smooth penalty: {config['SMOOTH_PENALTY']}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config["RESULT_DIR"], exist_ok=True)
    
    # Get patient list
    patients = pre_scan_patients(config["DATA_ROOT"])[:config["MAX_PATIENTS"]]
    print(f"\nFound {len(patients)} patients")
    
    # Handle existing results
    results = []
    if os.path.exists(config["CSV_PATH"]):
        if config["FORCE_RERUN"]:
            os.remove(config["CSV_PATH"])
            print("Deleted old results, starting fresh...")
        else:
            existing = pd.read_csv(config["CSV_PATH"])
            results = existing.to_dict('records')
            print(f"Loaded {len(results)} existing results, resuming...")
    
    total_start = time.time()
    
    for idx, pid in enumerate(patients):
        # Skip already processed patients
        if not config["FORCE_RERUN"] and any(r.get('Patient_ID') == pid for r in results):
            print(f"\n[{idx + 1}/{len(patients)}] ⏭️  Skipping {pid} (already processed)")
            continue
        
        print(f"\n{'=' * 50}")
        print(f"[{idx + 1}/{len(patients)}] Processing {pid}")
        print(f"{'=' * 50}")
        start_time = time.time()
        
        try:
            # Load data
            print("  Loading data...")
            fixed_orig, moving_orig, dose_orig = load_patient_data_fast(
                os.path.join(config["DATA_ROOT"], pid),
                config["TARGET_SPACING"]
            )
            if fixed_orig is None:
                print(f"  Failed to load {pid}, skipping")
                continue
            
            # Downsample for memory efficiency
            factor = config["DOWNSAMPLE_FACTOR"]
            new_size = [s // factor for s in fixed_orig.shape[2:]]
            fixed = F.interpolate(fixed_orig, size=new_size, mode='trilinear')
            moving = F.interpolate(moving_orig, size=new_size, mode='trilinear')
            
            fixed = fixed.to(config["DEVICE"])
            moving = moving.to(config["DEVICE"])
            
            print(f"  Size: {fixed_orig.shape[2:]} -> {fixed.shape[2:]}")
            
            # Initialize model
            print("  Initializing model...")
            model = EfficientUNet().to(config["DEVICE"])
            stn = SpatialTransformer(fixed.shape[2:]).to(config["DEVICE"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
            
            # Training loop
            print(f"  Training for {config['EPOCHS']} epochs...")
            print("  ┌" + "─" * 50 + "┐")
            
            for epoch in range(config["EPOCHS"]):
                epoch_start = time.time()
                
                optimizer.zero_grad()
                flow = model(torch.cat([moving, fixed], 1))
                warped = stn(moving, flow)
                loss = fast_ncc_loss(fixed, warped) + config["SMOOTH_PENALTY"] * gradient_loss(flow)
                loss.backward()
                optimizer.step()
                
                epoch_time = time.time() - epoch_start
                
                if (epoch + 1) % config["PRINT_EVERY"] == 0:
                    remaining = (config["EPOCHS"] - epoch - 1) * epoch_time
                    print(f"  │ Epoch {epoch + 1:3d}/{config['EPOCHS']} | Loss: {loss.item():.4f} | "
                          f"Time: {epoch_time:.1f}s | Est. remaining: {remaining / 60:.1f}min")
            
            print("  └" + "─" * 50 + "┘")
            
            # Evaluation
            print("  Evaluating...")
            with torch.no_grad():
                flow = model(torch.cat([moving, fixed], 1))
                flow_full = F.interpolate(flow, size=fixed_orig.shape[2:], mode='trilinear') * factor
                
                stn_orig = SpatialTransformer(fixed_orig.shape[2:]).to(config["DEVICE"])
                fixed_orig = fixed_orig.to(config["DEVICE"])
                moving_orig = moving_orig.to(config["DEVICE"])
                warped_orig = stn_orig(moving_orig, flow_full)
                
                f_np = fixed_orig.squeeze().cpu().numpy()
                m_np = moving_orig.squeeze().cpu().numpy()
                w_np = warped_orig.squeeze().cpu().numpy()
                
                ssim_before = compute_ssim_uniform(f_np, m_np, config["SSIM_SAMPLES"])
                ssim_after = compute_ssim_uniform(f_np, w_np, config["SSIM_SAMPLES"])
                folding = compute_folding_ratio(flow_full)
            
            elapsed = time.time() - start_time
            print(f"\n  Completed in {elapsed / 60:.1f} minutes")
            print(f"  SSIM: {ssim_before:.4f} -> {ssim_after:.4f} (Δ = {ssim_after - ssim_before:.4f})")
            print(f"  Folding rate: {folding:.2f}%")
            
            results.append({
                "Patient_ID": pid,
                "SSIM_Before": float(ssim_before),
                "SSIM_VoxelMorph": float(ssim_after),
                "Folding_VoxelMorph": float(folding),
                "Time_min": elapsed / 60
            })
            
            # Save results incrementally
            pd.DataFrame(results).to_csv(config["CSV_PATH"], index=False)
            
            completed = len(results)
            remaining = len(patients) - completed
            if completed > 0:
                avg_time = (time.time() - total_start) / completed
                eta = remaining * avg_time / 60
                print(f"\n  Progress: {completed}/{len(patients)} | ETA: {eta:.1f} min")
            
            # Clean up memory
            del model, fixed, moving, flow, flow_full, warped_orig
            if config["DEVICE"] == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    total_time = (time.time() - total_start) / 60
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(results)}/{len(patients)} patients")
    print(f"Total time: {total_time:.1f} minutes ({total_time / 60:.1f} hours)")
    
    if results:
        df = pd.DataFrame(results)
        print(f"\nResults Summary:")
        print(f"  SSIM Before: {df['SSIM_Before'].mean():.4f} ± {df['SSIM_Before'].std():.4f}")
        print(f"  SSIM After:  {df['SSIM_VoxelMorph'].mean():.4f} ± {df['SSIM_VoxelMorph'].std():.4f}")
        print(f"  Folding Rate: {df['Folding_VoxelMorph'].mean():.2f}% ± {df['Folding_VoxelMorph'].std():.2f}%")
    
    print(f"\nResults saved to: {config['CSV_PATH']}")
    print("=" * 60)


if __name__ == "__main__":
    main()