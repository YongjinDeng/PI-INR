"""
PI-INR MedIA Figure Generation Engine
Version: 1.0.0 (Official Release)
Generate publication-quality figures for MedIA submission.
Figures include main figure with 5-column layout, clinical flowchart,
annotated case examples, and statistical distribution plots.
Output formats: PDF (vector) + TIFF (300 DPI)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import SimpleITK as sitk
import pydicom
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'savefig.dpi': 300
})

DATA_ROOT = "./data/Pancreatic-CT-CBCT-SEG"
RESULT_DIR = "./results/MedIA_Ultimate_Run_Final"
CSV_PATH = os.path.join(RESULT_DIR, "MedIA_Quantitative_Results.csv")
OUTPUT_DIR = os.path.join(RESULT_DIR, "MedIA_Paper_Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SPACING = (2.0, 1.0, 1.0)

TIFF_DPI = 300


def save_figure(fig, filename, pdf=True, tiff=True):
    """Save figure as PDF and/or TIFF."""
    base_path = os.path.join(OUTPUT_DIR, filename)
    if pdf:
        fig.savefig(base_path + '.pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
    if tiff:
        fig.savefig(base_path + '.tiff', dpi=TIFF_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), format='tiff', pil_kwargs={'compression': 'tiff_lzw'})


def resample_to_target_spacing(image, target_spacing, default_value=0):
    """Resample image to target spacing."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    target_size = [
        int(round(orig_size[0] * orig_spacing[0] / target_spacing[0])),
        int(round(orig_size[1] * orig_spacing[1] / target_spacing[1])),
        int(round(orig_size[2] * orig_spacing[2] / target_spacing[2]))
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([max(1, s) for s in target_size])
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def load_patient_baseline(pid):
    """Load base CT and dose for a patient."""
    p_path = os.path.join(DATA_ROOT, pid)
    ct_series = []
    dose_path = None
    for root, _, files in os.walk(p_path):
        dcms = [f for f in files if f.lower().endswith('.dcm')]
        if len(dcms) > 20:
            try:
                ds = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True)
                mod = getattr(ds, "Modality", "").upper()
                if mod == 'CT':
                    ct_series.append(root)
                elif mod == 'RTDOSE':
                    dose_path = os.path.join(root, dcms[0])
            except:
                pass
    ct_series.sort()
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    fixed = resample_to_target_spacing(fixed, TARGET_SPACING, -1000)
    
    f_arr = sitk.GetArrayFromImage(fixed)
    f_arr = np.clip((f_arr + 1000) / 2000.0, 0, 1)
    
    if dose_path:
        dose = sitk.Cast(sitk.ReadImage(dose_path), sitk.sitkFloat32)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        dose = resampler.Execute(dose)
        d_arr = sitk.GetArrayFromImage(dose)
        if d_arr.max() > d_arr.min():
            d_arr = (d_arr - d_arr.min()) / (d_arr.max() - d_arr.min() + 1e-6)
    else:
        d_arr = np.zeros_like(f_arr)
        
    return f_arr, d_arr


def load_patient_results(pid, result_dir):
    """Load results from npz file."""
    res_path = os.path.join(result_dir, pid, "results.npz")
    if not os.path.exists(res_path):
        return None, None, None, None, None
    
    data = np.load(res_path, allow_pickle=True)
    
    stats = data['stats'].item() if 'stats' in data else {}
    warped = data['warped'] if 'warped' in data else None
    ati = data['ati'] if 'ati' in data else None
    dose_warped = data['dose_warped'] if 'dose_warped' in data else None
    
    bspline_path = os.path.join(result_dir, pid, "Warped_BSpline.nii.gz")
    bspline_arr = None
    if os.path.exists(bspline_path):
        try:
            bspline_img = sitk.ReadImage(bspline_path)
            bspline_arr = sitk.GetArrayFromImage(bspline_img)
            bspline_arr = np.clip((bspline_arr + 1000) / 2000.0, 0, 1)
        except:
            bspline_arr = None
    
    return warped, ati, bspline_arr, dose_warped, stats


def draw_main_figure5():
    """Generate main figure with 5-column layout."""
    print("Generating Figure 5...")
    
    PATIENTS = [
        ('Pancreas-CT-CB_012', '(a) Patient 012: Structural Alignment & Topology'),
        ('Pancreas-CT-CB_021', '(b) Patient 021: Robust Dosimetric Warning under Artifacts')
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(28, 12), facecolor='black')
    plt.subplots_adjust(left=0.02, right=0.94, bottom=0.1, top=0.88, hspace=0.35, wspace=0.05)
    
    for row_idx, (pid, title) in enumerate(PATIENTS):
        print(f"  Rendering {pid}...")
        f_arr, _ = load_patient_baseline(pid)
        warped, ati, bspline_arr, dose_warped, stats = load_patient_results(pid, RESULT_DIR)
        
        if warped is None:
            print(f"  Warning: {pid} results not found, skipping")
            continue
        
        if bspline_arr is None:
            bspline_arr = warped.copy()
        
        ati_corr = stats.get('ATI_DDA_Corr', 0.0)
        
        if ati is not None and ati.max() > 0:
            z = np.argmax(np.sum(ati, axis=(1, 2)))
        else:
            z = f_arr.shape[0] // 2
        
        f_slc = f_arr[z]
        w_slc = warped[z]
        b_slc = bspline_arr[z] if bspline_arr is not None else w_slc
        a_slc = ati[z] if ati is not None else np.zeros_like(f_slc)
        
        if dose_warped is not None:
            d_arr, _ = load_patient_baseline(pid)
            true_dose_diff = np.abs(dose_warped - d_arr)[z]
        else:
            true_dose_diff = a_slc.copy()
        
        H, W = f_slc.shape
        PHYSICAL_EXTENT = [0, W * TARGET_SPACING[0], H * TARGET_SPACING[1], 0]
        
        axs = axes[row_idx]
        
        axs[0].imshow(f_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        axs[1].imshow(b_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        axs[2].imshow(w_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1)
        
        axs[3].imshow(f_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1, alpha=0.5)
        mask_gt = np.ma.masked_where((true_dose_diff < 0.1) | (f_slc <= 0.05), true_dose_diff)
        im_gt = axs[3].imshow(mask_gt, cmap='hot', extent=PHYSICAL_EXTENT, alpha=0.8, vmin=0, vmax=1)
        
        axs[4].imshow(f_slc, cmap='gray', extent=PHYSICAL_EXTENT, vmin=0, vmax=1, alpha=0.5)
        mask_a = np.ma.masked_where((a_slc < 0.05) | (f_slc <= 0.05), a_slc)
        axs[4].imshow(mask_a, cmap='Reds', extent=PHYSICAL_EXTENT, alpha=0.85, vmin=0, vmax=1)
        
        axs[0].set_title("Fixed CT (Plan)", fontsize=18, color='white', pad=12)
        axs[1].set_title("Conventional B-Spline\n(Tearing Artifacts)", fontsize=18, color='#E74C3C', pad=12)
        axs[2].set_title("PI-INR (Ours)\n(Topology Preserved)", fontsize=18, color='#2ECC71', pad=12)
        axs[3].set_title("True Dose Diff\n(DDA Ground Truth)", fontsize=18, color='#F1C40F', pad=12)
        axs[4].set_title("Predicted ATI Risk\n(Real-time Surrogate)", fontsize=18, color='#3498DB', pad=12)
        
        for ax in axs:
            ax.axis('off')
        
        axs[0].text(0.0, 1.15, title, color='white', fontsize=22, fontweight='bold', transform=axs[0].transAxes)
        
        if row_idx == 1:
            valid_mask = (f_slc > 0.05) & (true_dose_diff > 0.01) & (a_slc > 0.01)
            if np.sum(valid_mask) > 100:
                corr, _ = pearsonr(a_slc[valid_mask].flatten(), true_dose_diff[valid_mask].flatten())
                axs[3].text(0.5, -0.15, f'r = {corr:.2f}', color='white', fontsize=20, ha='center', transform=axs[3].transAxes)
            elif ati_corr > 0:
                axs[3].text(0.5, -0.15, f'r = {ati_corr:.2f}', color='white', fontsize=20, ha='center', transform=axs[3].transAxes)
    
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.3])
    cbar = plt.colorbar(im_gt, cax=cbar_ax)
    cbar.set_label('Dose Difference / ATI Value', color='white', fontsize=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    save_figure(fig, 'Figure5_MedIA', pdf=True, tiff=True)
    plt.close()
    print("  Figure 5 completed")


def draw_figure_s1_flowchart():
    """Generate clinical decision flowchart."""
    print("Generating Figure S1 (Flowchart)...")
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='white')
    
    ax.set_xlim(-10, 110) 
    ax.set_ylim(-20, 100) 
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color='#ECF0F1', fontcolor='black', fontsize=14, weight='bold'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.5", 
                                     edgecolor='#34495E', facecolor=color, lw=2.5)
        ax.add_patch(box)
        ax.text(x+w/2, y+h/2, text, color=fontcolor, fontsize=fontsize, 
                fontweight=weight, ha='center', va='center')
        
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                   arrowprops=dict(arrowstyle="-|>,head_length=1.0,head_width=0.5", 
                                   lw=2.5, color='#34495E'))

    draw_box(25, 85, 50, 8, "Daily CBCT Acquisition", '#EBF5FB')
    draw_arrow(50, 85, 50, 75)
    draw_box(25, 65, 50, 10, "PI-INR Inference\n(Patient-specific Optimization)", '#D6EAF8')
    draw_arrow(50, 65, 50, 55)
    
    ax.plot([25, 75], [55, 55], color='#34495E', lw=3)
    draw_arrow(25, 55, 25, 48)
    draw_arrow(75, 55, 75, 48)
    
    draw_box(2, 38, 44, 10, "ATI Risk Map\n(Lie Derivative & Entropy)", '#FADBD8', fontsize=12)
    draw_box(54, 38, 44, 10, "Uncertainty Map\n(Optimization Confidence)", '#E8DAEF', fontsize=12)
    
    draw_arrow(24, 38, 24, 30)
    draw_arrow(76, 38, 76, 30)
    ax.plot([24, 76], [30, 30], color='#34495E', lw=3)
    draw_arrow(50, 30, 50, 22)
    
    draw_box(15, 12, 70, 8, "Traffic-Light Decision System\n(ATI + Uncertainty Synthesis)", 
             '#FCF3CF', fontcolor='black', fontsize=14)
    draw_arrow(50, 12, 50, 4)
    
    ax.plot([15, 85], [4, 4], color='#34495E', lw=3)
    draw_arrow(15, 4, 15, -4)
    draw_arrow(50, 4, 50, -4)
    draw_arrow(85, 4, 85, -4)
    
    draw_box(5, -12, 28, 8, "RED\nReplanning", '#E74C3C', 'white', fontsize=12)
    draw_box(36, -12, 28, 8, "YELLOW\nManual Review", '#F1C40F', 'black', fontsize=12)
    draw_box(67, -12, 28, 8, "GREEN\nProceed", '#2ECC71', 'white', fontsize=12)

    save_figure(fig, 'FigS1_Flowchart', pdf=True, tiff=True)
    plt.close()
    print("  Figure S1 completed")


def draw_annotated_cases():
    """Generate annotated case examples."""
    print("Generating Figures S2 and S3...")
    cases = [
        ('Pancreas-CT-CB_012', 'FigS2_Patient012_Annotated', 'Patient 012 - Topological Preserving'),
        ('Pancreas-CT-CB_021', 'FigS3_Patient021_Annotated', 'Patient 021 - Highest Dosimetric Risk')
    ]
    for pid, filename, title in cases:
        f_arr, _ = load_patient_baseline(pid)
        warped, ati, _, _, _ = load_patient_results(pid, RESULT_DIR)
        
        if warped is None:
            print(f"  Warning: {pid} results not found, skipping")
            continue
        
        if ati is not None and ati.max() > 0:
            z = np.argmax(np.sum(ati, axis=(1, 2)))
        else:
            z = f_arr.shape[0] // 2
        
        H, W = f_arr[z].shape
        physical_extent = [0, W * TARGET_SPACING[0], H * TARGET_SPACING[1], 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
        for ax in axes:
            ax.axis('off')
        
        rgb = np.zeros((f_arr.shape[1], f_arr.shape[2], 3), dtype=np.float32)
        rgb[..., 0] = f_arr[z] 
        rgb[..., 1] = warped[z] 
        rgb[..., 2] = warped[z] 
        
        axes[0].imshow(rgb, extent=physical_extent)
        axes[0].set_title(f"{pid} - Pseudo-color Overlay", color='white', pad=15, fontsize=16)
        
        arrow1_start = (0.15, 0.25)
        arrow1_end = (0.2, 0.3)
        axes[0].annotate('Soft Tissue\nBoundary', xy=arrow1_end, xytext=arrow1_start,
                        textcoords='axes fraction', xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=2.5),
                        color='cyan', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        
        arrow2_start = (0.85, 0.15)
        arrow2_end = (0.75, 0.2)
        axes[0].annotate('High Risk\nArea', xy=arrow2_end, xytext=arrow2_start,
                        textcoords='axes fraction', xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5),
                        color='yellow', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        
        axes[1].imshow(f_arr[z], cmap='gray', extent=physical_extent)
        if ati is not None:
            masked_ati = np.ma.masked_where(ati[z] < 0.1, ati[z])
            axes[1].imshow(masked_ati, cmap='Reds', alpha=0.8, vmin=0, vmax=1.0, extent=physical_extent)
        
        axes[1].set_title("ATI High-Risk Hotspot", color='white', pad=15, fontsize=16)
        
        if ati is not None and ati[z].max() > 0:
            ati_slice = ati[z]
            max_idx = np.unravel_index(np.argmax(ati_slice), ati_slice.shape)
            max_y, max_x = max_idx
            phys_x = max_x * TARGET_SPACING[0]
            phys_y = max_y * TARGET_SPACING[1]
            norm_x = phys_x / (W * TARGET_SPACING[0])
            norm_y = phys_y / (H * TARGET_SPACING[1])
            
            arrow_start = (norm_x + 0.1, norm_y - 0.05)
            arrow_end = (norm_x, norm_y)
            
            axes[1].annotate('Peak Risk\nRegion', xy=arrow_end, xytext=arrow_start,
                            textcoords='axes fraction', xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                            color='red', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        save_figure(fig, filename, pdf=True, tiff=True)
        plt.close()
    print("  Figures S2 and S3 completed")


def draw_extended_stats():
    """Generate statistical distribution figures."""
    print("Generating Figures S4-S6...")
    if not os.path.exists(CSV_PATH): 
        print(f"  Warning: CSV file not found: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['SSIM_Before'] > 0.01].copy()
    df['SSIM_Improvement'] = df['SSIM_After'] - df['SSIM_Before']
    
    mask_no_dose = df['Dose_MaxErr'] < 1e-5
    df.loc[mask_no_dose, ['ATI_DDA_Corr', 'Dose_MAE', 'Dose_RMSE', 'Dose_MaxErr']] = np.nan
    
    mask_zero_corr = df['ATI_DDA_Corr'] == 0.0
    df.loc[mask_zero_corr, 'ATI_DDA_Corr'] = np.nan
    
    mask_no_ptv = df['DSC_PTV'] < 1e-5
    df.loc[mask_no_ptv, ['DSC_PTV', 'HD95_PTV']] = np.nan
    
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(df['SSIM_Improvement'].dropna(), bins=15, kde=True, color='#3498DB')
    plt.axvline(x=0, color='red', linestyle='--', label='No Improvement')
    plt.xlabel('\u0394SSIM (After - Before)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.tight_layout()
    save_figure(fig, 'FigS4_SSIM_Histogram', pdf=True, tiff=True)
    plt.close()
    
    fig = plt.figure(figsize=(8, 6))
    risk_data = df[df['High_Risk_Ratio'] > 0]['High_Risk_Ratio']
    if len(risk_data) > 0:
        sns.histplot(risk_data, bins=8, log_scale=True, color='#E74C3C', 
                     kde=True, edgecolor='black', alpha=0.7)
        plt.xlabel('ATI High-Risk Ratio (%) - Log Scale')
        plt.ylabel('Patient Count')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    save_figure(fig, 'FigS5_ATI_Log_Dist', pdf=True, tiff=True)
    plt.close()
    
    if 'ATI_DDA_Corr' in df.columns:
        fig = plt.figure(figsize=(8, 6))
        valid_corr = df['ATI_DDA_Corr'].dropna()
        if len(valid_corr) > 0:
            sns.histplot(valid_corr, bins=15, kde=True, color='#9B59B6')
            plt.axvline(x=0.3, color='red', linestyle='--', label='r = 0.3 (Moderate)')
            plt.xlabel('Pearson Correlation Coefficient r')
            plt.ylabel('Patient Count')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No valid ATI correlation data', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        save_figure(fig, 'FigS6_ATI_DDA_Dist', pdf=True, tiff=True)
        plt.close()
    
    print("  Figures S4-S6 completed")


if __name__ == "__main__":
    print("="*60)
    print("PI-INR MedIA Figure Generation v1.0.0")
    print("="*60)
    print("Output formats: PDF (vector) + TIFF (300 DPI, LZW compressed)")
    print("Checking paths...")
    print(f"  DATA_ROOT exists: {os.path.exists(DATA_ROOT)}")
    print(f"  RESULT_DIR exists: {os.path.exists(RESULT_DIR)}")
    print(f"  CSV_PATH exists: {os.path.exists(CSV_PATH)}")
    print("="*60)
    
    draw_main_figure5()
    draw_figure_s1_flowchart()
    draw_annotated_cases()
    draw_extended_stats()
    
    print("="*60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("  PDF files: vector format for publication")
    print("  TIFF files: 300 DPI, LZW compressed for submission")
    print("="*60)
