from huggingface_hub import hf_hub_download
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# HuggingFace token — only needed for private/gated repos
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # or set directly: HF_TOKEN = "hf_..."

def download_if_missing(repo_id, filename, local_dir, repo_type="model", token=None):
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename}")
        return
    print(f"  [download] {filename}...")
    hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token
    )

# --- SynBrain: pre-computed eval outputs ---
print("[SynBrain] Eval outputs:")
for file in [
    "fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode/single_s1_vs1_all_clipvoxels_recon.pt",
    "fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode/single_s1_vs1_all_clipvoxels.pt",
    "fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode/single_s1_vs1_all_recon_fmri.pt",
    "fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode/single_s1_vs1_all_recon_mindeye2.pt",
    "all_clipvoxels.pt",
    "all_images.pt",
]:
    download_if_missing("MichaelMaiii/SynBrain", file, os.path.join(ROOT_DIR, "SynBrain_Sub1"))

# --- SynBrain: model checkpoints ---
print("[SynBrain] Checkpoints:")
for file in [
    "checkpoint/vae-nsd-s1-vs1-bs24-350/last.pth",
    "checkpoint/fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode/last.pt",
]:
    download_if_missing("MichaelMaiii/SynBrain", file, ROOT_DIR)

# --- SynBrain: processed fMRI test data ---
print("[SynBrain] fMRI test data:")
download_if_missing("MichaelMaiii/SynBrain", "data/nsd_test_fmri_scale_sub1.npy", ROOT_DIR)

# --- MindEye2: subject-1 encoder checkpoint ---
print("[MindEye2] Encoder checkpoint:")
download_if_missing(
    "pscotti/mindeyev2",
    "train_logs/final_subj01_pretrained_40sess_24bs/last.pth",
    ROOT_DIR,
    repo_type="dataset",
    token=HF_TOKEN
)

# --- MindEye2: SDXL UnClip decoder weights ---
os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)
print("[MindEye2] SDXL UnClip weights:")
download_if_missing(
    "pscotti/mindeyev2",
    "unclip6_epoch0_step110000.ckpt",
    os.path.join(ROOT_DIR, "checkpoints"),
    repo_type="dataset",
    token=HF_TOKEN
)

print("\nDone.")
print(f"Checkpoints:  {ROOT_DIR}/checkpoint/")
print(f"fMRI data:    {ROOT_DIR}/data/nsd_test_fmri_scale_sub1.npy")
print(f"SDXL UnClip:  {ROOT_DIR}/checkpoints/unclip6_epoch0_step110000.ckpt")
print(f"Eval outputs: {ROOT_DIR}/SynBrain_Sub1/")