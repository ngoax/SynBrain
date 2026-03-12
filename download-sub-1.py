from huggingface_hub import hf_hub_download
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# --- SynBrain: Stage 1 (BrainVAE) checkpoint ---
print("[SynBrain] Stage 1 checkpoint:")
download_if_missing("MichaelMaiii/SynBrain", "checkpoint/vae-nsd-s1-vs1-bs24-350/last.pth", ROOT_DIR)

print("\nDone.")
print(f"Checkpoint: {ROOT_DIR}/checkpoint/vae-nsd-s1-vs1-bs24-350/last.pth")