"""
Generate minimal dummy data for running SynBrain Stage 1 (BrainVAE) with Subject 1.

Creates the exact .npy files expected by src/dataset.py → multisub_clip_dataset().
Uses --hour=1 by default → 750 training samples needed.

Usage:
    python create_dummy_data.py --data_path ./dummy_data
    python create_dummy_data.py --data_path ./dummy_data --hour 1 --num_test 30
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dummy_data")
    parser.add_argument("--hour", type=int, default=1,
                        help="Number of hours of training data (750 samples per hour)")
    parser.add_argument("--num_voxels", type=int, default=15724,
                        help="Number of voxels for subject 1")
    parser.add_argument("--num_test", type=int, default=30,
                        help="Number of test images (each with 3 trials)")
    parser.add_argument("--clip_seq_len", type=int, default=256)
    parser.add_argument("--clip_emb_dim", type=int, default=1664)
    args = parser.parse_args()

    num_train = 750 * args.hour  # training samples (already 3-trial expanded for subj 1)
    num_test_trials = args.num_test * 3  # 3 trials per test image
    # CLIP train: before 3x expansion → num_train // 3 unique images
    num_clip_train_unique = num_train // 3

    subj_dir = os.path.join(args.data_path, "subj01")
    os.makedirs(subj_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Training fMRI: shape [num_train, num_voxels] ---
    # Values scaled like real data (divided by 2000), typical range ~ [-0.5, 0.5]
    print(f"Creating train fMRI: ({num_train}, {args.num_voxels})")
    train_fmri = rng.normal(0, 0.15, (num_train, args.num_voxels)).astype(np.float32)
    np.save(os.path.join(subj_dir, "nsd_train_fmri_all_scale_sub1.npy"), train_fmri)
    print(f"  Saved. Size: {train_fmri.nbytes / 1e6:.1f} MB")
    del train_fmri

    # --- Training CLIP: shape [num_clip_train_unique, 256, 1664] ---
    # Dataset code does 3x repeat for subj in [1,2,5,7], so save unique images only
    print(f"Creating train CLIP: ({num_clip_train_unique}, {args.clip_seq_len}, {args.clip_emb_dim})")
    train_clip = rng.normal(0, 1, (num_clip_train_unique, args.clip_seq_len, args.clip_emb_dim)).astype(np.float32)
    np.save(os.path.join(subj_dir, "nsd_sdxl_clip_train_sub1.npy"), train_clip)
    print(f"  Saved. Size: {train_clip.nbytes / 1e6:.1f} MB")
    del train_clip

    # --- Test fMRI: shape [num_test_trials, num_voxels] ---
    # Reshaped to (num_test, 3, num_voxels) then averaged in dataset code
    print(f"Creating test fMRI: ({num_test_trials}, {args.num_voxels})")
    test_fmri = rng.normal(0, 0.15, (num_test_trials, args.num_voxels)).astype(np.float32)
    np.save(os.path.join(subj_dir, "nsd_test_fmri_all_scale_sub1.npy"), test_fmri)
    print(f"  Saved. Size: {test_fmri.nbytes / 1e6:.1f} MB")
    del test_fmri

    # --- Test CLIP: shape [num_test, 256, 1664] ---
    print(f"Creating test CLIP: ({args.num_test}, {args.clip_seq_len}, {args.clip_emb_dim})")
    test_clip = rng.normal(0, 1, (args.num_test, args.clip_seq_len, args.clip_emb_dim)).astype(np.float32)
    np.save(os.path.join(subj_dir, "nsd_sdxl_clip_test_sub1.npy"), test_clip)
    print(f"  Saved. Size: {test_clip.nbytes / 1e6:.1f} MB")
    del test_clip

    print(f"\nDummy data created in: {os.path.abspath(subj_dir)}")
    print(f"Total files: 4")
    print(f"\nTo train Stage 1 with this data, run:")
    print(f"  cd {os.path.abspath(os.path.join(args.data_path, '..'))}")
    print(f"  python src/vae/train_vae.py \\")
    print(f"    --data_path {os.path.abspath(args.data_path)} \\")
    print(f"    --save_path ./output \\")
    print(f"    --subject '[1]' --valid-sub 1 --hour {args.hour} \\")
    print(f"    --batch_size 4 --num_epochs 5 --wandb_log False --plot_recon False")


if __name__ == "__main__":
    main()
