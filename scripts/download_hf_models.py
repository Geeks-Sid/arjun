#!/usr/bin/env python3
"""
==============================================================================
MedFM Hugging Face Foundation Models Downloader
Downloads all primary, fallback, and research Hugging Face model weights for MedFM.
==============================================================================
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    print("[ERROR] huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


MODEL_CATALOG: dict[str, dict[str, str]] = {
    # 2D Radiology
    "rad-dino": {
        "repo_id": "microsoft/rad-dino",
        "category": "2D Radiology",
        "gated": "false",
        "description": "DINOv2 Chest X-Ray encoder (MIT License)",
    },
    "medsiglip-448": {
        "repo_id": "google/medsiglip-448",
        "category": "2D Radiology",
        "gated": "true",
        "description": "Medical SigLIP 2D contrastive vision-text model (HAI-DEF Gated)",
    },
    # 3D Radiology & VLM
    "ct-fm": {
        "repo_id": "project-lighter/CT-FM",
        "category": "3D CT",
        "gated": "false",
        "description": "3D CT Volumetric Foundation Model",
    },
    "m3d-lamed": {
        "repo_id": "BAAI/M3D-LaMed-Phi-3-4B",
        "category": "3D CT VLM",
        "gated": "false",
        "description": "3D Medical VLM (BAAI-DCAI)",
    },
    "m3d-clip": {
        "repo_id": "BAAI/M3D-CLIP",
        "category": "3D CT VLM",
        "gated": "false",
        "description": "3D Medical Contrastive Image-Text Model",
    },
    # Pathology
    "h-optimus-0": {
        "repo_id": "bioptimus/H-optimus-0",
        "category": "Pathology",
        "gated": "true",
        "description": "1.1B Histology Tile Encoder (Bioptimus, Apache-2.0 Gated)",
    },
    "gigapath": {
        "repo_id": "prov-gigapath/prov-gigapath",
        "category": "Pathology",
        "gated": "true",
        "description": "Prov-GigaPath Whole-Slide Foundation Model (Gated)",
    },
    "titan": {
        "repo_id": "MahmoodLab/TITAN",
        "category": "Pathology",
        "gated": "true",
        "description": "Whole-Slide Multimodal Encoder (Mahmood Lab, Gated)",
    },
    "conch": {
        "repo_id": "MahmoodLab/CONCH",
        "category": "Pathology",
        "gated": "true",
        "description": "Histology Vision-Language Model (Mahmood Lab, Gated)",
    },
    "uni": {
        "repo_id": "MahmoodLab/UNI",
        "category": "Pathology",
        "gated": "true",
        "description": "General-Purpose Pathology ViT (Mahmood Lab, Gated)",
    },
    "virchow2": {
        "repo_id": "paige-ai/Virchow2",
        "category": "Pathology",
        "gated": "true",
        "description": "Paige AI Virchow 2 Pathology ViT (Gated)",
    },
    "phikon-v2": {
        "repo_id": "owkin/phikon-v2",
        "category": "Pathology",
        "gated": "false",
        "description": "Owkin Phikon-v2 Pathology Encoder",
    },
    # Generative VLMs & LMs
    "medgemma-1.5-4b-it": {
        "repo_id": "google/medgemma-1.5-4b-it",
        "category": "Generative VLM",
        "gated": "true",
        "description": "Medical Gemma 1.5 4B Multimodal Model (HAI-DEF Gated)",
    },
    "gemma-3-4b-it": {
        "repo_id": "google/gemma-3-4b-it",
        "category": "Language Baseline",
        "gated": "true",
        "description": "Google Gemma 3 4B Instruct Baseline (Gemma Terms)",
    },
    "qwen2.5-7b-instruct": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "category": "Language Baseline",
        "gated": "false",
        "description": "Qwen 2.5 7B Instruct Language Model (Apache-2.0)",
    },
    "biomedclip": {
        "repo_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "category": "Vision-Language",
        "gated": "false",
        "description": "Microsoft BiomedCLIP PubMedBERT ViT Model",
    },
}


def download_model(model_key: str, info: dict[str, str], cache_dir: Path | None, token: str | None) -> bool:
    repo_id = info["repo_id"]
    category = info["category"]
    is_gated = info["gated"] == "true"
    description = info["description"]

    print(f"\n---> [{category}] Downloading: {model_key} ({repo_id})")
    print(f"     Description: {description}")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.flax"],
        )
        print(f"[SUCCESS] Downloaded {model_key} to: {local_path}")
        return True
    except (GatedRepoError, HfHubHTTPError) as e:
        if is_gated:
            print(f"[GATED NOTICE] Access to {repo_id} requires accepted terms on Hugging Face.")
            print(f"               Visit: https://huggingface.co/{repo_id}")
            print("               Log in with 'huggingface-cli login' or pass --token / HF_TOKEN.")
        else:
            print(f"[WARNING] Could not download {repo_id}: {e}")
        return False
    except RepositoryNotFoundError:
        print(f"[WARNING] Repository {repo_id} not found or revision unavailable.")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error downloading {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face model weights for MedFM")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to Hugging Face cache directory (default: ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face User Access Token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CATALOG.keys()) + ["all", "open"],
        default=["all"],
        help="Specific model keys to download (default: all)",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" MedFM Hugging Face Foundation Models Downloader")
    print("======================================================================")

    if args.token:
        print("[INFO] Using provided Hugging Face authentication token.")
    else:
        print("[INFO] No HF_TOKEN provided. Downloading open models; gated models will require terms acceptance.")

    target_keys = list(MODEL_CATALOG.keys())
    if "open" in args.models:
        target_keys = [k for k, v in MODEL_CATALOG.items() if v["gated"] == "false"]
    elif "all" not in args.models:
        target_keys = [k for k in args.models if k in MODEL_CATALOG]

    results: dict[str, bool] = {}
    for key in target_keys:
        info = MODEL_CATALOG[key]
        success = download_model(key, info, args.cache_dir, args.token)
        results[key] = success

    print("\n======================================================================")
    print(" Download Summary Report")
    print("======================================================================")
    successful = [k for k, v in results.items() if v]
    failed_or_gated = [k for k, v in results.items() if not v]

    print(f"Successfully Downloaded ({len(successful)}):")
    for k in successful:
        print(f"  [OK] {k} ({MODEL_CATALOG[k]['repo_id']})")

    if failed_or_gated:
        print(f"\nSkipped / Gated Access Required ({len(failed_or_gated)}):")
        for k in failed_or_gated:
            gated_note = " (Gated)" if MODEL_CATALOG[k]["gated"] == "true" else ""
            print(f"  [PENDING] {k} ({MODEL_CATALOG[k]['repo_id']}){gated_note}")

    print("\nNote: For gated models marked PENDING, visit their HF pages, accept the license terms,")
    print("      run 'huggingface-cli login', and re-run this script.")


if __name__ == "__main__":
    main()
