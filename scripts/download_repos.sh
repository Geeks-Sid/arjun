#!/usr/bin/env bash
# ==============================================================================
# MedFM Upstream Repositories Downloader
# Clones or updates all primary foundation model repositories into external_repos/
#
# Verified 2026-08-06: every URL below resolves to a live, public repository.
#   - Most are GitHub code repos (cloned directly).
#   - Models not hosted on GitHub (RAD-DINO, BiomedCLIP, Virchow) now point to
#     their canonical Hugging Face model repos, which are git-cloneable.
#   - Virchow is gated on HF: you must register and accept the license before a
#     plain git clone succeeds, so it will [WARN] until you authenticate
#     (e.g. HF_TOKEN set, or git credentials with accepted terms).
#   - "v3ntus/STIP" was removed: that path belongs to an unrelated personal
#     account and no medical/CT foundation model named STIP exists.
# ==============================================================================

set -euo pipefail

TARGET_DIR="${1:-external_repos}"
mkdir -p "$TARGET_DIR"

# --- Colored output (auto-disabled when not a TTY or when NO_COLOR is set) ---
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    c_bold=$'\033[1m'
    c_red=$'\033[31m'
    c_green=$'\033[32m'
    c_yellow=$'\033[33m'
    c_blue=$'\033[34m'
    c_cyan=$'\033[36m'
    c_magenta=$'\033[35m'
    c_reset=$'\033[0m'
else
    c_bold=''; c_red=''; c_green=''; c_yellow=''; c_blue=''; c_cyan=''; c_magenta=''; c_reset=''
fi

# --- Message helpers ---
ok()   { printf '%s\n' "${c_green}[OK]${c_reset}     $1"; }      # success
warn() { printf '%s\n' "${c_yellow}[WARN]${c_reset}   $1"; }     # failure, continuing
note() { printf '%s\n' "${c_bold}[$2]${c_reset} $1"; }           # progress line
err()  { printf '%s\n' "${c_red}[RESULT]${c_reset} $1"; }        # final summary (failed)
sec()  { printf '%s\n' "" "${c_cyan}${c_bold}--- $1 ---${c_reset}"; }

FAILED=0

clone_or_update() {
    local repo_url="$1"
    local extra_note="${2:-}"
    local repo_name dest_path
    repo_name="$(basename "$repo_url" .git)"
    dest_path="$TARGET_DIR/$repo_name"

    if [ -n "$extra_note" ]; then
        printf '%s\n' "${c_cyan}  note: ${extra_note}${c_reset}"
    fi

    if [ -d "$dest_path/.git" ]; then
        note "$repo_name exists. Pulling latest changes..." "UPDATE"
        if (cd "$dest_path" && git pull --ff-only) >/dev/null 2>&1; then
            ok "$repo_name is up to date."
        else
            FAILED=$((FAILED + 1))
            warn "Failed to pull $repo_name ($repo_url). Continuing..."
        fi
    else
        note "Cloning $repo_url -> $dest_path..." "CLONE"
        if git clone --depth 1 "$repo_url" "$dest_path" >/dev/null 2>&1; then
            ok "Cloned $repo_name."
        else
            FAILED=$((FAILED + 1))
            warn "Failed to clone $repo_url — unreachable or access gated. Continuing..."
        fi
    fi
}

printf '%s\n' \
    "======================================================================" \
    " ${c_magenta}${c_bold}MedFM Upstream Repository Downloader${c_reset}" \
    " Downloading / Updating upstream foundation model repositories into: $TARGET_DIR" \
    "======================================================================"

sec "1. 2D Radiology & Medical VLM Repositories"
clone_or_update "https://github.com/Google-Health/medsiglip.git"
clone_or_update "https://huggingface.co/microsoft/rad-dino.git" "RAD-DINO: official home is HF (no GitHub repo)"
clone_or_update "https://github.com/Google-Health/medgemma.git"

sec "2. 3D Radiology (CT & MRI) Volumetric Repositories"
clone_or_update "https://github.com/project-lighter/CT-FM.git"
clone_or_update "https://github.com/ricklisz/FlexiCT.git"
clone_or_update "https://github.com/StanfordMIMI/Merlin.git"
clone_or_update "https://github.com/wangshansong1/Triad.git"
clone_or_update "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git"
clone_or_update "https://github.com/bowang-lab/MedSAM2.git"
clone_or_update "https://github.com/BAAI-DCAI/M3D.git"

sec "3. Pathology & Whole-Slide Image Repositories"
clone_or_update "https://github.com/prov-gigapath/prov-gigapath.git"
clone_or_update "https://github.com/mahmoodlab/TITAN.git"
clone_or_update "https://github.com/mahmoodlab/CONCH.git"
clone_or_update "https://github.com/mahmoodlab/TRIDENT.git"
clone_or_update "https://github.com/mahmoodlab/UNI.git"
clone_or_update "https://huggingface.co/paige-ai/Virchow.git" "Virchow: gated on HF — register + accept license before clone succeeds"
clone_or_update "https://github.com/owkin/HistoSSLscaling.git" "Phikon: official release lives here (no owkin/phikon repo)"

sec "4. Extended Literature & CT Foundation Repositories"
clone_or_update "https://github.com/wasserth/TotalSegmentator.git"
clone_or_update "https://github.com/JunMa11/AbdomenCT-1K.git"
# v3ntus/STIP removed 2026-08-06: v3ntus is an unrelated personal GH account and
# no medical/CT foundation model named "STIP" exists. Re-add the correct URL here
# if this entry was meant to point at a specific project.
clone_or_update "https://github.com/BMEII-AI/RadImageNet.git" "RadImageNet: official repo moved to BMEII-AI org"
clone_or_update "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.git" "BiomedCLIP: official home is HF (this is the exact model id)"
clone_or_update "https://github.com/rajpurkarlab/CheXzero.git" "CheXzero: moved to rajpurkarlab org (ex-stanfordmlgroup)"
clone_or_update "https://github.com/microsoft/LLaVA-Med.git"

printf '%s\n' ""
echo "======================================================================"
if [ "$FAILED" -gt 0 ]; then
    err "${FAILED} repository(s) failed to clone/update — see [WARN] lines above."
else
    ok "All repositories cloned/updated successfully."
fi
echo " Repositories saved in: $TARGET_DIR"
echo "======================================================================"
