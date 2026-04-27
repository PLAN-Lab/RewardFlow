# import os
from huggingface_hub import HfApi, login

# repo_id = "onkarsus13/rewardflowflux"  # change as needed
# local_dir = "/data/onkar/models/FluxRewardFlowSnechell"  # the folder you want to upload
# repo_type = "model"  # "model", "dataset", or "space"

# # # Optional: authenticate (or set HF_TOKEN env var)
# # login(token=os.getenv("HF_TOKEN"))

# api = HfApi()
# api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

# api.upload_folder(
#     folder_path=local_dir,
#     repo_id=repo_id,
#     repo_type=repo_type,
#     path_in_repo=".",  # change to "subdir" if you want it under a folder in the repo
# )

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo.")
    parser.add_argument(
        "--repo-id",
        default="onkarsus13/RewardFlow",
        help="Hugging Face model repo id (default: onkarsus13/RewardFlow).",
    )
    parser.add_argument(
        "--local-dir",
        default="/data/onkar/models/RewardFlow",
        help="Target directory to download into.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF token (optional for public repos; required for private repos).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/tag/commit to download.",
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        token=args.token,
        revision=args.revision,
    )

    print(f"Download complete: {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()
