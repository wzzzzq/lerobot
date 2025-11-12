from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir="/pfs/pfs-ilWc5D/ziqianwang/robotwin",
    repo_type="dataset",
    resume_download=True,
)
