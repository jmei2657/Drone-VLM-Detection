# run once while online
from huggingface_hub import snapshot_download

snapshot_download(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    local_dir="./models/SmolVLM-500M-Instruct",
    local_dir_use_symlinks=False
)
