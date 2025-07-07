from huggingface_hub import hf_hub_download
mapper_path = hf_hub_download(
     repo_id   = "rmokady/clipcap",
     filename  = "coco_prefix-009.pt",   # ← the mapper file
     cache_dir = "./models",             # any folder you like
)
