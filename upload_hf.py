import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="curia",
    repo_id="raidium/curia",
    repo_type="model",
)
