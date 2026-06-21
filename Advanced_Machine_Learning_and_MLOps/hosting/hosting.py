from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/content/drive/MyDrive/Colab Notebooks/Advanced_Machine_Learning_and_MLOps/deployment",     # the local folder containing your files
    repo_id="shrikantpillay/Advanced_Machine_Learning_and_MLOps",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
