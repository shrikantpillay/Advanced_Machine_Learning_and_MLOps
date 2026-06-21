from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "shrikantpillay/Advanced_Machine_Learning_and_MLOps"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")
# 1. Get the directory where data_register.py is actually located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Safely jump up one level to 'Advanced_Machine_Learning_and_MLOps' and into 'data'
data_folder_path = os.path.normpath(os.path.join(current_dir, "..", "data"))
api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
