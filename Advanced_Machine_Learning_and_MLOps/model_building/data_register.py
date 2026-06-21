import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

repo_id = "shrikantpillay/Advanced_Machine_Learning_and_MLOps"
repo_type = "dataset"

# Initialize API client with token
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

# Step 1: Check if the repository/dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repository...")
    # Using api.create_repo ensuring your token auth carries over
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Dynamically locate the 'data' folder relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.normpath(os.path.join(current_dir, "..", "data"))

print(f"DEBUG: Attempting to upload folder from path: {data_folder_path}")

# Step 3: Upload files to Hugging Face
api.upload_folder(
    folder_path=data_folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Upload complete!")
