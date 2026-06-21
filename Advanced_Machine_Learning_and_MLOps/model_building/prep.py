import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
REPO_ID = "shrikantpillay/Advanced_Machine_Learning_and_MLOps"
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

# Dynamically locate the local data folder relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
local_data_path = os.path.normpath(os.path.join(current_dir, "..", "data", "tourism.csv"))
HF_DATASET_PATH = f"hf://datasets/{REPO_ID}/tourism.csv"

# Load the dataset safely
if os.path.exists(local_data_path):
    print(f"Loading dataset locally from: {local_data_path}")
    df = pd.read_csv(local_data_path)
else:
    print(f"Local file not found. Loading dataset from Hugging Face: {HF_DATASET_PATH}")
    df = pd.read_csv(HF_DATASET_PATH)

print("Dataset loaded successfully.")

# Drop the unique identifier
if 'UDI' in df.columns:
    df.drop(columns=['UDI'], inplace=True)

# Encoding the categorical 'Type' column
if 'Type' in df.columns:
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])

target_col = 'Failure'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save split splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload split files back to the dataset repository root
print("Uploading processed splits to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  
        repo_id=REPO_ID,
        repo_type="dataset",
    )

print("Data preparation and upload complete!")
