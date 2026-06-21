# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize API and authenticate
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

# 1. EXACT REPO DETAILS FROM YOUR HUGGING FACE SPACE
REPO_ID = "shrikantpillay/Advanced_Machine_Learning_and_MLOps"

# 2. CORRECTED LOCAL PATH FROM YOUR GITHUB SCREENSHOT
DATASET_PATH = "Advanced_Machine_Learning_and_MLOps/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['UDI'], inplace=True)

# Encoding the categorical 'Type' column
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

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# 3. UPLOAD DIRECTLY TO YOUR HUGGING FACE SPACE ROOT
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # uploads directly to the root of your Space
        repo_id=REPO_ID,
        repo_type="space",                      # correctly targets your Space
    )
print("All files uploaded to Hugging Face Space successfully!")
