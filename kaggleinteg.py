import os
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()
from kaggle.api.kaggle_api_extended import KaggleApi

    # Set your Kaggle API credentials (ensure kaggle.json is in ~/.kaggle or set env variables)
    # os.environ['KAGGLE_USERNAME'] = 'your_username'
    # os.environ['KAGGLE_KEY'] = 'your_key'

def download_spam_dataset(download_path='spam_data'):

    
    api = KaggleApi()
    api.authenticate()
    # Download latest version
    api.dataset_download_files("purusinghvi/email-spam-classification-dataset", path=download_path, unzip=True)
    print(f"Dataset downloaded and extracted to {download_path}")

if __name__ == "__main__":
    download_spam_dataset()