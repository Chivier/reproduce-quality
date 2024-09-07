import os
import tarfile
import requests

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download file from {url}")

def extract_tgz(file_path, extract_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# 下载和解压数据集
urls = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
}

for name, url in urls.items():
    dest_path = f"{name}.tgz"
    extract_path = "./datasets/qasper"
    
    # 下载文件
    print(f"Downloading {url}...")
    download_file(url, dest_path)
    
    # 创建解压目录
    os.makedirs(extract_path, exist_ok=True)
    
    # 解压文件
    print(f"Extracting {dest_path} to {extract_path}...")
    extract_tgz(dest_path, extract_path)

print("Download and extraction completed.")
