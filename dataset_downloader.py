import kagglehub

# Chạy file này nếu cần tải dataset
path = kagglehub.dataset_download("ravalsmit/customer-segmentation-data")

print("Path to dataset files:", path)