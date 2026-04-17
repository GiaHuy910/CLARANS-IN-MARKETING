# Customers_clustering_with_CLARANS

**Customers_clustering_with_CLARANS** là một **dự án phân cụm khách hàng** sử dụng thuật toán phân cụm CLARANS và các thuật toán khác để phân tích dữ liệu và khám phá ra các nhóm khách hàng tiềm năng, hành vi tiêu dùng để hỗ trợ doanh nghiệp quyết định các chiến lược kinh doanh hiệu quả.

---

## Tính năng chính

- **Phân cụm dữ liệu bằng nhiều thuật toán**
  - CLARANS.
  - K-Means.
  - Pam.
  - DBSCAN.

- **So sánh hiệu quả phân cụm**
  - Silhouette score.
  - Davies Bouldin score.
  - Inertia.

- **Trực quan hoá**
  - Hiển thị kết quả bằng biểu đồ và so sánh ra trang web local.

---

## Kiến trúc & Công nghệ

### Tech Stack

| Category         | Technology                    |
| ---------------- | ----------------------------- |
| Language         | Python                        |
| Data Processing  | Pandas, NumPy                 |
| Machine Learning | Scikit-learn                  |
| Clustering       | CLARANS, K-Means, DBSCAN, PAM |
| Visualization    | Matplotlib, Seaborn           |
| Environment      | Jupyter Notebook              |
| Frontend         | HTML, CSS                     |

---

## Cài đặt & Chạy project

### 1) Clone repo

```bash
git clone https://github.com/GiaHuy910/CLARANS-IN-MARKETING.git
cd Customers_clustering_with_CLARANS
```

### 2) Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 3) Build & Run

- Lúc này có thể chạy các file Jupyter Notebook. có thể sử dụng feature khác bằng việc chỉnh sửa "feature_to_use" trong file

- Chạy web :

  ## 1. Train models (saves actual runtimes)

  python retrain_models.py

  ## 2. Compare algorithms (loads runtimes, normalizes data, computes metrics)

  python compare_algorithms.py

  ## 3. Generate academic report

  python generate_report.py

  ## 4. Open Output/clustering_comparison_report.html in browser

---

## Cấu trúc dự án

```
Customers_clustering_with_CLARANS/
├─ customer_clustering/
│  ├─ compare/                      # các file chạy web
│  │  ├─ Algorithm.py
│  │  ├─ compare_algorithms.py
│  │  ├─ generate_report.py
│  │  ├─ retrain_models.py
│  │  └─ style.css                  # style web
│  ├─ datasets/                     # dataset
│  │  └─ marketing_campaign.csv
│  ├─ Output/                       # lưu mô hình model
│  │  ├─ calrans.mdl
│  │  ├─ kMeans.mdl
│  │  ├─ DBSCAN.mdl
│  │  └─ PAM.mdl
│  ├─ Clarans.ipynb                 # chạy và phân tích bằng CLARANS
│  ├─ data_processing.py            # Xử lý dữ liệu cho các file khác sử dụng
│  ├─ Dbscan.ipynb                  # chạy và phân tích bằng DBSCAN
│  ├─ K-means.ipynb                 # chạy và phân tích bằng K-Means
│  ├─ Pam.ipynb                     # chạy và phân tích bằng Pam
│  └─ Tiền_xử_lý_dữ_liệu.ipynb      # Notebook phân tích tiền xử lý dữ liệu
├─ README.md
└─ requirements.txt                 # các thư viện cần tải
```

---

## Data Flow

- Load dữ liệu khách hàng
- Tiền xử lý (missing values, scaling)
- Chọn feature
- Áp dụng các thuật toán phân cụm
- Đánh giá và so sánh kết quả
- Visualization bằng biểu đồ và hiển thị lên web

---

## Tác giả

Dự án được thực hiện bởi các thành viên nhóm:

| Tên thành viên    | Mã số sinh viên | GitHub                                                   |
| :---------------- | :-------------- | :------------------------------------------------------- |
| Bùi Hoàng Hải     | 31241021840     | [Link GitHub](https://github.com/HHai2006)               |
| Đỗ Trọng Hiếu     | 31221022015     | [Link GitHub](https://github.com/Hiu11)                  |
| Lê Vũ Hoàng       | 31241023055     | [Link GitHub](https://github.com/hoangle31241023055-cmd) |
| Phạm Thị Kim Hồng | 31221025429     | [Link GitHub](https://github.com/kimhongpham)            |
| Bùi Gia Huy       | 31241023914     | [Link GitHub](https://github.com/GiaHuy910)              |
