# Anomaly Detection - University Exam (Phát hiện Bất thường Điểm thi THPTQG) 

## Giới thiệu (Introduction)
Dự án này được thực hiện trong quá trình thực tập, nhằm vận dụng các kiến thức đã học về **Khoa học Dữ liệu (Data Science)** và **Học máy Không giám sát (Unsupervised Machine Learning)** vào một bài toán thực tế: Phát hiện dị biệt (Anomaly Detection). 

Mục tiêu của dự án là xây dựng một hệ thống phát hiện các hành vi gian lận thi cử (cá nhân và có tổ chức) dựa trên phân tích điểm thi THPT Quốc Gia, bằng cách chuyển đổi dữ liệu thô thành **Không gian Đặc trưng 4 chiều (Unified Feature Space)** và áp dụng thuật toán **Isolation Forest** kết hợp **K-Means Clustering**.

## Nguồn Dữ liệu (Dataset)
Dữ liệu điểm thi thô được thu thập từ Kaggle:
- **Link Dataset:** [Điểm thi THPTQG](https://www.kaggle.com/datasets/trongnguyen0408/im-thi-thptqg)

## Cấu trúc Thư mục (Project Structure)
Dưới đây là cấu trúc mã nguồn chính của dự án (đã bỏ qua các thư mục dữ liệu thô, file cache và kết quả điều tra tạm thời):

```text
Anomaly Detection University Exam 2026/
├── src/
│   ├── features/                    # Chứa các Module xử lý cốt lõi (Core Pipeline)
│   │   ├── data_cleaner.py          # Làm sạch dữ liệu, xử lý điểm liệt và chia khối thi
│   │   ├── feature_engineer.py      # Trích xuất 4 đặc trưng (F1-F4) để bắt "Sói cô độc" và "Gian lận tập thể"
│   │   ├── model.py                 # Huấn luyện mô hình Isolation Forest đa luồng
│   │   └── analysis.py              # Dùng K-Means gom cụm (Clustering) truy vết tâm chấn gian lận
│   │
│   └── notebooks/                   # Môi trường chạy thử nghiệm và trực quan hóa (EDA/Visualization)
│       ├── cleaning_&_EDA.ipynb     # Phân tích phân phối điểm số (Normal/Skewed distribution)
│       ├── feature_engineering.ipynb# Ép mảng dữ liệu thô có NaN thành Ma trận Dense
│       └── anomaly_detector.ipynb   # Trạm điều khiển trung tâm: Chạy Pipeline và xuất Blacklist
│
├── .gitignore                       # Bỏ qua các file rác, dữ liệu nhạy cảm (data, results...)
└── README.md                        # Tài liệu dự án
