# 🛡️ STAN-Mamba for Financial Fraud Detection

Hệ thống phát hiện gian lận tài chính ứng dụng mô hình học sâu kết hợp giữa **Spatio-Temporal Attention Network (STAN)** và **Mamba (Selective State Space Model)** trên tập dữ liệu **S-FFSD (Synthetic Financial Fraud Dataset)**.

---

## 📖 Giới thiệu đề tài

Phát hiện giao dịch gian lận trong hệ thống tài chính là một bài toán thách thức do dữ liệu có tính chất:
- **Mất cân bằng nghiêm trọng:** Số lượng giao dịch gian lận chiếm tỷ lệ rất nhỏ so với giao dịch thông thường.
- **Tương quan Không - Thời gian phức tạp:** Hành vi gian lận thường thể hiện qua chuỗi giao dịch liên tiếp trong các khoảng thời gian ngắn hoặc phân tán qua nhiều vị trí bất thường.

Dự án này đề xuất kiến trúc lai **STAN + Mamba**:
1. **STAN (Spatio-Temporal Attention Network):** Học và trích xuất tương quan giữa các cửa sổ thời gian (Time Windows) và khoảng cách không gian (Spatio Windows).
2. **Mamba (State Space Model - SSM):** Xử lý chuỗi biểu diễn thời gian với khả năng nắm bắt phụ thuộc xa và độ phức tạp tuyến tính $O(N)$, tối ưu hóa tốc độ huấn luyện và suy luận so với Transformer truyền thống.

---

## 🏗️ Kiến trúc mô hình

Mô hình trong [`stan.py`](stan.py) được xây dựng theo luồng tuần tự:
```
Input Data (Transactions)
   │
   ▼
[ Feature Engineering ] ───> 3D Tensor: (Batch, Time_Windows, Spatio_Windows, Feat_Dim)
   │
   ▼
[ Spatio-Temporal Attention ] ───> Kết hợp ma trận Attention theo trục Thời gian & Không gian
   │
   ▼
[ Linear Projection & LayerNorm ] ───> Ánh xạ sang không gian ẩn (d_model = 128)
   │
   ▼
[ Mamba Residual Block ] ───> Xử lý chuỗi chọn lọc (Selective SSM)
   │
   ▼
[ Global Pooling (Max + Avg) ] ───> Tổng hợp biểu diễn đặc trưng toàn cục
   │
   ▼
[ Classification Head ] ───> Logits nhãn gian lận (0: Bình thường, 1: Gian lận)
```

---

## 📂 Cấu trúc thư mục

```text
├── data_engineering.py     # Hàm tính Entropy giao dịch và tạo ma trận đặc trưng 3D Không-Thời gian
├── stan.py                 # Định nghĩa lớp mô hình stan_model (STAN + Mamba)
├── mamba_minimal.py        # Cài đặt khối Mamba thuần túy trên PyTorch (MambaBlock, Selective SSM)
├── Train.py                # Vòng lặp huấn luyện, tối ưu hóa Loss và hàm tính chỉ số đánh giá
├── stan_mamba_main.py      # Entry point thực thi toàn bộ pipeline huấn luyện & kiểm thử
├── requirements.txt        # Danh sách thư viện cần thiết
├── .gitignore              # Danh sách các file/thư mục bỏ qua không đẩy lên Git
└── README.md               # Tài liệu hướng dẫn sử dụng dự án
```

---

## ⚙️ Cài đặt & Hướng dẫn sử dụng

### 1. Yêu cầu hệ thống
* Python >= 3.9
* PyTorch >= 2.0 (khuyến nghị có GPU CUDA nếu huấn luyện tập dữ liệu lớn)

### 2. Cài đặt môi trường
Khuyến nghị tạo môi trường ảo Python trước khi cài đặt:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 3. Chuẩn bị tập dữ liệu
Tải tập dữ liệu `S-FFSD.csv` và đặt vào thư mục `data/`:
```text
2351010049_HoVanDo_TH/
└── data/
    └── S-FFSD.csv
```
*(Lưu ý: Mã nguồn tự động ưu tiên nạp file tại `data/S-FFSD.csv`, nếu không tìm thấy sẽ thử đường dẫn fallback trên máy phát triển).*

### 4. Huấn luyện mô hình
Chạy lệnh sau để bắt đầu quá trình trích xuất đặc trưng và huấn luyện:
```bash
python stan_mamba_main.py
```

Các siêu tham số mặc định:
* `epochs`: 20
* `batch_size`: 256
* `lr`: 0.003
* `attention_hidden_dim`: 150
* `time_windows`: `[1, 3, 5, 10, 20, 50, 100, 500]`
* `spatio_windows`: `[1, 2, 3, 4, 5]`

---

## 📈 Đánh giá mô hình

Mô hình xuất ra các độ đo định lượng chuẩn cho bài toán phát hiện gian lận:
* **F1-Score:** Đánh giá độ cân bằng giữa Precision và Recall trên lớp nhãn gian lận.
* **ROC-AUC:** Diện tích dưới đường cong ROC đánh giá khả năng phân loại.
* **Average Precision (PR-AUC):** Độ đo then chốt cho dữ liệu có độ lệch nhãn cao.

---

## 👨‍💻 Tác giả
* **Họ và tên:** Hồ Văn Độ
* **MSSV:** 2351010049
* **Đề tài:** Nghiên cứu và ứng dụng Spatio-Temporal Attention kết hợp Mamba trong phát hiện gian lận tài chính.
