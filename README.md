 # Hệ thống phân loại cảnh quan

Tài liệu ngắn giúp kiểm tra và chạy demo của đồ án.

**Yêu cầu**
- Python 3.8 - 3.11
- Cài đặt phụ thuộc:

```bash
pip install -r requirements.txt
```

**Các tệp chính**
- `App demo/streamlit_app.py`: ứng dụng Streamlit trình diễn dự đoán hai mô hình
- `App demo/Model/`: chứa mô hình huấn luyện (`scene_mobilenetv2.h5`, `scene_efficientnetb0.h5`)
- `images/`: ảnh kết quả (ma trận nhầm lẫn, ảnh minh họa, pipeline)
- `evaluate_models.py`, `create_examples_image.py`: script đánh giá và tạo ảnh minh họa
- `Training/`: notebooks huấn luyện và mã liên quan

**Chạy nhanh (local)**
1. Đảm bảo hai file mô hình có trong thư mục `App demo/Model/`. Nếu không, làm theo phần "Tải mô hình" phía dưới.
2. Từ thư mục gốc dự án chạy:

```bash
streamlit run "App demo/streamlit_app.py"
```

Giao diện sẽ cho phép upload ảnh và hiển thị dự đoán cùng độ tin cậy từ hai mô hình.

```bash
mkdir -p "App demo/Model"
curl -L -o "App demo/Model/scene_mobilenetv2.h5" "<MODEL_URL_1>"
curl -L -o "App demo/Model/scene_efficientnetb0.h5" "<MODEL_URL_2>"
```

Hoặc trên Windows PowerShell:

```powershell
mkdir "App demo/Model"
Invoke-WebRequest -Uri "<MODEL_URL_1>" -OutFile "App demo/Model/scene_mobilenetv2.h5"
Invoke-WebRequest -Uri "<MODEL_URL_2>" -OutFile "App demo/Model/scene_efficientnetb0.h5"
```

**Lưu ý**
- Ứng dụng Streamlit tự động tìm `Model/` và `seg/` bằng cách dò lên các thư mục cha, nên có thể chạy từ `App demo/`.
- Nếu gặp lỗi đường dẫn, chạy từ thư mục gốc hoặc kiểm tra rằng các file `.h5` nằm đúng chỗ.


**Hỗ trợ / Liên hệ**
- Nếu cần trợ giúp chạy demo, gửi lỗi kèm thông tin OS và phiên bản Python.

**Dataset**

 Vì Dataset lớn nên nhóm đã đưa Dataset lên Drive
 
 Link drive: https://drive.google.com/drive/folders/1f0wO7qSVitYsnQdC3-UH4HMt8Xbe1lxY?usp=sharing
