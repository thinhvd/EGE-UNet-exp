# notebooks/

Nơi chứa mọi Jupyter/Colab notebook của project. Notebook mới (dù cho mục đích gì) đặt vào đây, không để ở thư mục gốc.

## Trạng thái hiện tại

| Notebook | Dùng cho | Trạng thái |
|---|---|---|
| `colab_train.ipynb` | Train trên Google Colab, lưu kết quả vào Google Drive | **Ngừng dùng từ 09/2026** — giữ lại để tham khảo |
| `colab_analysis.ipynb` | Chạy pipeline phân tích trên Colab | **Ngừng dùng từ 09/2026** — giữ lại để tham khảo |

Từ 09/2026 việc training chuyển sang **GPU server thuê**: xem `scripts/setup_server.sh` (dựng môi trường) và
`scripts/sync_results.sh` (kéo kết quả về máy local). Phân tích chạy local trên CPU như trước.

Hai notebook trên vẫn mô tả đúng cách chạy trên Colab nếu cần quay lại, nhưng không được cập nhật theo các
thay đổi mới của codebase (ví dụ các cờ CLI thêm sau 09/2026) — kiểm tra lại `train.py --help` trước khi dùng.
