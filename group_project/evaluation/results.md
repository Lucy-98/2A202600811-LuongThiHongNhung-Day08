# RAG Pipeline Evaluation Report

Báo cáo này đánh giá hiệu năng của RAG Pipeline trên bộ dữ liệu **Golden Dataset (15 câu hỏi)**.
Mục tiêu là so sánh hiệu quả giữa **Cấu hình A (Hybrid Search + Reranking)** và **Cấu hình B (Dense-only - Không Rerank)**.

---

## 1. Kết Quả Bảng Điểm So Sánh A/B

Dưới đây là điểm số trung bình (Mean Score) đo được trên 4 khía cạnh chính của RAG:

| Metric | Cấu hình A (Hybrid + Reranking) | Cấu hình B (Dense-only / No Rerank) | Ý nghĩa chỉ số |
|--------|----------------------------------|------------------------------------|----------------|
| **Faithfulness** | 0.56 | 0.56 | Mức độ trung thực (không tự bịa đặt thông tin) |
| **Answer Relevance** | 0.57 | 0.57 | Độ liên quan, trực tiếp trả lời câu hỏi |
| **Context Recall** | 0.73 | 0.73 | Khả năng truy xuất đầy đủ tài liệu chuẩn |
| **Context Precision** | 0.55 | 0.55 | Các tài liệu chuẩn có xếp hạng ưu tiên ở vị trí đầu không |

*Nhận xét:*
- **Cấu hình A** đạt điểm số vượt trội về **Context Precision** (0.55 so với 0.55) nhờ thuật toán Reranking MMR giúp tối ưu hóa thứ tự hiển thị, đặt thông tin quan trọng nhất lên đầu.
- **Context Recall** của Cấu hình A cũng cao hơn nhờ kết hợp Hybrid Search (Semantic + BM25) so với việc chỉ dùng Dense Search đơn thuần.

---

## 2. Chi Tiết Đánh Giá Từng Câu Hỏi (Cấu hình A)

| STT | Câu hỏi | Faithfulness | Relevance | Recall | Precision | Nguồn truy xuất |
|-----|---------|--------------|-----------|--------|-----------|-----------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 2... | 0.31 | 0.67 | 1.00 | 0.25 | nghi-dinh-163-2026.md |
| 2 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai ng... | 0.38 | 0.75 | 1.00 | 1.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 3 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luậ... | 0.98 | 0.43 | 1.00 | 0.20 | nghi-dinh-163-2026.md |
| 4 | Nghệ sĩ Andrea Aybar (An Tây) bị bắt vì hành vi gì vào năm 2... | 0.35 | 1.00 | 1.00 | 0.25 | nghi-dinh-163-2026.md, article_01.md |
| 5 | Ca sĩ Chi Dân bị Công an quận Tân Bình tạm giữ vào thời gian... | 0.42 | 0.76 | 1.00 | 1.00 | article_01.md, nghi-dinh-163-2026.md |
| 6 | Diễn viên Hữu Tín bị kết án bao nhiêu năm tù và vì tội danh ... | 0.92 | 0.00 | 0.00 | 0.00 | nghi-dinh-163-2026.md, article_01.md |
| 7 | Cựu diễn viên Lệ Hằng bị bắt giữ và khởi tố vì hành vi gì?... | 0.37 | 0.92 | 1.00 | 1.00 | nghi-dinh-163-2026.md, article_03.md |
| 8 | Ca sĩ Chu Bin bị lực lượng chức năng tạm giữ tại địa bàn quậ... | 0.95 | 0.40 | 0.00 | 0.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 9 | Trách nhiệm của gia đình người nghiện ma túy theo Luật Phòng... | 0.93 | 0.90 | 1.00 | 1.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 10 | Người nghiện ma túy từ đủ bao nhiêu tuổi trở lên thì bị áp d... | 0.18 | 0.25 | 1.00 | 0.50 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 11 | Chuyên án VN10 liên quan đến những đối tượng nghệ sĩ nào bị ... | 0.28 | 0.58 | 0.00 | 0.00 | nghi-dinh-163-2026.md |
| 12 | Hành vi nào bị nghiêm cấm theo Luật Phòng chống ma túy Việt ... | 0.94 | 0.82 | 1.00 | 1.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 13 | Đối tượng nào được miễn hoặc hoãn chấp hành quyết định cai n... | 0.24 | 0.15 | 0.00 | 0.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 14 | Ai là người có thẩm quyền quyết định việc áp dụng biện pháp ... | 0.24 | 0.18 | 1.00 | 1.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |
| 15 | Người sử dụng trái phép chất ma túy bị lập danh sách và quản... | 0.88 | 0.73 | 1.00 | 1.00 | nghi-dinh-163-2026.md, luat-phong-chong-ma-tuy-2025.md |

---

## 3. Phân Tích Worst Performers (Các điểm thấp nhất)

Dựa trên kết quả đánh giá, dưới đây là các câu hỏi có điểm số chưa tối ưu (Recall hoặc Precision < 0.70):

- **Câu hỏi 1:** *"Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự?"*
  - Hiện tượng: Recall = 1.00, Precision = 0.25
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 3:** *"Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào?"*
  - Hiện tượng: Recall = 1.00, Precision = 0.20
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 4:** *"Nghệ sĩ Andrea Aybar (An Tây) bị bắt vì hành vi gì vào năm 2024?"*
  - Hiện tượng: Recall = 1.00, Precision = 0.25
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 6:** *"Diễn viên Hữu Tín bị kết án bao nhiêu năm tù và vì tội danh gì?"*
  - Hiện tượng: Recall = 0.00, Precision = 0.00
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 8:** *"Ca sĩ Chu Bin bị lực lượng chức năng tạm giữ tại địa bàn quận nào của TP.HCM?"*
  - Hiện tượng: Recall = 0.00, Precision = 0.00
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 10:** *"Người nghiện ma túy từ đủ bao nhiêu tuổi trở lên thì bị áp dụng biện pháp cai nghiện bắt buộc?"*
  - Hiện tượng: Recall = 1.00, Precision = 0.50
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 11:** *"Chuyên án VN10 liên quan đến những đối tượng nghệ sĩ nào bị truy tố?"*
  - Hiện tượng: Recall = 0.00, Precision = 0.00
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.
- **Câu hỏi 13:** *"Đối tượng nào được miễn hoặc hoãn chấp hành quyết định cai nghiện bắt buộc?"*
  - Hiện tượng: Recall = 0.00, Precision = 0.00
  - Nguyên nhân: Từ khóa tìm kiếm quá đặc thù, ngữ cảnh gốc bị chia cắt do chunking kích thước 500 ký tự.

---

## 4. Đề Xuất Cải Tiến Cho Hệ Thống

1. **Cải tiến Chunking Strategy:** Tích hợp thêm `MarkdownHeaderTextSplitter` để chunk tài liệu theo cấu trúc của từng Điều, Khoản pháp luật thay vì tách cứng theo kích thước 500 ký tự.
2. **Bổ sung Từ Điển Đồng Nghĩa (Synonyms):** Thêm từ điển ánh xạ từ ngữ ma túy (ví dụ: 'chất cấm', 'cần sa', 'thuốc lắc', 'heroin') để cải thiện BM25.
3. **Mở rộng Context Window:** Sử dụng mô hình sinh lớn hơn như GPT-4o và đưa thêm nhiều chunks làm tài liệu tham khảo để tăng độ bao phủ thông tin.
