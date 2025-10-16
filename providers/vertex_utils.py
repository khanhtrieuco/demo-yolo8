import os
import re
import mimetypes
import requests
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel, Part

# === Cấu hình mặc định ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcv_key.json"


class GCVVertexService:
    def __init__(self, project_id: str, location: str, model_name: str, threshold: float = 70.0):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.threshold = threshold
        init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)

    # --- Helper ---
    def _load_image_part(self, path_or_url: str):
        """Load ảnh từ file hoặc URL thành Part object cho VertexAI."""
        if path_or_url.startswith(("http://", "https://")):
            try:
                response = requests.get(path_or_url, timeout=30)
                response.raise_for_status()
                data = response.content
                mime = response.headers.get("Content-Type", "image/jpeg")
            except Exception as e:
                raise ValueError(f"Lỗi khi tải ảnh từ URL {path_or_url}: {e}")
        else:
            mime = mimetypes.guess_type(path_or_url)[0] or "image/jpeg"
            with open(path_or_url, "rb") as f:
                data = f.read()

        if not data:
            raise ValueError(f"Dữ liệu ảnh rỗng: {path_or_url}")

        return Part.from_data(data=data, mime_type=mime)

    def _parse_score_and_result(self, full_text: str):
        """Phân tích text do Gemini trả về, tìm score (float) và trạng thái pass/fail."""
        if not full_text:
            return None, None

        text = full_text.lower().strip()
        m = re.search(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%?', text)
        score = float(m.group(1)) if m else None

        # Ưu tiên 'không đạt' trước
        if re.search(r'\b(không đạt|khong dat|fail)\b', text):
            passed = False
        elif re.search(r'\b(đạt|pass)\b', text):
            passed = True
        else:
            passed = None

        if passed is None and score is not None:
            passed = (score >= self.threshold)

        return score, passed

    def _extract_issues(self, full_text: str):
        """Lấy danh sách 'vấn đề' từ phản hồi."""
        issues = []
        m = re.search(r'Vấn đề\s*:([\s\S]+?)(?:\n\(END\)|$)', full_text, flags=re.IGNORECASE)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip(" -•\t")
                if line:
                    issues.append(line)
        return issues

    # --- Core method ---
    def analyze_display(self, original_img_path: str, test_img_path: str, description: str = None) -> dict:
        """Phân tích & so sánh hai ảnh trưng bày."""
        img_orig = self._load_image_part(original_img_path)
        img_test = self._load_image_part(test_img_path)

        desc_text = f'Mô tả ảnh gốc: "{description}"\n' if description else ""
        prompt = f"""
Bạn là chuyên gia kiểm tra kệ trưng bày sản phẩm siêu thị.
Ảnh A là mẫu chuẩn, ảnh B là ảnh cần kiểm tra.

{desc_text}
Nhiệm vụ:
1️⃣ Tính độ giống nhau (0–100%)
2️⃣ Kết luận "Đạt" nếu >= {self.threshold}%, ngược lại "Không đạt"
3️⃣ Liệt kê lỗi dưới phần "Vấn đề:"
4️⃣ Format đúng:

Kết quả: Đạt|Không đạt
Độ giống nhau: <float>
Vấn đề:
- Thiếu: ...
- Thừa: ...
- Cải thiện: ...
(END)
"""

        response = self.model.generate_content(
            [
                Part.from_text(prompt),
                Part.from_text("Ảnh A (original):"),
                img_orig,
                Part.from_text("Ảnh B (test):"),
                img_test,
            ],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 4096,
                "candidate_count": 1,
            },
        )

        # Ghép text phản hồi
        full_text = ""
        if hasattr(response, "candidates") and response.candidates:
            for c in response.candidates:
                for p in getattr(c.content, "parts", []):
                    if hasattr(p, "text") and p.text:
                        full_text += p.text + "\n"
        elif hasattr(response, "text"):
            full_text = response.text

        full_text = full_text.strip()

        if not full_text:
            return {"score": None, "passed": None, "issues": [], "raw": ""}

        score, passed = self._parse_score_and_result(full_text)
        issues = self._extract_issues(full_text)

        return {
            "score": score,
            "passed": passed,
            "issues": issues,
            "raw": full_text
        }
