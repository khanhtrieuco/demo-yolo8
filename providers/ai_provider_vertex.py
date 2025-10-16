from providers.vertex_utils import GCVVertexService

class AIProviderVertex:
    def __init__(self, threshold=70.0):
        self.service = GCVVertexService(
            project_id="igneous-gamma-474704-r1",
            location="us-central1",
            model_name="gemini-2.5-flash",
            threshold=threshold
        )

    def analyze(self, ref_path: str, test_path: str, description: str = "") -> dict:
        result = self.service.analyze_display(ref_path, test_path, description)

        score = float(result.get("score") or 0)
        passed = bool(result.get("passed"))
        issues = result.get("issues", [])
        summary = "; ".join(issues) if issues else "Không phát hiện lỗi."
        status = "Đạt" if passed and score >= self.service.threshold else "Không đạt"

        return {
            "score": score,
            "passed": passed,
            "status": status,
            "issues": issues,
            "summary": summary
        }