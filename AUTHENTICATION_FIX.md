# Giải quyết lỗi Google API Authentication

## Vấn đề
Bạn đang gặp lỗi: `google.api_core.exceptions.Unauthenticated: 401 Request had invalid authentication credentials`

## Nguyên nhân có thể
1. Service account key không có quyền truy cập Vertex AI
2. Các API chưa được enable trên Google Cloud Project
3. Service account bị disable hoặc hết hạn

## Giải pháp

### Giải pháp 1: Sử dụng Google GenerativeAI API Key (Khuyến nghị)

1. Truy cập [Google AI Studio](https://aistudio.google.com/)
2. Tạo API key mới
3. Thiết lập biến môi trường:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Hoặc tạo file `.env`:
```
GOOGLE_API_KEY=your-api-key-here
```

### Giải pháp 2: Sửa Service Account (Nâng cao)

1. Truy cập [Google Cloud Console](https://console.cloud.google.com/)
2. Chọn project: `igneous-gamma-474704-r1`
3. Enable các APIs:
   - Vertex AI API
   - Generative Language API
4. Kiểm tra service account `vision-demo@igneous-gamma-474704-r1.iam.gserviceaccount.com`:
   - Có quyền: `Vertex AI User` hoặc `AI Platform User`
   - Chưa bị disable
5. Tạo lại key nếu cần

### Test hệ thống

Chạy script test:
```python
from providers.ai_provider_vertex import AIProviderVertex

# Test với API key
ai = AIProviderVertex(api_key="your-api-key")
print("System ready!")
```

## Cấu trúc hiện tại

- Hệ thống đã được cập nhật để hỗ trợ cả 2 phương pháp xác thực
- Ưu tiên Google GenerativeAI API key (đơn giản hơn)
- Fallback về Vertex AI service account nếu không có API key