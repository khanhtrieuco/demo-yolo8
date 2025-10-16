#!/usr/bin/env python3
"""
Script demo để test với API key thực tế
"""

def demo_with_api_key():
    """Demo sử dụng với API key"""
    
    # Bước 1: Nhập API key
    print("🔑 Demo test với Google API Key")
    print("=" * 40)
    
    api_key = input("Nhập GOOGLE_API_KEY của bạn (hoặc Enter để skip): ").strip()
    
    if not api_key:
        print("⚠️  Bỏ qua test với API key")
        return
    
    # Bước 2: Test với API key
    try:
        from providers.ai_provider_vertex import AIProviderVertex
        
        print("🧪 Đang test với API key...")
        ai = AIProviderVertex(api_key=api_key)
        print("✅ Khởi tạo thành công!")
        
        # Test một request đơn giản (nếu có ảnh)
        print("💡 Hệ thống sẵn sàng để xử lý ảnh!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Kiểm tra lại API key hoặc thử lại sau")
        return False

if __name__ == "__main__":
    demo_with_api_key()