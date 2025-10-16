#!/usr/bin/env python3
"""
Script test để kiểm tra xác thực Google API
"""

import os
import sys
from providers.ai_provider_vertex import AIProviderVertex

# Thử load từ file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv không có sẵn

def test_authentication():
    """Test xác thực với các phương pháp khác nhau"""
    
    print("🔍 Kiểm tra xác thực Google API...")
    print("=" * 50)
    
    # Kiểm tra API key từ environment
    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ Tìm thấy GOOGLE_API_KEY: {api_key[:10]}...{api_key[-4:]}")
        
        try:
            print("🧪 Test với GenerativeAI API key...")
            ai = AIProviderVertex(api_key=api_key)
            print("✅ Khởi tạo thành công với API key!")
            return True
        except Exception as e:
            print(f"❌ Lỗi với API key: {e}")
    else:
        print("⚠️  Không tìm thấy GOOGLE_API_KEY trong environment")
    
    # Test với service account
    print("\n🧪 Test với Service Account...")
    try:
        ai = AIProviderVertex()
        print("✅ Khởi tạo thành công với Service Account!")
        
        # Test một request đơn giản
        if hasattr(ai.service, 'use_genai') and not ai.service.use_genai:
            print("⚠️  Chỉ khởi tạo được, chưa test API call do lỗi 401")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Lỗi với Service Account: {e}")
        return False

def main():
    success = test_authentication()
    
    if success:
        print("\n🎉 Hệ thống sẵn sàng hoạt động!")
    else:
        print("\n💡 Gợi ý giải quyết:")
        print("1. Lấy API key từ: https://aistudio.google.com/")
        print("2. Thiết lập trong terminal:")
        print("   export GOOGLE_API_KEY='your-actual-api-key'")
        print("3. Hoặc tạo file .env:")
        print("   cp .env.example .env")
        print("   # Sau đó edit file .env và thêm API key")
        print("4. Hoặc xem hướng dẫn chi tiết trong AUTHENTICATION_FIX.md")
        
        sys.exit(1)

if __name__ == "__main__":
    main()