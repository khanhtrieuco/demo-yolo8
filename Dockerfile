FROM python:3.12-slim

# Thiết mục thư mục làm việc
WORKDIR /app

# Cài đặt các công cụ cơ bản và lib cần thiết cho psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ mã nguồn vào container
COPY . .

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=6000
EXPOSE 6000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]