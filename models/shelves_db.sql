-- Bảng ảnh chuẩn (A, B, C)
CREATE TABLE reference_images (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
	description TEXT
);

-- Bảng ảnh kiểm tra (A1, A2, B1...)
CREATE TABLE check_results (
    id SERIAL PRIMARY KEY,
    reference_id INTEGER REFERENCES reference_images(id) ON DELETE CASCADE,
    test_path VARCHAR(255) NOT NULL,
    similarity FLOAT,
    result VARCHAR(20),	
    issues TEXT,
    checked_at TIMESTAMP DEFAULT NOW()
);


SELECT * FROM reference_images;
SELECT * FROM check_results;