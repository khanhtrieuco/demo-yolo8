import psycopg2
from psycopg2.extras import RealDictCursor
import os

class DBProvider:
    
    def __init__(self):
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_name = os.getenv("DB_NAME", "shelves_db")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "Admin@123")
        self.db_port = os.getenv("DB_PORT", "5432")

    # Kết nối Postgres 
    def get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
            port=self.db_port
        )
    # Lấy ảnh mẫu theo code
    def get_reference_image_by_code(self, code: str):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM reference_images WHERE code = %s", (code,))
        ref = cur.fetchone()
        cur.close()
        conn.close()
        return ref
    
    # Lấy ảnh test theo id
    def get_test_image(self, test_id: int):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM test_images WHERE id = %s", (test_id,))
        test_img = cur.fetchone()
        if not test_img:
            cur.close()
            conn.close()
        return test_img
    
    # Lưu kết quả kiểm tra
    def insert_check_result(self, reference_id: int, test_path: str, similarity: float, result: str, issues: str):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO check_results (reference_id, test_path, similarity, result, issues)
            VALUES (%s, %s, %s, %s, %s)
        """, (reference_id, test_path, similarity, result, issues))
        conn.commit()
        cur.close()
        conn.close()
    
    # Thêm hoặc cập nhật ảnh mẫu
    def insert_or_update_reference(self, code: str, path: str, description: str):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reference_images (code, path, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (code)
            DO UPDATE SET path = EXCLUDED.path, description = EXCLUDED.description
            RETURNING id;
        """, (code, path, description))
        ref_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return ref_id
    
    def list_all_references(self):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM reference_images")
        refs = cur.fetchall()
        cur.close()
        conn.close()
        return refs
    
    def get_check_result(self, ref_id):
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM check_results WHERE reference_id = %s ORDER BY checked_at DESC",
            (ref_id,)
        )
        check_history = cur.fetchall()
        cur.close()
        conn.close()
        return check_history