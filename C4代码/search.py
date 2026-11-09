import pandas as pd
import random
import sqlite3
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SQLiteVectorStore:
    def __init__(self, db_path="./shoe_vector_store.db", init_if_missing=True):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
     
        db_exists = self._check_database_exists()
        
        if not db_exists and init_if_missing:
            print("数据库不存在，进行初始化...")
            self._init_db()
        elif db_exists:
            print(f"加载已存在的数据库: {db_path}")
        else:
            print("数据库不存在且未启用自动初始化")
    
    def _check_database_exists(self):
        """检查数据库文件是否存在且包含必要的表"""
        import os
        if not os.path.exists(self.db_path):
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            table_exists = cursor.fetchone() is not None
            conn.close()
            return table_exists
        except:
            return False
    
    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                brand TEXT,
                name TEXT,
                color TEXT,
                price REAL,
                categories TEXT,
                time TEXT,
                discount TEXT,
                discount_type TEXT,
                prices_merchant TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
       
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_brand ON documents(brand)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price ON documents(price)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_categories ON documents(categories)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_time ON documents(time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discount ON documents(discount)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_merchant ON documents(prices_merchant)')
        
        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {self.db_path}")
    
    def add_documents(self, texts, metadatas, clear_existing=False):
        """添加文档到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if clear_existing:
            cursor.execute("DELETE FROM documents")
            print("已清空现有数据")
     
        print("正在生成embedding向量...")
        embeddings = self.embedding_model.encode(texts)
        
        print(f"正在添加 {len(texts)} 个文档到数据库...")
        for i, text in enumerate(texts):
            # 序列化embedding向量
            embedding_blob = np.array(embeddings[i]).astype(np.float32).tobytes()
            metadata = metadatas[i]
            
            # 插入文档数据 - 所有字段都在同一个表中
            cursor.execute('''
                INSERT INTO documents 
                (content, embedding, brand, name, color, price, categories, time, discount, discount_type, prices_merchant)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                text,
                embedding_blob,
                metadata.get('brand', ''),
                metadata.get('name', ''),
                metadata.get('color', ''),
                metadata.get('price', 0),
                metadata.get('categories', ''),
                metadata.get('time', ''),
                metadata.get('discount', ''),
                metadata.get('discount_type', ''),
                metadata.get('prices_merchant', '')
            ))
        
        conn.commit()
        conn.close()
        print(f"成功添加 {len(texts)} 个文档到数据库")
    
    def similarity_search(self, query_text, k=5, filter_dict=None):
        """相似性搜索：支持元数据过滤"""
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_conditions = ["1=1"]  
        query_params = []
        
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, dict):
                    for op, op_value in value.items():
                        if op == "$eq":
                            where_conditions.append(f"{key} = ?")
                            query_params.append(op_value)
                        elif op == "$ne":
                            where_conditions.append(f"{key} != ?")
                            query_params.append(op_value)
                        elif op == "$gt":
                            where_conditions.append(f"{key} > ?")
                            query_params.append(op_value)
                        elif op == "$gte":
                            where_conditions.append(f"{key} >= ?")
                            query_params.append(op_value)
                        elif op == "$lt":
                            where_conditions.append(f"{key} < ?")
                            query_params.append(op_value)
                        elif op == "$lte":
                            where_conditions.append(f"{key} <= ?")
                            query_params.append(op_value)
                        elif op == "$in":
                            placeholders = ",".join(["?"] * len(op_value))
                            where_conditions.append(f"{key} IN ({placeholders})")
                            query_params.extend(op_value)
                else:
                    where_conditions.append(f"{key} = ?")
                    query_params.append(value)
        
        where_clause = " AND ".join(where_conditions)

        cursor.execute(f'''
            SELECT id, content, embedding, brand, name, color, price, categories, time, discount, discount_type, prices_merchant
            FROM documents 
            WHERE {where_clause}
        ''', query_params)
        
        candidates = cursor.fetchall()
        conn.close()
        
        if not candidates:
            print("未找到匹配的文档")
            return []

        similarities = []
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        for candidate in candidates:
            doc_id, content, embedding_blob, brand, name, color, price, categories, time, discount, discount_type, prices_merchant = candidate

            doc_vec = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)

            similarity = cosine_similarity(query_vec, doc_vec)[0][0]

            metadata = {
                "brand": brand,
                "name": name,
                "color": color,
                "price": price,
                "categories": categories,
                "time": time,
                "discount": discount,
                "discount_type": discount_type,
                "prices_merchant": prices_merchant
            }
            
            similarities.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "similarity": similarity
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        similar=[item['content'] for item in similarities[:k]]
        return similar
    
    def get_document_count(self):
        """获取文档总数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_database_info(self):
        """获取数据库信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='documents'")
        indexes = cursor.fetchall()
        
        conn.close()
        
        return {
            "tables": [table[0] for table in tables],
            "document_count": doc_count,
            "indexes": [index[0] for index in indexes]
        }

def prepare_initial_data():
    input_file_data = "/data1/wuquanda/agent_start/data/cleaned_shoe_data.csv"
    df = pd.read_csv(input_file_data)
    
    texts = []
    metadatas = []
    
    print("正在准备数据...")
    for _, row in df.iterrows():
        low_price = row.get('prices.amountMin', 0)
        high_price = row.get('prices.amountMax', 0)
        time = row.get('dateUpdated', '未知')

        try:
            dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            year_month = dt.strftime("%Y-%m")
        except:
            year_month = "未知"

        discount = '是' if random.random() < 0.4 else '否'
        discount_type = 'null'
        
        if discount == '是':
            discount_ = round((random.random() + 1) / 2, 1)
            discount_type = '满减' if random.random() < 0.5 else str(discount_)
        
        merchants = ['淘宝', '京东', '拼多多']
        merchant_index = random.randint(0, 2)
        
        if low_price != high_price:
            price = random.randint(round(low_price), round(high_price))
        else:
            price = low_price

        if discount_type == '满减':
            d_price = random.randint(round(0.1 * price), round(0.2 * price))
            s_price = round(0.9 * price)
            discount_type = f'满{s_price}减{d_price}'

        text = (
            f"品牌: {row['brand']}, 名称: {row['name']}, 颜色: {row.get('prices.color', '')}, 价格: {price}; "
            f"是否折扣: {discount}; 折扣类型: {discount_type}; 在售商店: {merchants[merchant_index]}; "
            f"时间: {year_month}; 类别: {row.get('categories', '')};"
        )
        
        texts.append(text)

        metadata = {
            "brand": str.lower(row["brand"]),
            "name": str.lower(row["name"]),
            "color": str.lower(row.get('prices.color', '')),
            'price': float(price),
            'categories': "Women's shoe",
            'time': year_month,
            'discount': discount,
            'discount_type': discount_type,
            'prices_merchant': merchants[merchant_index],
        }
        
        metadatas.append(metadata)
    
    return texts, metadatas

if __name__ == "__main__":
    def first_time_setup():
        texts, metadatas = prepare_initial_data()
        store = SQLiteVectorStore("./shoe_vector_store.db", init_if_missing=True)
        store.add_documents(texts, metadatas, clear_existing=False)

        result_csv_path = './shoe_metadata.csv'
        results_df = pd.DataFrame(metadatas)
        results_df.to_csv(result_csv_path, index=False, encoding="utf-8")
        print(f"元数据已保存到: {result_csv_path}")
        
        return store

    def normal_usage():
        print("=== 后续运行：直接加载数据库 ===")
        store = SQLiteVectorStore("./shoe_vector_store.db", init_if_missing=False)

        info = store.get_database_info()
        print(f"数据库信息: {info}")
        
        return store

    store_path = "./shoe_vector_store.db"
    import os
    if not os.path.exists(store_path):
        store = first_time_setup()
    else:
        store = normal_usage()

    print(f"数据库中共有 {store.get_document_count()} 个文档")

    print("\n=== 执行查询 ===")
    results = store.similarity_search(
        "运动鞋", 
        k=3,
        filter_dict={
            "brand": "nike",
            "price": {"$lte": 500},
            "discount": "是"
        }
    )
    
    for i, result in enumerate(results):
        print(f"内容: {result['content']}")