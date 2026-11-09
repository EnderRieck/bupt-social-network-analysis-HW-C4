import json
import re
from new_test_deepseek import DeepSeekChat
from openai import OpenAI  # 或使用其他LLM接口
from search import SQLiteVectorStore

class Search_agent(SQLiteVectorStore):
    def __init__(self, db_path="./shoe_vector_store.db", init_if_missing=True, llm=None):
        super().__init__(db_path, init_if_missing)
        self.llm=llm
        self.supported_fields = {
            "brand": "品牌",
            "name": "商品名称", 
            "color": "颜色",
            "price": "价格",
            "categories": "类别",
            "time": "时间",
            "discount": "是否有折扣",
            "discount_type": "折扣类型",
            "prices_merchant": "商家"
        }
        
        self.supported_operators = {
            "$eq": "等于",
            "$ne": "不等于",
            "$gt": "大于",
            "$gte": "大于等于", 
            "$lt": "小于",
            "$lte": "小于等于",
            "$in": "在列表中"
        }
    def natural_language_to_filter(self, query_text):
        """将自然语言转换为过滤字典"""
        prompt = self._build_prompt(query_text)
        
        response = self.llm.invoke(prompt)
        filter_str = response.content
        print(filter_str)
        # 解析JSON响应
        try:
            filter_dict = json.loads(filter_str)
            filter_dict['categories']="Women's shoe"
            return self._validate_filter(filter_dict)
        except json.JSONDecodeError:
            print("LLM返回的不是有效JSON，使用规则解析")
            return self._rule_based_parsing(query_text)
    
    def _build_prompt(self, query_text):
        """构建提示词"""
        return f"""
            请将以下自然语言查询转换为JSON格式的过滤条件。只返回JSON，不要其他内容。

            支持的字段：
            {json.dumps(self.supported_fields, ensure_ascii=False, indent=2)}

            支持的操作符：
            {json.dumps(self.supported_operators, ensure_ascii=False, indent=2)}

            转换规则：
            1. 品牌、颜色等文本字段使用 $eq 或 $in
            2. 价格等数值字段可以使用比较操作符
            3. 时间格式为 "YYYY-MM"
            4. 折扣字段值为 "是" 或 "否"
            5. prices_merchant字段值有 "淘宝"、"天猫"、"京东"
            示例：
            用户输入："找找耐克品牌的价格低于500元并且有折扣的运动鞋"
            输出：{{"brand": "nike", "price": {{"$lt": 500}}, "discount": "是", "categories": "Women's shoe"}}

            用户输入："阿迪达斯或彪马的红色鞋子"
            输出：{{"brand": {{"$in": ["adidas", "puma"]}}, "color": "red"}}

            用户输入："2023年以后的耐克运动鞋"
            输出：{{"brand": "nike", "time": {{"$gt": "2023-01"}}, "categories": "Women's shoe"}}

            现在请转换：
            用户输入："{query_text}"
            输出：
            """
    
    def _validate_filter(self, filter_dict):
        """验证过滤条件"""
        validated = {}
        
        for key, value in filter_dict.items():
            if key not in self.supported_fields:
                print(f"警告: 不支持的字段 '{key}'，已跳过")
                continue
            
            # 验证操作符
            if isinstance(value, dict):
                validated_value = {}
                for op, op_value in value.items():
                    if op in self.supported_operators:
                        validated_value[op] = op_value
                    else:
                        print(f"警告: 不支持的操作符 '{op}'，已跳过")
                if validated_value:
                    validated[key] = validated_value
            else:
                # 直接值，转换为等于操作
                validated[key] = value
        
        return validated    
    def run(self, query_text, k=3):
       
        filter_dict = self.natural_language_to_filter(query_text)
        print(f"生成的过滤条件: {filter_dict}")
        

        return self.similarity_search(query_text, k=k, filter_dict=filter_dict)
    

# 使用示例
if __name__ == "__main__":
    llm=DeepSeekChat()
    store = Search_agent("./shoe_vector_store.db", llm=llm)  
    
    test_queries = [
        "找找耐克品牌的价格低于500元并且有折扣的运动鞋",
        "阿迪达斯或彪马的红色鞋子", 
        "2023年以后的耐克运动鞋",
        "价格在300到600之间的黑色鞋子",
        "淘宝上的打折女鞋"
    ]
    
    for query in test_queries:
        print(f"\n=== 查询: '{query}' ===")
        results = store.run(query, k=2)
        
        for i, result in enumerate(results):
            print(result)