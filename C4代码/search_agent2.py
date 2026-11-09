import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from new_test_deepseek import DeepSeekChat
from transform_ import to_json,from_json
def extract_json_from_model_output(response):
    code_block_match = re.search(r"```(?:json)?\s*(.*?)```", response.content, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
    else:
        json_str = response.content.strip()
    try:
        response = json.loads(json_str)
    except json.JSONDecodeError:
        response = {}
    return response
class QueryConditionAgent:
    def __init__(self, llm, vectordb_list, embedding_model):
        """Initialize the agent with an LLM, ChromaDB instance, and embedding model."""
        self.llm = llm
        self.vectordb_list = vectordb_list
        self.embedding_model = embedding_model

    def extract_conditions(self, user_input):
        """Extract query text and conditions from user input using LLM and return them as a tuple."""
        prompt = f"""
请从以下用户输入中提取两部分内容：
1. 用于相似度查询的文本描述。
2. 查询条件，以JSON格式返回。可能的查询条件包括：品牌、价格、颜色、类别、时间、是否折扣、在售商店。如果某个条件未提及，请设为null。

示例：
用户输入：“我想买一双舒适的nike跑鞋，价格在100到200元之间，在淘宝上买”
输出：{{"query_text": "舒适的nike跑鞋", "conditions": {{"brand": "nike", "price": {{"min": 100, "max": 200}}, "color": null, "categories": null, "time": null, "discount": null, "prices_merchant": "淘宝"}}}}

用户输入：“最近想买双透气的通勤女鞋，希望是打折的。”
输出：{{"query_text": "透气的通勤女鞋", "conditions": {{"brand": null, "price": null, "color": null, "categories": "Women's shoe", "time": null, "discount": "是", "prices_merchant": null}}}}

用户输入：“我想买一双Muk luks的亮色女鞋，价格在20到200元之间，在淘宝或拼多多上买。”
输出：{{"query_text": "Muk luks的亮色女鞋", "conditions": {{"brand": "muk luks", "price": {{"min": 20, "max": 200}}, "color": null, "categories": "Women's shoe", "time": null, "discount": null, "prices_merchant": {{"$in": ["淘宝", "拼多多"]}}}}}}

注意，若用户输入包含亮色/暗色等模糊条件，对应属性的条件应视为未提及，即null。若输入的品牌名为中文，则翻译为相应的英文；提取到的时间应为%Y--%M格式的，若为其他格式，视作null.

现在，请分析以下输入：
{user_input}
"""
        response = self.llm.invoke(prompt)
        code_block_match = re.search(r"```(?:json)?\s*(.*?)```", response.content, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            json_str = response.content.strip()
        try:
            result = json.loads(json_str)
            query_text = result.get("query_text", "")
            conditions = result.get("conditions", {})
        except json.JSONDecodeError:
            query_text = ""
            conditions = {}
        return query_text, conditions

    def build_where_clause(self, conditions):
        """Build a ChromaDB where clause from extracted conditions."""
        where = {}
        whe = []
        brands = conditions.get("brand")
        colors = conditions.get("color")
        merchants = conditions.get("prices_merchant")
        if brands:
            try:
                brands = brands.get("$in")
                brands = [brand.lower() for brand in brands]
            except:
                brands = [brands.lower()]
            whe.append({"brand": {"$in": brands}})
        if conditions.get("price"):
            if "min" in conditions["price"] and conditions["price"]["min"] is not None:
                whe.append({"price": {"$gte": conditions["price"]["min"]}})
            if "max" in conditions["price"] and conditions["price"]["max"] is not None:
                whe.append({"price": {"$lte": conditions["price"]["max"]}})
        if colors:
            try:
                colors = colors.get("$in")
                colors = [color.lower() for color in colors]
            except:
                colors = [colors.lower()]
            whe.append({"color": {"$in": colors}})
        if conditions.get("categories"):
            catagory="Women's shoe"
            whe.append({"categories":catagory})
            #whe.append({"categories": conditions["categories"]})
        if conditions.get("time"):
            whe.append({"time": conditions["time"]})
        if conditions.get("discount"):
            whe.append({"discount": conditions["discount"]})
        if merchants:
            try:
                merchants = merchants.get("$in")
                merchants = [merchant.lower() for merchant in merchants]
            except:
                merchants = [merchants.lower()]
            whe.append({"prices_merchant": {"$in": merchants}})

        where = {"$and": whe} if whe else {}
        print(f"构建的查询条件：{where}")
        return where

    def query_database(self, query_text, where):
        """Query the database using similarity search with metadata filtering."""
        results = []
        for vectordb_path in self.vectordb_list:
            if vectordb_path is None:
                continue
            vectordb = Chroma(persist_directory=vectordb_path, embedding_function=self.embedding_model)
            # Perform similarity search with metadata filtering
            result = vectordb.similarity_search(query_text, k=2, filter=where)
            result=[res.page_content for res in result]
            results.append(result)
        return results

    def run(self, user_input):
        """Main method to process user input, generate conditions, and return query results."""
        query_text, conditions = self.extract_conditions(user_input)
        print(query_text)
        print(conditions)
        where = self.build_where_clause(conditions)
        results = self.query_database(query_text, where)
        return results

if __name__ == "__main__":
    print('starting...')
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb_list = [
        "/data1/wuquanda/agent_start/personal_assistant/tb_shoe_chroma_dbs",
        "/data1/wuquanda/agent_start/personal_assistant/jd_shoe_chroma_dbs",
        "/data1/wuquanda/agent_start/personal_assistant/pdd_shoe_chroma_dbs"
    ]
    llm = DeepSeekChat()

    query_agent = QueryConditionAgent(llm, vectordb_list, embedding_model)

    user_input = "我想买一双Muk luks的亮色女鞋，价格在20到200元之间，在淘宝或拼多多上买。"
    results = query_agent.run(user_input)
    print("Query Results:")
    print(f"Total results: {len(results)}")
    print(results)