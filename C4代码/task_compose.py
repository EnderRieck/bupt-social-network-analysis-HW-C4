import json, re,os
from datetime import datetime, timedelta
from new_test_deepseek import DeepSeekChat

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
class Decision_Agent:
    def __init__(self, llm, memory_dir="./user_memory"):
        self.llm = llm
        self.memory_dir = memory_dir
    
    def _load_purchase(self,path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
        
    def sanitize_purchase_records(self,user_json):
        """ 从用户信息中抽出购买记录 """
        recs = user_json.get("purchase_records", [])
        out = []
        for r in recs:
            out.append({
                "time": r.get("time"),
                "item": r.get("item"),
                "price": r.get("price"),
            })
        return out

    def recent_purchases(self,purchase_records, days=90):
        cutoff = datetime.utcnow() - timedelta(days=days)
        out = []
        for r in purchase_records:
            try:
                ts = datetime.strptime(r.get("time"), "%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = None
            if ts and ts >= cutoff:
                out.append(r)
        return out

    def build_prompt_with_purchases(self,user_input, purchases_summary):
        purch_text = "\n".join(purchases_summary) if purchases_summary else "用户近期无购买记录。"
        prompt = f"""
    你是一个任务分解助手。请结合用户近期的购买记录，判断哪些购买与当前用户意图重复（即相似，不需要再次推荐），哪些购买与当前意图相关（提示用户考虑补充其他物品），并基于这些判断生成优化后的任务分解，注意，每一个分解的任务都是一个购买任务。

    用户输入：
    {user_input}

    用户近期购买记录：
    {purch_text}

    请输出严格的 JSON，包含以下字段：
    - "意图分析": 简短说明用户的核心需求
    - "purchase_decisions": 列表，每项包含 {{"item": "...", "decision": "similar/relevant/none", "reason": "..."}}
    - decision 说明： "similar" = 已买过类似，不建议重复；"relevant" = 与当前意图相关，可提示补充；"none" = 与当前需求无关
    - "任务分解": 列表，优化后的购物任务
    - "说明": 可选，说明哪些任务被排除、哪些是基于历史补充的，哪些是基于购买任务添加的

    示例：
    {{
    "意图分析": "用户准备去登山，需要登山装备",
    "purchase_decisions": [
        {{"item":"Asics 跑鞋","decision":"similar","reason":"已在近期买过跑鞋，功能类似,故无需购买鞋子品类"}},
        {{"item":"登山袜","decision":"relevant","reason":"与登山相关，可提示用户补充鞋子"}},
        {{"item":"休闲鞋","decision":"none","reason":"已买过鞋子，但与登山无关"}}
    ],
    "任务分解": [
        "购买防晒帽",
        "购买水壶"
    ],
    "说明": ["由于近期已购买跑鞋，与登山鞋功能相似，故不需要再购买登山鞋","基于登山袜建议补充登山鞋"]
    }}

    ，
    {{
    "意图分析": "用户准备去登山，需要登山装备",
    "purchase_decisions": [
        {{"item":"登山袜","decision":"relevant","reason":"与登山相关，可提示用户补充鞋子"}},
    ],
    "任务分解": [
        "购买登山鞋"
        "购买防晒帽",
        "购买水壶"
    ],
    "说明": ["客户近期购买过登山袜，准备去登山，需要登山装备且近期无购买鞋相关记录，建议补充登山鞋","客户近期准备登山，需要防晒帽","客户近期准备登山，可能需要购买水壶"]
    }}
    """
        return prompt
    def run(self, user_input, user_id="001"):
        print(f'用户输入:{user_input}')
        path=os.path.join(self.memory_dir,f"{user_id}.json")
        user_json=self._load_purchase(path)
        purchases = self.sanitize_purchase_records(user_json)
        
        recent = self.recent_purchases(purchases, days=90)
        purchases_summary = [
            f"{r.get('time')} | {r.get('item')} | 价格:{r.get('price')}"
            for r in recent
        ]

        prompt = self.build_prompt_with_purchases(user_input, purchases_summary)
        try:
            response = self.llm.invoke(prompt)
         
        except Exception as e:
            print("Error during LLM invocation:", e)
            return None
        #print(response['content'])
        response=extract_json_from_model_output(response)
        #print(response)
        return response


if __name__=='__main__':
    llm=DeepSeekChat()
    decisionagent=Decision_Agent(llm)
    user_input='我准备两个月后秋天去登山，应该做哪些准备？'
    user_input=user_input
    result=decisionagent.run(user_input)
    print(result)

