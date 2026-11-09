import os
import requests
from new_test_deepseek import DeepSeekChat


class Web_Search_Agent:
    """联网搜索Agent：调用DeepSeek官方Search/Fetch API，把搜索结果喂给LLM生成带引用的回答。

    Search API与chat接口共用同一个api_key。接口说明：https://api-docs.deepseek.com/guides/search
    计费：每1000次成功搜索0.75美元，空结果与失败请求不计费。
    """

    SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
    FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"

    def __init__(self, llm, api_key=None, count=5, timeout=30):
        self.llm = llm
        self.api_key = api_key or os.getenv("api_key")
        self.count = count          # 每次搜索返回的结果条数，范围1到50
        self.timeout = timeout

    # ---------- 底层接口 ----------
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def search(self, query, count=None, need_content=False):
        """调用搜索接口，返回results列表。每条含title/url/snippet/summary/date/site_name等字段。"""
        query = query.strip()[:128]  # 接口限制query为1到128个字符
        payload = {
            "query": query,
            "count": count or self.count,
            "need_content": need_content,
        }
        resp = requests.post(self.SEARCH_URL, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def fetch(self, url):
        """抓取单个网页正文，返回{"title": ..., "content": markdown文本}。"""
        resp = requests.post(self.FETCH_URL, headers=self._headers(), json={"url": url}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---------- 与LLM配合 ----------
    def compose_query(self, user_input, history=""):
        """让LLM把用户问题改写成一条简短的搜索关键词。"""
        prompt = f"""你是一个搜索关键词生成助手。请根据对话历史摘要和用户当前输入，生成一条用于搜索引擎的查询语句。
要求：只输出查询语句本身，不超过30个字，不要加引号、解释或其他内容。
对话历史摘要：{history or "null"}
用户输入：{user_input}
查询语句："""
        query = self.llm.invoke(prompt).content.strip().strip('"“”')
        return query or user_input

    @staticmethod
    def format_results(results):
        """把搜索结果整理成带编号的文本，供LLM阅读和引用。"""
        lines = []
        for i, r in enumerate(results, 1):
            summary = r.get("summary") or r.get("snippet") or ""
            meta = "，".join(x for x in [r.get("site_name"), (r.get("date") or "")[:10]] if x)
            lines.append(f"[{i}] {r.get('title', '')}（{meta}）\n摘要：{summary}\n链接：{r.get('url', '')}")
        return "\n\n".join(lines)

    def answer(self, user_input, history=""):
        """完整流程：生成关键词 -> 搜索 -> 基于结果回答。返回最终回复字符串。"""
        query = self.compose_query(user_input, history)
        print(f"联网搜索：{query}")
        try:
            results = self.search(query)
        except requests.RequestException as e:
            print(f"搜索接口调用失败：{e}")
            fallback = self.llm.invoke(user_input).content.strip()
            return f"（联网搜索暂时不可用，以下为未联网的回答）{fallback}"

        if not results:
            return "没有搜索到相关内容，请换个说法再试试。"

        context = self.format_results(results)
        prompt = f"""你是一个购物推荐助手，下面是针对用户问题的联网搜索结果。请基于这些结果回答用户的问题。
要求：
1. 只使用搜索结果中的信息，不要编造；若结果不足以回答，如实说明。
2. 回答中引用到某条结果时，在句末用[编号]标注。
3. 回答简洁，不超过200字；最后另起一行列出引用过的来源，格式为"[编号] 标题 链接"。

搜索结果：
{context}

对话历史摘要：{history or "null"}
用户问题：{user_input}
"""
        return self.llm.invoke(prompt).content.strip()


if __name__ == "__main__":
    llm = DeepSeekChat()
    agent = Web_Search_Agent(llm)
    print(agent.answer("北京这周末适合爬山吗，天气怎么样"))
