import json
import re
import readline  # 让input()支持按字符删除中文，避免退格产生残缺的UTF-8字节
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from new_test_deepseek import DeepSeekChat
from multiplatform_agent import Multi_Platform_Agent
from prefer_rec import RecQueryComposer
from task_compose import Decision_Agent
from search_agent2 import QueryConditionAgent
from search_a import Search_agent
from web_search_agent import Web_Search_Agent
import datetime
from prefer_rec import RecQueryComposer
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def intent_recognition(input,history):
    """返回 0=普通闲聊，1=购物意图，2=需要联网查询实时或外部信息"""
    prompt=f"""
    你是一个用户意图判断助手，你将收到用户历史聊天摘要和当前用户输入，请判断当前用户输入属于哪一类，只输出一个数字：
    1：存在购物意图（想买东西、询问购物准备、需要商品推荐）
    2：需要联网查询才能回答（天气、新闻、近期赛事或活动、实时价格行情、某商品的最新评测或发布信息等依赖实时或外部信息的问题），但没有明确的购买意图
    0：以上都不是的普通闲聊或常识问答
    注意：购物意图优先于联网查询，即同时满足1和2时输出1。不要添加其他任何内容。
    示例
    1.
    历史聊天摘要： null
    用户输入：我最近两个月打算去爬山，在秋天，应该做什么准备呢
    输出：1

    2.
    历史聊天摘要： null
    用户输入：今天北京天气怎么样？
    输出：2
    
    3.
    历史聊天摘要： null
    用户输入：我准备两个月后 在冬天登山，应该做哪些购物准备？
    输出：1
    
    4.
    历史聊天摘要：今天可能下雨
    用户输入：看看天气，如果下雨的话我有可能会买雨具
    输出：1
    
    5.
    历史聊天摘要：null
    用户输入：看看天气，如果下雨的话我有可能会买雨具
    输出：2

    6.
    历史聊天摘要：null
    用户输入：最近有什么新出的跑鞋评测吗
    输出：2

    7.
    历史聊天摘要：null
    用户输入：你好，你是谁？
    输出：0
    
    历史聊天摘要：{history}
    以下为用户输入{input}
    """
    response=llm.invoke(prompt).content.strip()
    #print(response)
    if response.startswith('1'):
        return 1
    if response.startswith('2'):
        return 2
    return 0

def confirm_purchase_intent():
    print("助手: 您确定要进行购物推荐吗？(是/否)")
    response = input("用户: ").strip().lower()
    return response == "是"

def maybe_summarize(history_obj: BaseChatMessageHistory, llm,message_count, max_msgs: int =5):
    
    msgs = history_obj.messages
    # 如果消息数量没超阈值，什么都不做
    if message_count <= max_msgs:
        return message_count

    # 拼接成一段对话文本
    message_count=1
    print('开始生成摘要')
    dialog_txt = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in msgs
    )
    # 调用 LLM 生成要点式摘要
    summary = llm.invoke(f"请对以下对话内容进行要点式总结：\n\n{dialog_txt}").content.strip()

    # 清空旧历史，只保留一条“AI摘要”作为上下文开头
    history_obj.clear()
    history_obj.add_ai_message("（对话摘要）" + summary)
    return message_count

def run_chat(session_id: str, user_input: str,message_count: int):
    # 拿到这个会话的 history 对象
    message_count+=1
    history_obj = get_session_history(session_id)

    #print(history_obj)
    print('准备回复中')
    response = with_message_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            ).content.strip()

    # ③ 再次摘要（把最新的 user+AI 也纳入压缩）
    message_count=maybe_summarize(history_obj, llm, message_count,max_msgs=3)

    return response,message_count
    
def run_workflow(user_input,user_info):
    decision_result = decisionAgent.run(user_input)
    print(decision_result)
    intent = decision_result['意图分析']
    tasks = decision_result['任务分解']
    new_query = []
    for task in tasks:
        if '鞋' in task:
            input_recommend=[user_info,task,True]
            new_task = recommand_rec_Agent.compose_rec_query(input_recommend)
            new_query.append(new_task)
    for query in new_query:
        print(query)
        results=searchagent.run(query,k=3)#rank
        print(results)
        price_result = multiplatformAgent.run(results)
        print(price_result)
        return price_result
    
    return '未找到符合需求的商品，请修改描述或条件后重试'
if __name__=="__main__":

    llm = DeepSeekChat()
    decisionAgent = Decision_Agent(llm)
    # userprefer_agent = User_Prefer_Agent(llm)  # user_prefer.py源码缺失，且该对象后续未被使用
    multiplatformAgent = Multi_Platform_Agent(llm)
    recommand_rec_Agent=RecQueryComposer(llm)
    websearch_agent=Web_Search_Agent(llm)
    searchagent=Search_agent(db_path="./shoe_vector_store.db",llm=llm)
    # 用户信息路径
    user_info_path = './personal_inf.json'  # 原为服务器绝对路径
    user_info = recommand_rec_Agent.get_user_info(user_info_path)
    """
    # 初始化向量数据库
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb_list = [
        "/data1/wuquanda/agent_start/personal_assistant/tb_shoe_chroma_dbs",
        "/data1/wuquanda/agent_start/personal_assistant/jd_shoe_chroma_dbs",
        "/data1/wuquanda/agent_start/personal_assistant/pdd_shoe_chroma_dbs"
    ]
    query_agent = QueryConditionAgent(llm, vectordb_list, embedding_model)
    """
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个商家聊天机器人，负责和客户进行对话，如果用户输入和购物无关，请在简短回答问题后隐晦地提醒用户你只是个购物推荐助手，如果用户展现比较强的否定或者攻击性意图，请坦率地承认自己错了注意，不要思考太久，回复不要超过50个字。",
            ),
            MessagesPlaceholder(variable_name='history'),
            ("human","{input}"),
        ]
    )
    runnable= prompt | llm
    session_id= "abc123"
    store = {}
    with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    message_count=0
    response_flag=0
    while True:
        #print(f'message_count={message_count}')
        user_input = input("用户: ").strip()
        # 丢弃终端传入的残缺字节（否则发请求时会报UnicodeEncodeError: surrogates not allowed）
        user_input = user_input.encode("utf-8", "surrogateescape").decode("utf-8", "ignore")
        history_obj = get_session_history(session_id)
        history_context = "\n".join([msg.content for msg in history_obj.messages])
        if user_input=='是' and response_flag:
            response_flag=0
            ##下单...
        intent = intent_recognition(user_input,history_context)
        if intent == 2:
            # 联网查询分支：搜索 -> 基于结果回答，并写入对话历史
            message_count += 1
            response = websearch_agent.answer(user_input, history_context)
            history_obj.add_user_message(user_input)
            history_obj.add_ai_message(response)
            message_count = maybe_summarize(history_obj, llm, message_count, max_msgs=3)
        elif intent == 1:
            message_count += 1
            history_obj.add_user_message(user_input)
            history_obj = get_session_history(session_id)
            message_count=maybe_summarize(history_obj, llm, message_count,max_msgs=3)
            if confirm_purchase_intent():
                
                response=run_workflow(user_input,user_info)
            else:
                response = "好的，如果您有其他问题，请随时告诉我。"
        
        else:
            response,message_count=run_chat(session_id,user_input,message_count)
        print(f"助手:{response}")
        if "请确认购买信息" in response:
            response_flag=1
        