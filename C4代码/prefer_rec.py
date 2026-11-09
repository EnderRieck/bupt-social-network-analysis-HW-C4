import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
class RecQueryComposer:
    def __init__(self, llm):
        self.llm = llm
    
    # 用户偏好相关方法
    def get_user_info(self, user_info_path):
        """获取用户信息"""
        user_info_path = user_info_path
        with open(user_info_path, 'r') as f:
            user_info = json.load(f)
        return user_info

    def extract_prefer_from_shorthistory(self, user_prefer, short_history):
        """从短期历史中提取用户偏好"""
        prompt = f"""你是一个用户偏好提取助手，你将根据过往的用户偏好内容和短期内用户对话消息历史，提取总结新的用户偏好并以json格式输出。
        注意：若历史用户偏好与聊天记录存在冲突，若聊天记录中存在强烈的意愿，则以聊天记录为主，若无强烈意愿且无大冲突，则尽量综合两者，若聊天记录中未提到某项偏好，则完全保存这项偏好，若某个总结的子偏好从属性上和已有偏好属性相似，则合成为已有偏好属性，例如，已有偏好中有color这一属性，新总结的偏好是色泽这一属性，合成为color
        示例输入：
        {{user_prefer:["color":"white"]}}
        history:"我最近买东西更喜欢暗色的而不像以前喜欢粉色或者白色这些亮色的，而且买东西更偏向是运动系的，宽松舒适透气。"
        示例输出：{{"prefer":["color":"喜欢暗色而不是亮色","style":"宽松，舒适，透气，偏运动风格"]}}
        现在，请分析以下输入：
        {user_prefer},
        history:{short_history}
        """
        response = self.llm.invoke(prompt)
        return extract_json_from_model_output(response)
        code_block_match = re.search(r"```(?:json)?\s*(.*?)```", response.content, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            json_str = response.content.strip()
        try:
            new_prefer = json.loads(json_str)
        except json.JSONDecodeError:
            new_prefer = {}
        return new_prefer
        
    def update_prefer(self, user_info_path, short_history):
        """更新用户偏好"""
        user_info_path = user_info_path
        with open(user_info_path, 'r') as f:
            user_info = json.load(f)
        user_prefer = user_info['prefer']
        #print(f'旧用户偏好{user_prefer}')
        new_prefer = self.extract_prefer_from_shorthistory(user_prefer, short_history)
        #print(new_prefer)
        user_info['prefer'] = new_prefer['prefer']
        #print(f"新用户偏好{new_prefer['prefer']}")
        with open(user_info_path, 'w', encoding="utf8") as f:
            json.dump(user_info, f, indent=4, ensure_ascii=False)
    
    # 推荐相关方法
    def build_rec_query_prompt(self, user_data, user_input, use_prefer=True):
        """生成推荐提示字符串，根据use_prefer决定是否包含用户偏好"""
        if use_prefer:
            # 格式化偏好字典为字符串，例如 "color: 白色, 粉色; style: 宽松，舒适，透气，偏运动风格"
            preferences = user_data['prefer']
            prefer_str = "; ".join([
                f"{key}: {', '.join(value) if isinstance(value, list) else value}"
                for key, value in preferences.items()
            ])
            prompt = f"""
            你是一个用户输入润色助手，你将根据用户的个人信息和偏好，润色用户的输入，生成一个新的、包含用户数据和偏好的查询语句，用于传递给信息查询agent,输出只包含查询语句，不要有其他类似根据xxx描述语句。

            用户的个人信息如下：
            性别: {user_data['gender']}
            年龄: {user_data['age']}
            体重: {user_data['weight']}
            身高: {user_data['height']}

            用户的偏好如下：
            {prefer_str}

            用户的输入是：
            {user_input}

            请生成一个新的查询语句，整合用户的输入和相关个人信息及偏好。
            """
        else:
            prompt = f"""
            你是一个用户输入润色助手，你将根据用户的个人信息，润色用户的输入，生成一个新的、包含用户数据的查询语句，用于传递给信息查询agent，输出只包含查询语句，不要有其他类似根据xxx描述语句。

            用户的个人信息如下：
            性别: {user_data['gender']}
            年龄: {user_data['age']}
            体重: {user_data['weight']}
            身高: {user_data['height']}

            用户的输入是：
            {user_input}

            请生成一个新的查询语句，整合用户的输入和相关个人信息。
            """
        return prompt.strip()  # 去除多余空白
    
    def compose_rec_query(self, input_):
        """生成推荐查询语句"""
        user_data, user_input, use_prefer = input_
        prompt = self.build_rec_query_prompt(user_data, user_input, use_prefer)
        #print(prompt)
        new_query = self.llm.invoke(prompt)  # 调用语言模型生成新查询
        return new_query.content.strip()

if __name__ == '__main__':
    # 测试代码
    llm = DeepSeekChat()
    assistant = RecQueryComposer(llm)
    
    # 测试用户信息获取
    user_info_path = '/data1/wuquanda/agent_start/personal_assistant/personal_inf.json'
    user_info = assistant.get_user_info(user_info_path)
    print("用户信息:", user_info)
    
    # 测试偏好更新
    user_history = '我最近买东西更偏向是运动系的，宽松舒适透气,而且相比以前，最近又多喜欢一种粉色。'
    assistant.update_prefer(user_info_path, user_history)
    
    # 测试推荐生成
    user_input = '我准备在秋天登山，打算买一双适合运动、防滑的鞋子'
    
    # 测试带用户偏好的推荐
    input_data = (user_info, user_input, True)
    new_query = assistant.compose_rec_query(input_data)
    print(f"带用户偏好的结果：{new_query}")
    
    # 测试不带用户偏好的推荐
    input_data2 = (user_info, user_input, False)
    new_query2 = assistant.compose_rec_query(input_data2)
    print(f"不带用户偏好的结果：{new_query2}")