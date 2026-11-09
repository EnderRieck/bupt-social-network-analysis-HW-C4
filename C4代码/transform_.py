import json


# 说明：原代码包中缺少本文件，只保留了__pycache__/transform_.cpython-310.pyc，
# 以下两个函数根据该pyc的字节码还原，行为与原实现一致。
def to_json(input):
    try:
        return json.dumps({"data": input}, ensure_ascii=False)
    except TypeError as e:
        raise ValueError(f"无法将输出对象转换为 JSON: {e}")


def from_json(input):
    try:
        return json.loads(input)["data"]
    except TypeError as e:
        raise ValueError(f"无法将输出对象转换为 JSON: {e}")
