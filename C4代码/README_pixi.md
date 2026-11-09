# 用pixi运行本项目

原`environment.yml`是Linux服务器上导出的conda环境，`pixi.toml`/`pixi.lock`是等价的pixi配置，支持osx-arm64（Apple Silicon Mac）和linux-64两个平台。

## 1. 安装环境

在`C4代码`目录下执行：

    pixi install

首次安装会下载torch、transformers等较大的包，macOS上约2GB左右。

## 2. 设置DeepSeek API密钥

    export api_key="你的密钥"

## 3. 运行

    pixi run check     # 自检：确认核心依赖可导入
    pixi run chat      # 运行主程序chat2.py

或者进入环境的shell后按原方式运行：

    pixi shell
    python chat2.py

## 说明

- `transform_.py`为补回的文件：原代码包只带了`__pycache__/transform_.cpython-310.pyc`而缺少源码，`search_agent2.py`导入时会报`ModuleNotFoundError`，现已按字节码还原。
- 首次运行会从HuggingFace下载嵌入模型`sentence-transformers/all-MiniLM-L6-v2`（约90MB）。国内网络不通时可先执行`export HF_ENDPOINT=https://hf-mirror.com`。
- 原conda环境中的`nvidia-*`、`triton`等CUDA包只存在于Linux，pixi会在linux-64上自动随torch安装，在macOS上则使用CPU/MPS版torch，不需要手动处理。

## 联网搜索功能（新增）

`web_search_agent.py`中的`Web_Search_Agent`调用DeepSeek官方Search API（`POST https://api.deepseek.com/v1/tools/search`），与chat接口共用同一个`api_key`，无需额外配置。

- 触发方式：`chat2.py`的意图识别由两分类改为三分类，0为闲聊、1为购物、2为需要联网查询（天气、新闻、商品最新评测等），识别为2时走联网分支。
- 流程：LLM生成搜索关键词 → 调用Search API取前5条结果 → LLM基于结果回答并标注来源编号。
- 计费：每1000次成功搜索0.75美元，空结果和失败请求不计费。搜索接口不可用时自动回退为不联网回答。
- 接口文档：https://api-docs.deepseek.com/guides/search
