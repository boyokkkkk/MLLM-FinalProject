import chainlit as cl
import asyncio

# ==========================================
# 1. 欢迎页与快捷启动指令 (降低使用门槛)
# ==========================================
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="📚 提取核心考点",
            message="请帮我总结这份课件的核心考点，并生成复习大纲。",
            icon="/public/book.svg", # 建议在项目目录下建 public 文件夹放图标，或直接靠文字
        ),
        cl.Starter(
            label="📝 生成模拟试卷",
            message="基于我上传的资料，生成一份包含单选、多选和简答题的期末模拟试卷。",
            icon="/public/test.svg",
        ),
        cl.Starter(
            label="🤔 费曼学习法",
            message="我不太懂某个概念，请用费曼学习法给我讲明白。",
            icon="/public/brain.svg",
        ),
        cl.Starter(
            label="🎧 录音转写与摘要",
            message="这是我上课的录音，请帮我转写并提取重点。",
            icon="/public/mic.svg",
        )
    ]

# ==========================================
# 2. 侧边栏设置与应用初始化 (打造专业工作台体验)
# ==========================================
@cl.on_chat_start
async def start():
    # 设置侧边栏配置项
    settings = await cl.ChatSettings(
        [
            cl.input.Select(
                id="Model",
                label="🧠 AI 引擎选择",
                values=["GPT-4o (多模态推荐)", "Claude 3.5 Sonnet (逻辑严密)", "Gemini 1.5 Pro"],
                initial_index=0,
            ),
            cl.input.Slider(
                id="Difficulty",
                label="🔥 测验难度系数 (1-10)",
                initial=5,
                min=1,
                max=10,
                step=1,
            ),
            cl.input.Switch(id="FocusMode", label="🧘 专注模式 (屏蔽无关闲聊)", initial=True),
        ]
    ).send()

    # 高级感的欢迎文案，支持 Markdown 结构
    welcome_msg = """
### 🎓 欢迎来到全能期末复习空间

我是你的专属多模态 AI 助教。为了让你高效备考，我已准备好以下核心能力：

* **📄 文档解析**：拖拽 PDF/PPT/Word 讲义，我来梳理知识框架。
* **🎙️ 语音转录**：上传课堂录音，一键生成重点摘要与时间戳。
* **📸 图像识别**：拍照上传板书或复杂图表，我为你详细拆解。

> **提示**：直接在下方输入框上传你的复习资料，或者点击上方快捷卡片开启复习之旅！
    """
    
    # 使用自定义头像提升品牌感
    await cl.Message(
        content=welcome_msg, 
        author="复习全能助手"
    ).send()

# ==========================================
# 3. 核心交互逻辑与多模态处理
# ==========================================
@cl.on_message
async def main(message: cl.Message):
    # 模拟处理多模态文件
    has_files = len(message.elements) > 0
    
    # UI 亮点：使用 Step 展示思考与处理过程，避免用户干等
    async with cl.Step(name="知识引擎分析中...", type="tool") as step:
        if has_files:
            step.input = f"正在读取 {len(message.elements)} 个附件资料"
            await asyncio.sleep(1.5) # 模拟 OCR 或音频转写耗时
            step.output = "✅ 资料解析完毕，已提取 15 个关键考点。"
        else:
            step.input = "正在分析用户学习诉求"
            await asyncio.sleep(0.5)
            step.output = "✅ 诉求解析明确。"

    # 构建富文本响应与内联元素
    elements = []
    if has_files:
        # 假装生成了一个知识大纲的文本组件
        elements.append(
            cl.Text(
                name="结构化知识大纲", 
                content="1. 核心概念A\n2. 核心公式B\n3. 重点案例C...", 
                display="inline"
            )
        )
        reply_content = "我已经仔细阅读了你的资料。基于你的上传内容，我为你整理了一份**结构化大纲**（见下方）。接下来你希望怎么巩固这些知识？"
    else:
        reply_content = f"针对你提到的：**“{message.content}”**，我已经梳理好了相关的复习思路。你想先进行哪一步？"

    # UI 亮点：使用 Action 按钮引导下一步操作，减少打字
    actions = [
        cl.Action(name="action_flashcards", value="flashcards", label="📇 生成记忆卡片 (Anki)"),
        cl.Action(name="action_quiz", value="quiz", label="🎯 开始随堂测验"),
        cl.Action(name="action_explain", value="explain", label="💡 重点深入讲解")
    ]

    await cl.Message(
        content=reply_content,
        elements=elements,
        actions=actions,
        author="复习全能助手"
    ).send()

# ==========================================
# 4. 按钮回调处理 (交互闭环)
# ==========================================
@cl.action_callback("action_flashcards")
async def on_action_flashcards(action: cl.Action):
    # 移除按钮，防止重复点击
    await action.remove()
    
    # 模拟生成闪卡
    await cl.Message(
        content="✅ 已经为你从资料中提取了 **10 张间隔重复记忆卡片**。让我们开始第一张背诵：\n\n> **问题**：什么是二八定律（帕累托法则）在复习中的应用？",
        author="复习全能助手"
    ).send()

@cl.action_callback("action_quiz")
async def on_action_quiz(action: cl.Action):
    await action.remove()
    await cl.Message(
        content="🎯 测验模式已开启！已根据你的难度设置（当前难度），生成了第一道单选题...\n\n**(A) ... (B) ... (C) ... (D) ...**\n\n请回复你的选项。",
        author="复习全能助手"
    ).send()