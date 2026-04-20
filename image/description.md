digraph Multimodal_RAG_AppleSquare {
    rankdir=TB;
    compound=true;
    splines=ortho;
    nodesep=0.3;
    ranksep=0.4;
    margin="0.2,0.2";
    
    node [shape=box, style="rounded,filled", fontname="SF Pro Text, Helvetica Neue, Microsoft YaHei", fontsize=10, penwidth=0.6, margin="0.35,0.28"];
    edge [fontname="SF Pro Text", fontsize=7, color="#86868B", penwidth=1.0, arrowhead="normal"];
    
    // ========== 全局背景 ==========
    bgcolor="#F5F5F7";
    
    // ========== 标题（独立显示）==========
    Title [label="📊 多模态 RAG 文档问答系统  —  总结 · 检索 · 解释", 
           shape=plaintext, 
           fontname="SF Pro Text, Helvetica Neue", 
           fontsize=16, 
           fontweight=bold, 
           fontcolor="#1D1D1F",
           fillcolor="#E8E8ED",
           style=filled,
           margin="0.3,0.15"];
    
    // ========== 第一层：输入 + 解析（蓝色系）==========
    subgraph cluster_input_parse {
        label="📥 输入层  +  🔧 文档解析层";
        style=dashed;
        color="#4A90D9";
        penwidth=2;
        fontname="SF Pro Text";
        fontsize=11;
        fontweight=bold;
        fontcolor="#4A90D9";
        margin="12";
        fillcolor="#F0F8FF";
        
        // 输入
        subgraph cluster_input {
            label="输入层";
            style=filled;
            fillcolor="#E8F0FE";
            color="#4A90D9";
            fontname="SF Pro Text";
            fontsize=9;
            fontweight=bold;
            fontcolor="#1D1D1F";
            margin="6";
            
            Input [label="📄 用户输入\nPDF / 截图 / 问题", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
        }
        
        // 文档解析
        subgraph cluster_parse {
            label="文档解析层";
            style=filled;
            fillcolor="#E8F0FE";
            color="#4A90D9";
            fontname="SF Pro Text";
            fontsize=9;
            fontweight=bold;
            fontcolor="#1D1D1F";
            margin="6";
            
            Preprocess [label="📑 OCR + 布局分析\nPaddleOCR · LayoutParser", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
            Chunking [label="✂️ 多模态分块\n文本块 · 表格块 · 图像块", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
        }
    }
    
    // ========== 第二层：存储（橙色系）==========
    subgraph cluster_storage {
        label="💾 存储层";
        style=dashed;
        color="#F39C12";
        penwidth=2;
        fontname="SF Pro Text";
        fontsize=11;
        fontweight=bold;
        fontcolor="#F39C12";
        margin="12";
        fillcolor="#FFFBF0";
        
        Embedding [label="🔢 向量化\nText(bge-small) · Image(CLIP)", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
        VectorDB [label="🗄️ 向量数据库 (FAISS)\n文本块 + 图像描述\n元数据: 页码 / 图号", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
    }
    
    // ========== 第三层：检索（紫色系）==========
    subgraph cluster_retrieval {
        label="🔍 检索层 (RAG 核心)";
        style=dashed;
        color="#8E44AD";
        penwidth=2;
        fontname="SF Pro Text";
        fontsize=11;
        fontweight=bold;
        fontcolor="#8E44AD";
        margin="12";
        fillcolor="#F8F0FF";
        
        Query [label="❓ 用户问题", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
        Retrieval [label="📚 多模态检索\nQuery Embedding → FAISS\nTop-K 召回 · 聚合结果", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
    }
    
    // ========== 第四层：推理 + 输出（红色系）==========
    subgraph cluster_infer_output {
        label="🧠 推理层  +  📤 输出层";
        style=dashed;
        color="#E74C3C";
        penwidth=2;
        fontname="SF Pro Text";
        fontsize=11;
        fontweight=bold;
        fontcolor="#E74C3C";
        margin="12";
        fillcolor="#FFF5F5";
        
        // 推理层
        subgraph cluster_generate {
            label="推理层 (核心)";
            style=filled;
            fillcolor="#FFE8E8";
            color="#E74C3C";
            fontname="SF Pro Text";
            fontsize=9;
            fontweight=bold;
            fontcolor="#1D1D1F";
            margin="6";
            
            Prompt [label="📝 Prompt 组装\n检索上下文 + 原始图片\n溯源约束 ([Page X], [图 Y])", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
            VLM [label="🤖 多模态大模型 (VLM)\nQwen2.5-VL\n总结 → 检索 → 解释", fillcolor="#FFDADA", fontcolor="#1D1D1F", margin="0.45,0.35", penwidth=1.8, color="#E74C3C"];
        }
        
        // 输出层
        subgraph cluster_output {
            label="输出层";
            style=filled;
            fillcolor="#FFE8E8";
            color="#E74C3C";
            fontname="SF Pro Text";
            fontsize=9;
            fontweight=bold;
            fontcolor="#1D1D1F";
            margin="6";
            
            Output [label="✨ 答案输出\n自然语言回答\n引用页码 [Page 3] · 图号 [图 2]", fillcolor="#FFFFFF", fontcolor="#1D1D1F", margin="0.4,0.3"];
        }
    }
    
    // ========== 纵向连接（主流程）==========
    Input -> Preprocess [color="#4A90D9", penwidth=1.2];
    Preprocess -> Chunking [color="#4A90D9", penwidth=1.2];
    Chunking -> Embedding [color="#F39C12", penwidth=1.2];
    Embedding -> VectorDB [color="#F39C12", penwidth=1.2];
    
    VectorDB -> Retrieval [color="#8E44AD", penwidth=1.2];
    Query -> Retrieval [color="#8E44AD", penwidth=1.2];
    
    Retrieval -> Prompt [color="#E74C3C", penwidth=1.2];
    VectorDB -> Prompt [color="#8E44AD", penwidth=1.2, style=dashed, label=" 图片路径 ", fontcolor="#8E44AD", fontsize=6];
    
    Prompt -> VLM [color="#E74C3C", penwidth=1.5];
    VLM -> Output [color="#E74C3C", penwidth=1.5];
    
    // ========== 反馈回路（多轮对话）==========
    Output -> Query [color="#34C759", penwidth=1.0, style=dashed, label=" 多轮对话 ", fontcolor="#34C759", constraint=false];
    
    // ========== 同层节点水平排列 ==========
    {rank=same; Input; Preprocess}
    {rank=same; Chunking; Embedding}
    {rank=same; VectorDB; Query}
    {rank=same; Retrieval}
    {rank=same; Prompt; Output}
    {rank=same; VLM}
    
    // ========== 控制整体尺寸 ==========
    size="7.5,8.5";
    ratio="auto";
}
