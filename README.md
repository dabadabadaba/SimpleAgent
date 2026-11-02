"# SimpleAgent" 
# TA Agent

基于 **LangGraph** 的智能体工作流管理平台。通过 Web 界面提交任务，由 Agent 自主调用工具（文件读写、Excel 处理、LLM 生成等）完成任务，并提供实时的任务进度、日志展示与人工介入（等待输入 / 强制中断）能力。

## 功能特性

- 🕸️ **LangGraph 工作流引擎**：多节点状态图驱动，支持中断-恢复（interrupt/resume）的人机协作模式
- 🌐 **Web 管理界面**：任务创建、列表、详情一条龙
  - 任务信息栏：UUID、状态、当前节点、Token 用量实时展示
  - 任务进度栏：Markdown 渲染的工作流进度
  - 任务日志栏：实时日志流 + 自动滚动（仅在有新行写入时滚动）
  - 等待用户输入时可在详情页直接回复，支持强制中断
- 🧩 **插件机制**：`Plugin/` 目录下的每个子文件夹即一个插件，可自定义节点、工具与提示词
- 🔌 **多基座模型**：通过 `LLM_config.json` 配置 DeepSeek / Qwen / Mimo 等模型，任务创建时选择
- 📊 **Excel / CSV 处理**：内置 pandas + openpyxl 工具

## 项目结构

```
TA_AGENT/
├── AgentFrame.py            # FastAPI 主服务（HTTP 接口 + 任务调度 + LangGraph 图构建）
├── index.html               # 首页：任务创建 + 任务列表
├── task_detail.html         # 任务详情页：信息/进度/日志 + 输入交互
├── LLM_config.json          # 基座模型配置（key 名称在首页下拉框展示）
├── requirements.txt         # Python 依赖
├── Assets/                  # Agent 核心模块
│   ├── TA_Nodes.py          #   Agent 工作流节点（LLM 初始化、图构建）
│   ├── TA_Tools.py          #   工具定义（测试用例生成等）
│   ├── FileTools.py         #   文件操作工具（读写/搜索/Excel）
│   ├── GeneralTools.py      #   通用工具
│   ├── LLMGenerate.py       #   独立 LLM 调用封装
│   ├── DS_Utilities.py      #   工具加载 / Token 计数 / Excel 解析
│   ├── Logger.py            #   日志记录
│   ├── OrmsgpackConsts.py   #   ormsgpack 兼容补丁（Windows 内存崩溃规避）
│   └── General/             #   通用节点/工具/提示词配置（yaml）
├── Plugin/                  # 插件目录（每个子文件夹一个插件）
│   ├── TA_SDK/              #   示例插件：节点、工具、提示词配置
│   └── AutomationDesk/      #   预留插件（空）
├── MCP/                     # MCP 服务预留目录（空）
└── task_workspaces/         # 任务运行工作区（运行时生成，不入库）
```

## 快速开始

### 1. 环境准备

要求 Python 3.11+。

```bash
# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置基座模型

编辑 `LLM_config.json`，每个 key 对应一个模型配置（首页下拉框会展示这些 key）：

```json
{
  "qwen_flash": {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3.7-flash",
    "temperature": 0.15,
    "api_key": "QWEN_API_KEY",
    "compress_trigger_token": 22768,
    "api_type": "openai"
  }
}
```

- `api_key` 字段是**环境变量名**（不是密钥本身），启动服务前需设置对应的环境变量，例如 `set QWEN_API_KEY=sk-xxx`
- `api_type` 为 `deepseek` 时走 DeepSeek 专用 SDK，否则走 OpenAI 兼容接口

### 3. 启动服务

```bash
python AgentFrame.py
```

启动后访问：

| 地址 | 说明 |
|------|------|
| http://localhost:9988 | 首页（创建任务 / 任务列表） |
| http://localhost:9988/docs | FastAPI 接口文档 |

### 4. 创建任务

1. 首页填写 **ROOT_WORKSPACE**（任务工作目录的根路径）
2. 选择**基座模型**（默认选列表最后一个）
3. 勾选**插件**（可多选）
4. 输入任务消息，点击启动
5. 自动跳转任务详情页，实时查看进度与日志

## 命令行直跑（无 Web 界面）

```bash
python Assets/TA_Nodes.py [llm_name] [plugin1] [plugin2 ...]
```

- 工作区根目录可通过环境变量 `TA_WORKSPACE_ROOT` 指定，默认使用项目下的 `demo_workspace/`
- 进程内交互式等待用户输入（输入 `exit` 强制结束）

## 插件开发

在 `Plugin/` 下新建文件夹（如 `MyPlugin/`），放置以下配置文件即可注册为插件：

```
MyPlugin/
├── nodes_config.yaml   # 节点定义（可选，覆盖/新增节点）
├── tools_config.yaml   # 工具注册
└── Prompt.yaml         # 节点提示词
```

配置格式参考 `Plugin/TA_SDK/` 与 `Assets/General/` 下的同名文件。

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/submit` | 创建任务 `{msg, root_workspace, llm_base, plugins}` |
| GET | `/task/{uuid}/detail` | 任务详情页 |
| GET | `/task/{uuid}/status` | 任务状态（进度/节点/Token） |
| GET | `/task/{uuid}/log/tail` | 任务日志尾部 N 行 |
| POST | `/task/{uuid}/send` | 向等待输入的任务发送回复 |
| POST | `/task/{uuid}/terminate` | 强制中断任务 |
| GET | `/llm-config` | 基座模型配置 key 列表 |
| GET | `/plugins` | 插件（子文件夹）列表 |

## 许可证

Apache-2.0

