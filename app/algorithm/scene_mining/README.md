# 🚗 Scene Mining Agents — 车载视频交通场景挖掘系统

基于视觉语言模型（VLM）和 LangGraph 多智能体架构，对 360° 车载摄像头视频进行自动化交通场景理解与异常事件识别。

## 系统概述

本系统从大量行车视频中自动识别道路环境、天气光照、道路类型、特殊路段以及潜在交通风险，输出结构化的场景标签与异常事件时间段。

### 核心能力

| 能力 | 说明 |
|------|------|
| **多维场景识别** | 自动判断时间段、光照、天气、路面状态、道路级别和特殊路段 |
| **异常事件检出** | 识别静态障碍、弱势交通参与者异常、车辆轨迹冲突和极端高危事件 |
| **事件时间定位** | 对异常事件输出发生时间段，便于快速回看关键片段 |
| **批量处理** | 支持大规模视频批量分析，配合断点续传机制 |
| **结构化输出** | 场景标签 + 异常事件列表，方便检索、统计和复核 |

## 架构设计

系统提供两种智能体编排模式，均基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建：

### DAG 模式（默认，推荐）

两阶段 Supervisor-Worker 流水线，通过 Send API 实现动态并行 fan-out/fan-in：

```
[START] → [Supervisor 1] → Fan-out(Simple Workers) → [Reducer] → [Supervisor 2] → Fan-out(Complex Workers) → [Output Formatter] → [END]
```

- **Stage 1 — 简单类别**：Supervisor 1 生成任务，Simple Workers 并行处理基础场景类别（时间段、天气、道路等）
- **Stage 2 — 复杂类别**：Supervisor 2 基于 Stage 1 结果规划复杂任务，Complex Workers 对异常事件进行时间片级精细分析

### ReAct 模式

单一 Root Agent 通过工具调用（`spawn_sub_agent`、`query_result`、`review_result`）自主协调所有分析任务：

```
[START] → [Root Agent] → [Output Formatter] → [END]
```

## 识别类别

系统覆盖 **10 个识别维度**，分为基础场景（6 类）和异常事件（4 类）：

| 类型 | 类别 | 复杂度 |
|------|------|--------|
| 基础场景 | 自然时间段、人工光源辅助、气象条件、路面状态与视线干扰、主干道路级别、特殊路段与设施 | Simple |
| 异常事件 | 静态障碍与全局异常 | Simple |
| 异常事件 | 弱势交通参与者异常、车辆轨迹与空间冲突、极端高危与失控事件 | Complex |

## 项目结构

```
scene_mining_agents/
├── main.py                    # 入口：参数解析、批量调度、断点续传
├── langgraph_pipeline.py      # LangGraph 图定义（DAG + ReAct 两种拓扑）
├── qwen_client.py             # VLM API 客户端（兼容 OpenAI 接口）
├── prompts.py                 # Prompt 管理与加载
├── response_parser.py         # 模型输出解析
├── categories.json            # 类别定义与标签枚举
├── config.yaml                # 主配置文件
├── nodes/                     # LangGraph 节点实现
│   ├── supervisor.py          #   Supervisor 1 & 2
│   ├── simple_worker.py       #   简单类别 Worker
│   ├── complex_worker.py      #   复杂类别 Worker（含 clip 工具调用）
│   ├── simple_reducer.py      #   Stage 1 结果聚合
│   ├── output_formatter.py    #   最终输出格式化
│   └── root_agent.py          #   ReAct Root Agent
├── tools/                     # Agent 工具集
│   ├── clip_select.py         #   视频片段截取分析
│   ├── frame_extract.py       #   关键帧提取
│   ├── yolo_detect.py         #   YOLO 目标检测预筛选
│   ├── spawn_sub_agent.py     #   子 Agent 派发（ReAct 模式）
│   ├── query_category_result.py  #   跨类别结果查询
│   └── review_result.py       #   结果复审
├── skills/                    # Prompt 模板库
│   ├── categories/            #   每个类别的专用 Prompt
│   └── tools/                 #   工具说明文档
├── docker/                    # Docker / vLLM 启动脚本
└── select_clip_fallback.py    # Clip 工具 fallback 实现
```

## 快速开始

### 环境要求

- Python 3.11+
- [ffmpeg / ffprobe](https://ffmpeg.org/)（视频处理）
- 兼容 OpenAI API 的 VLM 推理服务（如 [vLLM](https://github.com/vllm-project/vllm) 部署的 Qwen 系列模型）

### 安装依赖

```bash
pip install langgraph openai httpx pyyaml tqdm rich
```

### 启动 VLM 推理服务

使用 `docker/` 下的脚本启动 vLLM 服务。**推荐使用 Qwen3.5-Gemini-Distilled-9B 模型**（效果与效率的最佳平衡）：

```bash
# ✅ 推荐：Qwen3.5-Gemini-Distilled-9B
bash docker/start_Qwen-gemini.sh

# 备选：Qwen3.5 原版
bash docker/start_Qwen3_5.sh
```

### 配置

项目提供多个配置文件，**推荐使用 `config-qwen-gemini.yaml`**：

| 配置文件 | 模型 | 说明 |
|----------|------|------|
| **`config-qwen-gemini.yaml`** | Qwen3.5-Gemini-Distilled-9B | ✅ **推荐配置**，效果与速度最佳平衡 |
| `config.yaml` | Qwen3.5 | 原版大模型配置 |
| `config-qwopus.yaml` | Qwopus | 备选配置 |

推荐配置的核心参数：

```yaml
api:
  base_url: "http://localhost:8576/v1"   # VLM 服务地址
  model_name: "/models/qwen-gemini"      # Qwen3.5-Gemini-Distilled-9B

video:
  fps: 3                                 # 基础采样帧率
  category_fps:                          # 复杂类别使用更高帧率
    "弱势交通参与者异常": 8
    "车辆轨迹与空间冲突": 8
    "极端高危与失控事件": 8

paths:
  video_root: "/path/to/videos"          # 视频根目录
  video_paths: "video_path.txt"          # 待处理视频列表
  output_base: "outputs"                 # 输出目录

pipeline:
  agent_mode: "dag"                      # 推荐使用 DAG 模式
```

### 运行

```bash
# ✅ 推荐：使用 Qwen-Gemini 配置批量处理（DAG 模式）
python main.py --config config-qwen-gemini.yaml

# 单视频测试
python main.py --config config-qwen-gemini.yaml --video path/to/video.mp4 --test

# ReAct 模式 + 实时展示思考过程
python main.py --config config-qwen-gemini.yaml --video path/to/video.mp4 --agent-mode react --show-thinking

# 断点续传（从已有输出目录恢复）
python main.py --resume outputs/20260101_120000
```

### 输出格式

每段视频生成 `result.json`，包含各维度的分析结果：

```json
{
  "自然时间段": { "pred": ["白天"] },
  "气象条件": { "pred": ["晴天"] },
  "车辆轨迹与空间冲突": {
    "pred": ["他车切入或试图切入本车道"],
    "events": [
      { "start_time": 12.0, "end_time": 18.5, "description": "右侧白色SUV强行切入本车道" }
    ]
  }
}
```

## 主要命令行参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（默认 `config.yaml`） |
| `--video` | 单视频路径（用于测试） |
| `--test` | 测试模式（详细日志） |
| `--agent-mode dag\|react` | 选择智能体模式 |
| `--show-thinking` | 实时显示模型思考过程（单视频模式） |
| `--resume <dir>` | 从已有输出目录断点续传 |
| `--set key=value` | 覆盖配置项（如 `--set api.base_url=http://...`） |
| `--dry-run` | 仅打印配置，不实际运行 |

## 技术栈

- **VLM 推理**：Qwen3.5-Gemini-Distilled-9B（推荐）/ Qwen 系列模型（通过 vLLM 部署，兼容 OpenAI API）
- **智能体编排**：[LangGraph](https://github.com/langchain-ai/langgraph)（DAG + ReAct 双模式）
- **目标检测**：YOLO（可选预筛选，加速异常事件定位）
- **视频处理**：ffmpeg / ffprobe
- **异步并发**：Python asyncio + httpx

## License

本项目为 51WORLD 内部项目。
