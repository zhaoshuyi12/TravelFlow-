# 🧭 TravelFlow - 一站式智能差旅助手

> 基于 LangGraph 构建的多 Agent 差旅管理系统，支持航班查询、酒店预订、行程规划等功能。通过智能 Agent 协作与人工审核机制，实现安全可靠的差旅服务自动化。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.x-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![SQLite](https://img.shields.io/badge/SQLite-DB-green.svg)](https://www.sqlite.org/)

---

## ✨ 核心特性

### 🤖 多 Agent 协作架构
- **Supervisor Agent**：负责任务分发与协调
- **专业 Agent**：航班预订、酒店预订、汽车租赁、旅行推荐、网络搜索
- **智能流转**：通过 Handoff Tool 实现 Agent 间无缝协作

### 🔒 人工审核机制
- 在网络搜索等潜在风险操作前引入人工确认
- 用户可选择批准（y）或直接提供答案
- 实现安全可控的工具调用

### 📊 SQLite 数据管理
- 基于 `travel_new.sqlite` 存储航班、酒店、租车等核心数据
- 支持数据时间同步，确保测试数据时效性
- 事务安全，保证数据一致性

### 🧠 政策 RAG 系统
- 基于 BGE-Chinese 向量模型构建 FAQ 检索
- 支持航空退改政策等结构化查询
- 提升 Agent 决策准确性

### 💾 状态管理与持久化
- 基于 `MessagesState` + `MemorySaver` 实现会话状态追踪
- 支持中断恢复与历史回溯
- 保证多轮对话的上下文连续性

```bash
git clone https://github.com/zhaoshuyi12/TravelFlow-.git
cd TravelFlow-
