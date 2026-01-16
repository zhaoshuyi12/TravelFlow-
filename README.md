✈️ TravelFlow: 基于 LangGraph 的多智能体差旅编排引擎
TravelFlow 是一个模拟真实旅游服务场景（如携程/飞猪）的高级 AI Agent 系统。它不仅实现了多轮对话，更核心的是通过 LangGraph 状态机 解决了复杂业务链路中的意图切换、上下文污染、敏感操作审批等工业级痛点。

🏗️ 核心架构：多智能体协作 (Multi-Agent)
TravelFlow 放弃了脆弱的单体 Prompt 结构，转而采用 “总-分”分层调度架构：

Main Dispatcher (主助手)：负责全局意图识别与权限预检，通过动态路由将请求分发至垂直领域。

Specialized Agents (垂直子助手)：

Flight Agent: 机票搜索、改签、退票业务逻辑。

Hotel Agent: 酒店筛选、预订、订单状态管理。

Excursion Agent: 景点门票销售、旅游路线推荐。

Rental Car Agent: 接送机安排、车辆预订。
