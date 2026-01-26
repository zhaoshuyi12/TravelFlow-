import operator
import uuid
from typing import Annotated, Set, List

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from langgraph_supervisor import create_supervisor

# 从 all_agent 导入子智能体和配置（注意：all_agent 中不应再包含手动构建的 supervisor 逻辑）
from graph_chat.all_agent import (
    research_agent,
    flight_booking_agent,
    hotel_booking_agent,
    car_rental_booking_agent,
    excursion_booking_agent,
    llm,
    memory
)
from graph_chat.draw_png import draw_graph
from graph_chat.fetch_user_info_node import get_user_info
from graph_chat.my_print import pretty_print_messages
from tools.init_db import update_dates




# ==========================================
# 使用 create_supervisor 创建监督者（包含所有子智能体和 handoff 工具）
# ==========================================

supervisor = create_supervisor(
    agents=[
        research_agent,
        flight_booking_agent,
        hotel_booking_agent,
        car_rental_booking_agent,
        excursion_booking_agent
    ],
    model=llm,
    prompt=(
        "你是一个监督者或者管理者，管理五个智能体：\n"
        "- research_agent：分配与网络搜索、数据查询相关的任务\n"
        "- flight_booking_agent：分配与航班查询，预定，改签等相关的任务\n"
        "- hotel_booking_agent：分配与酒店查询，预定，修改订单等相关的任务\n"
        "- car_rental_booking_agent：分配与汽车租赁查询，预定，修改订单等相关的任务\n"
        "- excursion_booking_agent：分配与旅行推荐查询，预定，修改订单等相关的任务\n\n"
        "### 核心处理逻辑（重要）\n"
        "1. **意图识别**：仔细分析用户输入，识别出所有需要完成的独立任务（例如：用户说'订酒店和查攻略'，则识别出'订酒店'和'查攻略'两个任务）。\n"
        "2. **顺序执行**：你必须**按顺序**处理这些任务。不要试图同时做两件事。\n"
        "3. **单步分配**：每次只将**当前**需要处理的任务分配给对应的智能体。"
        "4. **状态保持**：在第一个任务完成后，子智能体会返回结果给你。你必须检查是否还有其他待处理的任务，"
        "如果有，继续分配下一个任务，直到所有任务都完成。\n"
        "5. **最终回复**：当所有任务都完成后，向用户总结所有结果。\n\n"
        "指令：\n"
        "- 当用户提出新需求时，先调用对应智能体查询信息。\n"
        "- **严禁代用户选择具体的服务（如具体的酒店或租车公司）**。\n"
        "- 当智能体返回多个选项时，你必须将选项呈现给用户，并询问用户的决定。\n"
        "- 只有当用户明确表达'预订某某'或提供具体偏好后，才能下达预订指令。\n\n"
        "### 约束\n"
        "- 你没有任何修改的权限，必须将预订请求转交给对应的 Agent 待其完成后再收回控制权。\n"
        "- 绝对禁止：不要一次性调用多个工具或试图并行处理。"
    ),supervisor_name="supervisor",
    # output_mode="last_message",  # 可选：控制输出格式
)

# 编译 supervisor（此时它内部已经包含所有子智能体和路由逻辑）
supervisor_app = supervisor.compile()

def should_fetch_user_info(state: MessagesState) -> str:
    """判断是否需要获取用户信息"""
    messages = state.get('messages', [])
    print('+++++++++++++++')
    print(messages)
    
    # 检查是否已有用户信息（如航班信息等）
    for msg in messages:
        if isinstance(msg, AIMessage):
            if getattr(msg, 'name', None) == "user_info_success":
                print("🔍 已有用户信息，跳过获取")
                return 'skip_fetch'
            elif getattr(msg, 'name', None) == "user_info_reset":
                print("🔄 需要重新获取用户信息")
                return "fetch_user_info"
    
    # 如果没有历史用户信息，则检查用户输入是否需要获取用户信息
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            user_input = last_message.content.lower()
            if any(keyword in user_input for keyword in ['我的', '个人信息', '用户信息', '航班', '航班信息', '机票', '目的地', '旅客']):
                print("🔍 检测到用户需要获取信息，准备获取用户信息")
                return "fetch_user_info"
    else:
        # 如果没有任何消息，也需要获取用户信息
        return "fetch_user_info"
    
    print("🔍 不需要获取用户信息，直接进入 supervisor")
    return 'skip_fetch'

graph = (
    StateGraph(MessagesState)
    # 获取用户信息节点（在 supervisor 之前执行）
    .add_node('fetch_user_info', get_user_info)

    # 将 supervisor 作为子图节点（包含所有子智能体和 handoff 逻辑）
    .add_node('supervisor_team', supervisor_app)

    .add_conditional_edges(
        START,
        should_fetch_user_info,
        {
            'fetch_user_info': 'fetch_user_info',
            'skip_fetch': 'supervisor_team'
        }
    )
    .add_edge('fetch_user_info', 'supervisor_team')  # 如果获取了信息，就进入 supervisor

    .compile(checkpointer=memory)
)
session_id = str(uuid.uuid4())
update_dates()

config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": session_id,
    }
}


def execute_graph(user_input: str):
    """执行工作流的函数"""
    current_state = graph.get_state(config)
    history_messages = current_state.values.get("messages", [])
    print(current_state)
    print(f"\n{'=' * 20} 会话信息 {'=' * 20}")
    print(f"当前 Thread ID: {config['configurable']['thread_id']}")
    print(f"历史消息总数: {len(history_messages)}")
    print(f"当前状态: {current_state.next}")
    print('input', user_input)
    if current_state.next:  # 处理中断/等待用户输入的情况
        input_data = Command(resume=user_input)
    else:
        input_data = {'messages': [HumanMessage(content=user_input)]}

    # 流式输出
    for chunk in graph.stream(input_data, config, stream_mode="updates"):
        if chunk is None:
            continue
        for node_name, output in chunk.items():
            print(f"\n--- [节点: {node_name}] ---")
            pretty_print_messages(output, last_message=True)


if __name__ == "__main__":
    draw_graph(graph, 'graph_supervisor.png')
    while True:
        try:
            user_msg = input('\n用户：').strip()
            if not user_msg:
                continue

            execute_graph(user_msg)
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            import traceback

            traceback.print_exc()