import operator
import uuid
from typing import Annotated, Set, List

from langchain_core.messages import HumanMessage
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command

# 1. 确保从 all_agent 导入唯一的 memory 实例
from graph_chat.all_agent import (
    supervisor_agent,
    research_agent,
    flight_booking_agent,
    hotel_booking_agent,
    car_rental_booking_agent,
    excursion_booking_agent,
    memory
)
from graph_chat.draw_png import draw_graph
from graph_chat.fetch_user_info_node import get_user_info
from graph_chat.my_print import pretty_print_messages
from tools.init_db import update_dates

class MyGraphState(MessagesState):
    # 使用 operator.or_ 确保集合是进行并集更新（即：添加新标记）
    _completed_tasks: Annotated[Set[str], operator.or_]
    # 使用 operator.add 确保列表是追加更新
    _pending_tasks: Annotated[List[str], operator.add]


def wrap_agent_with_completion(agent, task_type: str):
    """包装 agent，使其返回时标记任务完成"""

    def wrapper(state: MyGraphState):
        # 1. 正常执行 agent
        result = agent.invoke(state)

        # 2. 这里的 result 可能是消息字典，也可能是消息对象
        if not isinstance(result, dict):
            result = {"messages": [result]}

        # 3. 关键：只返回当前节点完成的 task_type
        # LangGraph 会自动执行: state["_completed_tasks"] | {task_type}
        return {
            **result,
            "_completed_tasks": {task_type},  # 返回一个 set
        }

    return wrapper
# --- 图定义 ---

graph = (
    StateGraph(MyGraphState)
    # 获取用户信息节点
    .add_node('fetch_user_info', get_user_info)

    # 2. 节点名称必须与 all_agent.py 中的定义严格对齐
    .add_node("supervisor", supervisor_agent)
    .add_node("research_agent", wrap_agent_with_completion(research_agent, "research"))
    .add_node("flight_booking_agent", wrap_agent_with_completion(flight_booking_agent, "flight"))
    .add_node("hotel_booking_agent", wrap_agent_with_completion(hotel_booking_agent, "hotel"))
    .add_node("excursion_booking_agent", wrap_agent_with_completion(excursion_booking_agent, "excursion"))
    .add_node("car_rental_booking_agent", wrap_agent_with_completion(car_rental_booking_agent, "car_rental"))

    # --- 连线逻辑 ---
    .add_edge(START, 'fetch_user_info')
    .add_edge('fetch_user_info', 'supervisor')
    .add_edge("research_agent", "supervisor")
    .add_edge("flight_booking_agent", "supervisor")
    .add_edge("hotel_booking_agent", "supervisor")
    .add_edge("excursion_booking_agent", "supervisor")
    .add_edge("car_rental_booking_agent", "supervisor")
    # 3. 使用共享的 memory
    .compile(checkpointer=memory)
)

# --- 运行配置 ---

import time
session_id =str(uuid.uuid4())# 强制每一轮测试都是全新的历史记录
update_dates()  # 重置数据库

config = {
    "configurable": {
        "passenger_id": "3442 587242",
        "thread_id": session_id,
    }
}


def execute_graph(user_input: str):
    """ 执行工作流的函数 """
    current_state = graph.get_state(config)
    history_messages = current_state.values.get("messages", [])
    print(f"\n{'=' * 20} 污染数据扫描仪 {'=' * 20}")
    print(f"当前 Thread ID: {config['configurable']['thread_id']}")
    print(f"历史消息总数: {len(history_messages)}")
    found_pollution = False
    for i, msg in enumerate(history_messages):
        # 核心逻辑：检查 content 的类型
        content_type = type(msg.content)

    if current_state.next:  # 处理中断/等待用户输入的情况
        input_data = Command(resume=user_input)
    else:
        # 修复点：不要用 ('user', user_input)，显式使用 HumanMessage 对象列表
        input_data = {'messages': [HumanMessage(content=user_input)]}

    # 为了更好的调试体验，建议捕获 chunk 里的 updates
    for chunk in graph.stream(input_data, config, stream_mode="updates"):
        for node_name, output in chunk.items():
            print(f"\n--- [节点: {node_name}] ---")
            # 调用你原有的打印工具
            pretty_print_messages(output, last_message=True)
draw_graph(graph,'graph_supervisor.png')

# --- 主循环 ---

if __name__ == "__main__":
    draw_graph(graph,'graph_supervisor.png')
    while True:
        try:
            user_msg = input('\n用户：').strip()

            execute_graph(user_msg)

        except KeyboardInterrupt:
            break
        except Exception as e:
            import  traceback
            traceback.print_exc()
