import uuid
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

# --- 图定义 ---

graph = (
    StateGraph(MessagesState)
    # 获取用户信息节点
    .add_node('fetch_user_info', get_user_info)

    # 2. 节点名称必须与 all_agent.py 中的定义严格对齐
    .add_node("supervisor", supervisor_agent)
    .add_node("research_agent", research_agent)
    .add_node("flight_booking_agent", flight_booking_agent)
    .add_node("hotel_booking_agent", hotel_booking_agent)
    .add_node("excursion_booking_agent", excursion_booking_agent)
    .add_node("car_rental_booking_agent", car_rental_booking_agent)

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

    if current_state.next:  # 处理中断/等待用户输入的情况
        input_data = Command(resume={'answer': user_input})
    else:
        # 修复点：不要用 ('user', user_input)，显式使用 HumanMessage 对象列表
        input_data = {'messages': [HumanMessage(content=user_input)]}

    # 为了更好的调试体验，建议捕获 chunk 里的 updates
    for chunk in graph.stream(input_data, config, stream_mode="updates"):
        for node_name, output in chunk.items():
            print(f"\n--- [节点: {node_name}] ---")
            # 调用你原有的打印工具
            pretty_print_messages(output, last_message=True)


# --- 主循环 ---

if __name__ == "__main__":
    draw_graph(graph,'graph_supervisor.png')
    while True:
        try:
            user_msg = input('\n用户：').strip()
            if not user_msg or user_msg.lower() in ['exit', 'quit']:
                break

            execute_graph(user_msg)

        except KeyboardInterrupt:
            break
        except Exception as e:
            import  traceback
            traceback.print_exc()
