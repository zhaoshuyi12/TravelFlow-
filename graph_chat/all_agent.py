from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

from graph_chat.my_llm import llm
from tools.car_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from tools.flight_tool import search_flights, update_ticket_to_new_flight, cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.retriever_vector import lookup_policy
from tools.search_tool import MySearchTool
from tools.trip_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
memory=InMemorySaver()
research_agent=create_react_agent(model=llm,tools=[MySearchTool()],prompt="你是一个网络搜索的智能体，指令：\n"
                                                           "-仅网络数据获取、网络查询、数据查询相关的任务，不要做任何数学计算"
                                                           "-回复时仅包括工作结果，不要包括任何其他文字",name='research_agent')
update_flight_safe_tools = [search_flights, lookup_policy]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
# 合并所有工具
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools

flight_booking_agent = create_react_agent(model=llm,tools=update_flight_tools,prompt= "您是专门处理航班查询，改签政策查询，改签和预定的智能体(Agent)。\n\n"
        "指令：\n"
        "- 在搜索航班时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。\n"
        "- 如果您的工具都不适用或客户改变主意，直接回复，并给出理由。\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字",
                                          checkpointer=memory,
                                          name="flight_booking_agent",
                                          )
# 定义安全工具（只读操作）和敏感工具（涉及更改的操作）
book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
#合并所有工具
book_hotel_tools=book_hotel_safe_tools + book_hotel_sensitive_tools

hotel_booking_agent=create_react_agent(model=llm,tools=book_hotel_tools,prompt=  "您是专门处理酒店查询，酒店预定，酒店订单修改的智能体(Agent)。\n\n"
        "指令：\n"
        "- 在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。\n"
        "- 根据用户的偏好搜索可用酒店，并与客户确认预订详情。\n"
        "- 如果您的工具都不适用或客户改变主意，直接回复，并给出理由。\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字",checkpointer=memory,name="hotel_booking_agent")

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
]

# 合并所有工具
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
# 汽车租赁处理的 子智能体
car_rental_booking_agent = create_react_agent(
    model=llm,
    tools=book_car_rental_tools,
    prompt=(
        "您是专门处理汽车租赁查询，汽车租赁预定，汽车租赁订单修改的智能体(Agent)。\n\n"
        "指令：\n"
        "- 在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。\n"
        "- 根据用户的偏好搜索可用租车，并与客户确认预订详情。\n"
        "- 如果您的工具都不适用或客户改变主意，直接回复，并给出理由。\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字"
    ),
    checkpointer=memory,
    name="car_rental_booking_agent",)

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]

# 合并所有工具
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
# 旅行推荐处理的 子智能体
excursion_booking_agent = create_react_agent(
    model=llm,
    tools=book_excursion_tools,
    prompt=(
        "您是专门处理旅行推荐查询，旅行产品预定，旅行订单修改的智能体(Agent)。\n\n"
        "指令：\n"
        "- 在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。\n"
        "- 根据用户的偏好搜索可行的旅行推荐，并与客户确认预订详情。\n"
        "- 如果您的工具都不适用或客户改变主意，直接回复，并给出理由。\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字"
    ),
    checkpointer=memory,
    name="excursion_booking_agent",
)

#监督者 也是一个智能体
def create_handoff_tool(*, agent_name: str,task_type:str, description: str | None = None):
    """
        创建一个用于将当前会话转接到指定代理的工具函数。

        该函数返回一个装饰器包装的工具函数，当调用时会生成一个工具消息并返回转接命令，
        指示流程控制器将控制权转移给指定的代理。

        参数:
            agent_name (str): 目标代理的名称，用于标识要转接的代理
            description (str | None): 工具的描述信息，如果未提供则使用默认描述

        返回:
            handoff_tool: 一个装饰器包装的工具函数，用于执行转接操作
        """
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."


    @tool(name, description=description)
    def handoff_tool(state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],)->Command:
        """
             执行实际的转接操作。

             创建一个工具消息表明转接成功，并返回一个命令对象指示流程控制器
             将控制权转移给指定代理，同时更新会话状态。

             参数:
                 state (MessagesState): 当前会话状态，包含消息历史等信息
                 tool_call_id (str): 工具调用的唯一标识符

             返回:
                 Command: 包含转接指令和状态更新的命令对象

             """
        pending_tasks = getattr(state, "_pending_tasks", [])
        completed_tasks = getattr(state, "_completed_tasks", set())
        # 首次调用：解析用户原始请求中的所有任务
        if not pending_tasks and state["messages"]:
            user_msg = next((m for m in state["messages"] if hasattr(m, 'role') and m.role == 'user'), None)
            if user_msg:
                content = user_msg.content.lower()
                new_pending = []
                if any(kw in content for kw in ["酒店", "住宿", "hotel"]):
                    new_pending.append("hotel")
                if any(kw in content for kw in ["旅行攻略", "景点", "推荐", "excursion", "tour"]):
                    new_pending.append("excursion")
                if any(kw in content for kw in ["航班", "机票", "flight"]):
                    new_pending.append("flight")
                if any(kw in content for kw in ["租车", "car"]):
                    new_pending.append("car_rental")
                if any(kw in content for kw in ["搜索", "查", "research"]):
                    new_pending.append("research")
                pending_tasks = new_pending
        pending_tasks = [t for t in pending_tasks if t not in completed_tasks]
        tool_message={
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        if pending_tasks and pending_tasks[0] != task_type:
            return Command(update=state)
        return Command(graph=Command.PARENT,goto=agent_name,
                       update={**state,"messages": state["messages"] +  [tool_message]})

    return handoff_tool

assign_to_research_agent=create_handoff_tool(task_type='research',agent_name='research_agent',description="将任务分配给：research_agent智能体。")
assign_to_flight_booking_agent = create_handoff_tool(
    task_type='flight',
    agent_name="flight_booking_agent",
    description="将任务分配给：flight_booking_agent智能体。",
)
assign_to_hotel_booking_agent = create_handoff_tool(
    task_type='hotel',
    agent_name="hotel_booking_agent",
    description="将任务分配给：hotel_booking_agent智能体。",
)
assign_to_car_rental_booking_agent = create_handoff_tool(
    task_type='car_rental',
    agent_name="car_rental_booking_agent",
    description="将任务分配给：car_rental_booking_agent智能体。",
)
assign_to_excursion_booking_agent = create_handoff_tool(
    agent_name="excursion_booking_agent",
    task_type='excursion',
    description="将任务分配给：excursion_booking_agent智能体。",
)
#创建主智能体
supervisor_agent = create_react_agent(
    model=llm,
    tools=[assign_to_research_agent, assign_to_flight_booking_agent, assign_to_hotel_booking_agent, assign_to_car_rental_booking_agent, assign_to_excursion_booking_agent],
    prompt=(
        "你是一个监督者或者管理者，管理五个智能体：\n"
        "- 网络搜索智能体：分配与网络搜索、数据查询相关的任务\n"
        "- 航班预订能体：分配与航班查询，预定，改签等相关的任务\n"
        "- 酒店预订智能体：分配与酒店查询，预定，修改订单等相关的任务\n"
        "- 汽车租赁预定智能体：分配与汽车租赁查询，预定，修改订单等相关的任务\n"
        "- 旅行产品预定智能体：分配与旅行推荐查询，预定，修改订单等相关的任务\n"
        "### 核心处理逻辑（重要）\n"
        "1. **意图识别**：仔细分析用户输入，识别出所有需要完成的独立任务（例如：用户说'订酒店和查攻略'，则识别出'订酒店'和'查攻略'两个任务）。\n"
        "2. **顺序执行**：你必须**按顺序**处理这些任务。不要试图同时做两件事。\n"
        "3. **单步分配**：每次只将**当前**需要处理的任务分配给对应的智能体。\n"
        "4. **状态保持**：在第一个任务（如订酒店）完成后，你必须**自动继续**分配第二个任务（如做攻略），直到所有任务都完成。\n"
        "5. **最终回复**：当所有任务都完成后，向用户总结结果。\n"
        "### 约束\n"
        "- 绝对禁止：不要自己执行任务，必须调用工具。\n"
        "- 绝对禁止：不要一次性调用多个工具或试图并行处理。\n"
    ),
    # checkpointer=memory,
    name="supervisor",
)