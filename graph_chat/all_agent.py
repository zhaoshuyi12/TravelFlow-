from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from graph_chat.my_llm import llm
from tools.car_tools import search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental
from tools.flight_tool import search_flights, update_ticket_to_new_flight, cancel_ticket
from tools.hotels_tools import search_hotels, book_hotel, update_hotel, cancel_hotel
from tools.retriever_vector import lookup_policy
from tools.search_tool import MySearchTool
from tools.trip_tools import search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
memory=InMemorySaver()
research_agent=create_agent(model=llm,tools=[MySearchTool()],system_prompt="你是一个网络搜索的智能体，指令：\n"
                                                           "-仅网络数据获取、网络查询、数据查询相关的任务，不要做任何数学计算"
                                                           "-回复时仅包括工作结果，不要包括任何其他文字",name='research_agent')
update_flight_safe_tools = [search_flights, lookup_policy]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
# 合并所有工具
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools

flight_booking_agent = create_agent(model=llm,tools=update_flight_tools,system_prompt= "您是专门处理航班查询，改签政策查询，改签和预定的智能体(Agent)。\n\n"
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

hotel_booking_agent=create_agent(model=llm,tools=book_hotel_tools,system_prompt=  "您是专门处理酒店查询，酒店预定，酒店订单修改的智能体(Agent)。\n\n"
        "指令：\n"
        "- 在搜索时，请坚持不懈。如果第一次搜索没有结果，请扩大查询范围。\n"
        "- 根据用户的偏好搜索可用酒店，并与客户确认预订详情。\n"
        "- 如果您的工具都不适用或客户改变主意，直接回复，并给出理由。\n"
        "- 如果订酒店，先将酒店名称、价格、评价等信息整理成列表展示给用户。展示完列表后，你必须询问用户：‘以上是为您找到的酒店，请问您想预订哪一家？’，然后结束本次对话\n"
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
car_rental_booking_agent = create_agent(
    model=llm,
    tools=book_car_rental_tools,
    system_prompt=(
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
excursion_booking_agent = create_agent(
    model=llm,
    tools=book_excursion_tools,
    system_prompt=(
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
        "- 绝对禁止：不要自己执行任务，必须调用 transfer_to_* 工具将任务分配给子智能体。\n"
        "- 你没有任何修改的权限，必须将预订请求转交给对应的 Agent 待其完成后再收回控制权。\n"
        "- 绝对禁止：不要一次性调用多个工具或试图并行处理。"
    ),
    # output_mode="last_message",  # 可选：last_message 或 full_history
).compile(checkpointer=memory, name="supervisor")