from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from tools.flight_tool import fetch_user_flight_information

def format_flight_info(flight_data):
    flight=flight_data[0]
    return (
        f"已查询到您的航班信息：\n"
        f"- 机票号：{flight['ticket_no']}\n"
        f"- 预订编号：{flight['book_ref']}\n"
        f"- 航班号：{flight['flight_no']}（{flight['flight_id']}）\n"
        f"- 出发机场：{flight['departure_airport']}\n"
        f"- 到达机场：{flight['arrival_airport']}\n"
        f"- 计划起飞时间：{flight['scheduled_departure']}\n"
        f"- 计划到达时间：{flight['scheduled_arrival']}\n"
        f"- 座位号：{flight['seat_no']}\n"
        f"- 舱位条件：{flight['fare_conditions']}"
    )
#节点函数,获取用户信息
def get_user_info(state:MessagesState,config:RunnableConfig):
    if 'messages' in state:

        for message in state['messages']:
            "如果已经查到了航班信息了几句直接返回"
            if isinstance(message,AIMessage) and message.id=="user_info_success":
                return

    flight_data=fetch_user_flight_information(config)
    if flight_data:
        flight_message=AIMessage(
            content=format_flight_info(flight_data),
            additional_kwargs={},
            id='user_info_success',

        )
    else:
        flight_message = AIMessage(
            content="未找到航班信息请检查旅客信息时候正确",
            additional_kwargs={},
            id='user_info_fail',

        )

    return {'messages':[flight_message]}