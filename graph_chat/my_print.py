from typing import List, Sequence

from langchain_core.messages import convert_to_messages, BaseMessage


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    # 1. 核心修复：增加空值和非字典类型的检查
    if update is None:
        print("🔍 Debug: update is None, skipping...")
        return

    if not isinstance(update, (dict, tuple)):
        print(f"🔍 Debug: Unexpected update type {type(update)}, value: {update}")
        return

    print("🔍 Debug: messages_data =", update)
    is_subgraph = False

    # 处理子图逻辑 (保持原样)
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    # 2. 再次确保 update 是字典再进行 .items()
    if not isinstance(update, dict):
        return

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        # 3. 如果 node_update 是空（None 或 {}），安全跳过
        if not node_update:
            print("No updates in this node.\n")
            continue

        # 兼容处理：检查是否存在 'messages' 键
        if isinstance(node_update, dict) and 'messages' in node_update:
            messages = convert_to_messages(node_update["messages"])
        elif isinstance(node_update, Sequence):  # 处理直接返回列表的情况
            messages = convert_to_messages(node_update)
        else:
            print(node_update)
            print("--------------\n")
            continue

        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")