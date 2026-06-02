from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process_node", process)
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

agent = graph.compile()
prompt = input("Enter: ")
conversation = []
while prompt != "exit":
    conversation.append(HumanMessage(content=prompt))
    response = agent.invoke(AgentState(messages=conversation))
    print(response['messages'][-1].content)
    conversation = response['messages']
    print(len(conversation))
    prompt = input("Enter: ")

with open("Agent2Conversation.txt", "w") as f:
    f.write("Agent 2 Pravachan: \n")
    for message in conversation:
        if isinstance(message, HumanMessage):
            f.write(f"User uvaca: {message.content}\n")
        else:
            f.write(f"AI uvaca: {message.content}\n")
    f.write("----------Samaapth----------")


print("Conversation stored to Agent2Conversation.txt")