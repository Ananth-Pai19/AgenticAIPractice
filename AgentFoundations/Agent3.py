from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, message, START, END
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This functions adds two integers together"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This function subtracts the first number from the second"""
    return b - a

@tool
def multiply(a: int, b: int) -> int:
    """This function multiplies two numbers"""
    return a * b

@tool
def num_digits(num: int) -> int:
    """This function counts the number of digits"""
    count = 0
    if num != 0:
        temp = num
        while temp:
            temp = int(temp / 10)
            count += 1
    else:
        return 1
    return count

# Create the ToolNode for our tools
tools = [add, subtract, multiply, num_digits]
tool_node = ToolNode(tools=tools)

llm = ChatOllama(model="qwen2.5:7b", temperature=0).bind_tools(tools)

# Now we make the various nodes of the graph
def agent_call(state: AgentState) -> AgentState:
    """This function calls the agent"""
    system_message = SystemMessage(content="""You are an AI assistant. When solving multi-step problems:
    - If you need to compute something, call the appropriate tool.
    - Only respond with plain text when there is no tool that satisfies the request.""")
    response = llm.invoke([system_message] + state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("model",agent_call)
graph.add_edge(START, "model")
graph.add_node("tools", tool_node)
graph.add_edge("tools", "model")
graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result with 2. Then tell me how many digits the final answer has.")]}
print_stream(app.stream(inputs, stream_mode="values"))