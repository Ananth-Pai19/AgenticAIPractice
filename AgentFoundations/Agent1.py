from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# State definition
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# Node
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

app = graph.compile()

# Run
result = app.invoke({"messages": [("human", "What is 2+2?")]})
print(result["messages"][-1].content)