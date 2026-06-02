from langgraph.graph import StateGraph
from typing import TypedDict, Dict

class AgentState(TypedDict):
    message : str


def glazer_node(state: AgentState) -> AgentState:
    """
    Node that compliments the user 
    """
    state['message'] = state['message'] + ", you are doing an amazing job in learning LangGraph!"
    return state

graph = StateGraph(AgentState)
graph.add_node("glazer", glazer_node)
graph.set_entry_point("glazer")
graph.set_finish_point("glazer")
app = graph.compile()

result = app.invoke({"message": "Bob"})

print(result['message'])