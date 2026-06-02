from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

class AgentState(TypedDict):
    message : str


def greeting_node(state: AgentState) -> AgentState:
    """
    Simple node that adds a greeting message to the state
    """

    state['message'] = "Hey " + state['message'] + ", how are you doing!"

    return state


graph = StateGraph(AgentState)
graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")
app = graph.compile()

# display(Image(app.get_graph().draw_mermaid_png()))
result = app.invoke({"message": "Bob"})

print(result)