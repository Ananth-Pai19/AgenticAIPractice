from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState(TypedDict):
    number1: int
    number2: int
    number3: int
    number4: int
    operation: str
    operation2: str
    finalNumber: int
    finalNumber2: int

def add_node(state: AgentState) -> AgentState:
    """This node adds number1 and number2"""
    state['finalNumber'] = state['number1'] + state['number2']
    return state


def subtract_node(state: AgentState) -> AgentState:
    """This node subtracts number1 and number2"""
    state['finalNumber'] = state['number1'] - state['number2']
    return state

def add_node2(state: AgentState) -> AgentState:
    """This node adds number3 and number4"""
    state['finalNumber2'] = state['number3'] + state['number4']
    return state

def subtract_node2(state: AgentState) -> AgentState:
    """This node subtracts number3 and number4"""
    state['finalNumber2'] = state['number3'] - state['number4']
    return state

def choose_first_node(state: AgentState):
    """This node chooses the first node to process"""

    if state['operation'] == '+':
        return "addition_operation"
    elif state['operation'] == '-':
        return "subtraction_operation"
    
def choose_second_node(state: AgentState):
    """This node chooses the second node to process"""

    if state['operation2'] == '+':
        return "addition_operation2"
    elif state['operation2'] == '-':
        return "subtraction_operation2"
    

graph = StateGraph(AgentState)
graph.add_node("add_node", add_node)
graph.add_node("subtract_node", subtract_node)
graph.add_node("add_node2", add_node2)
graph.add_node("subtract_node2", subtract_node2)
graph.add_node("router", lambda state:state)
graph.add_node("router2", lambda state:state)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    choose_first_node,
    {
        "addition_operation": "add_node",
        "subtraction_operation": "subtract_node"
    }
)
graph.add_edge("add_node", "router2")
graph.add_edge("subtract_node", "router2")
graph.add_conditional_edges(
    "router2",
    choose_second_node,
    {
        "addition_operation2": "add_node2",
        "subtraction_operation2": "subtract_node2" 
    }
)
graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)

app = graph.compile()
result = app.invoke(
    AgentState(number1=10, operation="-", number2=5, number3=7, number4=2, operation2="+", finalNumber=0, finalNumber2=0)
)
print(f"Results are: finalNumber = {result['finalNumber']} and finalNumber2 = {result['finalNumber2']}")