from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from random import randint


class AgentState(TypedDict):
    player_name: str
    guesses: List[int]
    attempts: int
    lower_bound: int
    upper_bound: int

number_to_guess = randint(1, 20)
max_guesses = 7

def setup_node(state: AgentState) -> AgentState:
    """Function to setup the game"""

    state['guesses'] = []
    state['attempts'] = 0
    state['lower_bound'] = 0
    state['upper_bound'] = 20
    return state


def guessing_node(state: AgentState) -> AgentState:
    """Function to perform a guess of the random number"""
    guess = randint(state['lower_bound'], state['upper_bound']) 
    print(f"GUESS: {guess}")
    state['guesses'].append(guess)
    state['attempts'] += 1
    return state

def hint_node(state: AgentState) -> AgentState:
    """Function to verify the guess"""

    current_guess = state['guesses'][-1]
    if current_guess == number_to_guess:
        print(f"GUESSED THE NUMBER: {current_guess}")

    elif current_guess < number_to_guess:
        print("HIGHER")
        state['lower_bound'] = current_guess + 1
    
    else:
        print("LOWER")
        state['upper_bound'] = current_guess - 1

    return state


def continue_guessing(state: AgentState):
    """Function to check whether to continue guessing or not"""

    if state["guesses"][-1] == number_to_guess:
        return "exit"
    elif state['attempts'] == max_guesses:
        return "exit"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("setup_node", setup_node)
graph.add_node("guess_node", guessing_node)
graph.add_node("hint_node", hint_node)

graph.add_edge(START, "setup_node")
graph.add_edge("setup_node", "guess_node")
graph.add_edge("guess_node", "hint_node")
graph.add_conditional_edges(
    "hint_node",
    continue_guessing,
    {
        "continue": "guess_node",
        "exit": END
    }
)

app = graph.compile()
result = app.invoke(AgentState(player_name="Ananth", guesses=[], attempts=0, lower_bound=0, upper_bound=20))