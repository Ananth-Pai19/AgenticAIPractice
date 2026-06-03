from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from typing import Annotated, Sequence, TypedDict

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"The document has been updated successfully! The current content of the document is:\n{document_content}"


@tool
def save(filename: str):
    """
    Saves the current document to a text file and exits the process

    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith("txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print("Document has successfully been saved!")
        return f"Document has successfully been saved to '{filename}'."

    except Exception as e:
        return f"Exception occured while saving to file: {str(e)}"
    
tools = [update, save]

model = ChatOllama(model='qwen2.5:7b', temperature=0).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """"""
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant, who is going to help the user in modifying, updating and drafting various documents.
                                  
    - If the user wants to modify or update the content of the document currently being drafted, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish drafting the document, use the 'save' tool, by providing an apt text file name, if not provided by the user.
    - Make sure to always show the current document state after modifications.

    The current document content is: {document_content}    
    """)

    if not state['messages']:
        user_input = "I'm ready to help you update your document, what would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"USER said: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state['messages']) + [user_message, response]}


def should_continue(state: AgentState):
    """Determine if we should continue or end the conversation"""

    messages = state['messages']

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"

    
    return "continue"


def print_messages(messages):
    """Function to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("agent", "tool_node")

graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n-------------- DRAFTER --------------")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step['messages'])

    print("\n---------- DRAFTER FINISHED ------------")


if __name__ == "__main__":
    run_document_agent()
