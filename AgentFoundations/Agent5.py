from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add as add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
pdf_path = "/home/ananth/AgenticAIPractice/ERC_RULES.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try: 
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} number of pages.")
except Exception as e:
    print(f"Exception occured while loading the PDF: {str(e)}") 
    raise e

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=1000
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"/home/ananth/AgenticAIPractice/Database"
collection_name = "erc_rules"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"Error while setting up ChromaDB: {str(e)}")
    raise e

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tools searches and retrieves the information from the ERC_RULES.pdf document"""

    docs = retriever.invoke(query)
    if not docs:
        return "I have not found any relevant information in the document ERC_RULES.pdf."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Checks if the last message contains tool calls"""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
                   

system_prompt = '''
You are an intelligent AI assistant that assists the users by answering their queries.
Give references and cite the section and page in the document from where you got the information from. For every query, after answering the query, give a some more insights about that section
so as to give a more complete and comprehensive answer to queries. Answer the queries keeping in mind that this is competition related and there are competition points on the line.
'''

tools_dict = {the_tool.name : the_tool for the_tool in tools}

# LLM Agent Node
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current agent state"""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages": [messages]}

def retriever_agent(state: AgentState) -> AgentState:
    """Execute tool calls from LLM's response. Retrieves information from the document via tool call"""
    tool_calls = state['messages'][-1].tool_calls   #type: ignore
    results = []
    for tool_call in tool_calls:
        print(f"Calling Tool: {tool_call['name']} with query: {tool_call['args'].get('query', 'No query provided')}")

        if not tool_call['name'] in tools_dict:
            print(f"\nTool {tool_call['name']} does not exist")
            result = "Incorrect tool name, please retry and select tool from the list of available tools"

        else:
            result = tools_dict[tool_call['name']].invoke(tool_call['args'])
            print(f"Result Length: {len(str(result))}")


        results.append(ToolMessage(tool_call_id=tool_call['id'], name=tool_call['name'], content=str(result)))

    print("Tool Execution completed")
    return {'messages': results}


# Make the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever", retriever_agent)
graph.set_entry_point("llm")
graph.add_edge("retriever", "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever",
        False: END
    }
)

rag_agent = graph.compile()

def main():
    print("\n---------- RAG AGENT -------------")
    while True:
        user_input = input("\nEnter your query: ")
        if user_input in ["exit", "quit", "close"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke(AgentState(messages=messages))
        print("\n------------ ANSWER -----------")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    main()