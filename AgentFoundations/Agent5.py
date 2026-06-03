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
embeddings = OllamaEmbeddings(model="text-embedding-3-small")
pdf_path = "/home/ananth/Downloads/ERC_RULES.pdf"

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
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"\home\ananth\AgenticAIPractice"
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
    return hasattr(result, 'tool_call') and len(result.tool_calls) > 0
                   

system_prompt = '''
Lite bro
'''
