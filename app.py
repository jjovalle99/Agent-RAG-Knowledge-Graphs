import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_community.graphs import Neo4jGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Neo4jVector
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langserve import add_routes

from src.async_llm_transformer import AsyncLLMGraphTransformer
from src.graph_retriever_utils import Entities, FinalRetriever, structured_retriever

# Load environment variables
load_dotenv()
agent_prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)
# llm_transformer = AsyncLLMGraphTransformer(llm=llm)

# Graph database
graph = Neo4jGraph()
# graph_documents = llm_transformer.aconvert_to_graph_documents(documents=documents)
# graph.add_graph_documents(graph_documents=graph_documents, baseEntityLabel=True, include_source=True)

# Vector-Keyword Retriever
vector_index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

# Graph Retriever
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ]
)
entity_chain = prompt | llm.with_structured_output(Entities)
retriever = FinalRetriever(
    structured_retriever=structured_retriever, vector_index=vector_index, entity_chain=entity_chain, graph=graph
)

# Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

# Create Agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
