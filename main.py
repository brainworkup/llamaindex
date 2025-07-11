# Import all required modules at the top
import os
import pathlib
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Configure the references directory
references_dir = os.getenv("REFERENCES_DIR", "./references")

# Create the references directory if it doesn't exist
try:
    pathlib.Path(references_dir).mkdir(exist_ok=True)
    print(f"Using references directory: {references_dir}")
except Exception as e:
    print(f"Error creating directory {references_dir}: {e}")

# LlamaIndex imports - wrapped in try-except to handle missing dependencies
try:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Retrieve HF_TOKEN from the environment variables
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
        exit(1)

    # Initialize the LLM
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.7,
        max_tokens=512,  # Increased max tokens for longer responses
        token=hf_token,
    )

    # Test the LLM
    print("Testing LLM connection...")
    response = llm.complete("Hello, how are you?")
    print(response)

    # Initialize the document reader and load documents
    print(f"Loading documents from {references_dir}...")
    reader = SimpleDirectoryReader(input_dir=references_dir)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")

    # Initialize the embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Create nodes from documents using a sentence splitter
    print("Processing documents and creating index...")
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True,
    )

    # Create a query engine
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    # Example queries
    print("\n--- Example Query Results ---")

    example_queries = [
        "Summarize what intolerance of uncertainty is.",
        "What is the relationship between uncertainty and mental health?",
    ]

    for query in example_queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")

    # Interactive mode
    print("\n--- Enter your queries (type 'exit' to quit) ---")
    while True:
        user_query = input("\nEnter a query: ")
        if user_query.lower() == "exit":
            break

        response = query_engine.query(user_query)
        print(f"Response: {response}")

except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print(
        "Please install required packages with: pip install llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface"
    )
except Exception as e:
    print(f"Error occurred: {e}")

    # Evaluation and observability
    from llama_index.core.evaluation import FaithfulnessEvaluator

    # Evaluation and observability
    from llama_index.core.evaluation import FaithfulnessEvaluator

    # Use the previously defined query_engine and llm
    # Evaluate the faithfulness of responses
    print("\n--- Evaluating Response Faithfulness ---")
    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_query = "What information can you provide based on the documents?"
    print(f"\nEvaluation Query: {eval_query}")
    response = query_engine.query(eval_query)
    print(f"Response: {response}")
    eval_result = evaluator.evaluate_response(response=response)
    print(f"Evaluation Result - Passing: {eval_result.passing}")
    print(f"Evaluation Score: {eval_result.score}")


# IngestionPipeline
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import asyncio

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)


# Define an async function to run the pipeline
async def run_pipeline():
    nodes = await pipeline.arun(documents=[Document.example()])
    return nodes


# Use this to run the async function if needed
# nodes = asyncio.run(run_pipeline())

# Vector storage

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

# VectorStoreIndex

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


# Streaming

query_engine = index.as_query_engine(
    streaming=True,
)
streaming_response = query_engine.query(
    "What did the author do growing up?",
)
streaming_response.print_response_stream()

# did not get to this yet

# import llama_index
# import os

# PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
# os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
# llama_index.core.set_global_handler(
#     "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
# )

# Loading and embedding documents
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./references")
documents = reader.load_data()

# FunctionTool
from llama_index.core.tools import FunctionTool


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"


# 5. A Practical Example#
# Below is an example demonstrating how to write and run async functions with asyncio:

python
import asyncio


async def fetch_data(delay):
    print(f"Started fetching data with {delay}s delay")

    # Simulates I/O-bound work, such as network operation
    await asyncio.sleep(delay)

    print("Finished fetching data")
    return f"Data after {delay}s"


async def main():
    print("Starting main")

    # Schedule two tasks concurrently
    task1 = asyncio.create_task(fetch_data(2))
    task2 = asyncio.create_task(fetch_data(3))

    # Wait until both tasks complete
    result1, result2 = await asyncio.gather(task1, task2)

    print(result1)
    print(result2)
    print("Main complete")


if name == "main":
    asyncio.run(main())

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
tool.call("New York")

# Creating a QueryEngineTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(
    query_engine, name="some useful name", description="some useful description"
)

# Creating Toolspecs
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

[(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]

# Model Context Protocol (MCP) in LlamaIndex

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.agent.openai import get_agent
from llama_index.core.agent import Context
import asyncio

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)


# Define an async function to get the agent and create context
async def setup_mcp_agent():
    # get the agent
    agent = await get_agent(mcp_tool)

    # create the agent context
    agent_context = Context(agent)
    return agent, agent_context


# Use this to run the async function if needed
# agent, agent_context = asyncio.run(setup_mcp_agent())

# More Agents
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool


# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)], llm=llm
)

# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is Bob.", ctx=ctx)
response = await agent.run("What was my name again?", ctx=ctx)

# Creating RAG Agents with QueryEngineTools

from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(
    llm=llm, similarity_top_k=3
)  # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. ",
)

# Creating Multi-agent systems

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)


# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm,
)

# Create and run the workflow
agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")

# Creating agentic workflows in LlamaIndex
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


w = MyWorkflow(timeout=10, verbose=False)
result = await w.run()

# Connecting multiple steps
from llama_index.core.workflow import Event


class ProcessingEvent(Event):
    intermediate_result: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(timeout=10, verbose=False)
result = await w.run()
result

# Loops and branches
from llama_index.core.workflow import Event
import random


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(verbose=False)
result = await w.run()
result

# Drawing workflows
from llama_index.utils.workflow import draw_all_possible_flows

w = ...  # as defined in the previous section
draw_all_possible_flows(w, "flow.html")
w
