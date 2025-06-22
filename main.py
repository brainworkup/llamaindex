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

query_engine = # from the previous section
llm = # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
