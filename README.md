# LlamaIndex Demo

This project demonstrates how to use LlamaIndex with Hugging Face models to process documents.

## Setup

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install the required dependencies (requirements.txt - to be added)
4. Configure the `.env` file with your Hugging Face API token and references directory path

## Configuration

The project uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```
HF_TOKEN="your_huggingface_token"

# Optional: Path to references directory
# REFERENCES_DIR="/path/to/your/references/directory"
```

If `REFERENCES_DIR` is not specified, the project will use a local `./references` directory.

## References Directory

The application uses a directory of reference documents for processing. By default, it uses the `./references` directory in the project root, but you can customize this by:

1. Setting the `REFERENCES_DIR` environment variable in the `.env` file
2. Adding your documents to the specified directory

## Running the Application

To run the application:

```
python main.py
```

This will:
1. Connect to the Hugging Face API
2. Process documents from the references directory
