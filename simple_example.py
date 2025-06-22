import os
import pathlib
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Use environment variable if available, otherwise use a relative path
references_dir = os.getenv("REFERENCES_DIR", "./references")

# Create the references directory if it doesn't exist
pathlib.Path(references_dir).mkdir(exist_ok=True)

print(f"Using references directory: {references_dir}")
print(f"Directory exists: {os.path.exists(references_dir)}")

# Simulate file operations without requiring LlamaIndex
files = os.listdir(references_dir)
print(f"Files in directory: {files}")

# This demonstrates the fix for the original issue:
# - Using environment variable for configuration
# - Falling back to relative path
# - Creating directory if it doesn't exist
