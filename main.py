import os
import dotenv
from textwrap import dedent
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from document_rag_tool import UploadDocument, QueryDocument, DocumentUploadRequest, DocumentQueryRequest
import logging
import time

# Set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4.1-nano")

# Path to the document or folder you want to process
document_path = input("Enter the path to your document or folder: ")

# Create document RAG handlers
upload_handler = UploadDocument()
query_handler = QueryDocument()

# Variable to store collection ID
collection_id = None

# Define functions for tools
def upload_document_func(file_path):
    global collection_id
    request = DocumentUploadRequest(file_path=file_path)
    result = upload_handler.execute(request, {})
    collection_id = result.collection_id
    return f"Status: {result.status}, Message: {result.message}, Collection ID: {result.collection_id}"

def query_document_func(query):
    global collection_id
    request = DocumentQueryRequest(query=query, collection_id=collection_id)
    result = query_handler.execute(request, {})
    return f"Response: {result.response}\nSources: {result.sources}"

# Create langchain Tool wrappers
upload_tool = Tool(
    name="UploadDocument",
    description="Tool for uploading and processing documents (PDF, DOC, TXT, etc.)",
    func=upload_document_func,
)

query_tool = Tool(
    name="QueryDocument",
    description="Tool for querying uploaded documents in the knowledge base",
    func=query_document_func,
)

# Define the Document RAG Agent
doc_rag_agent = Agent(
    role="Document RAG Agent",
    goal=dedent(
        """\
        Process uploaded documents to build a knowledge base.
        Answer questions based on the content of the uploaded documents.
        """
    ),
    verbose=False,  # Set to False to hide thinking process
    memory=True,
    backstory=dedent(
        """\
        You are an expert in document analysis and information retrieval.
        You can process various document types and answer questions based on their content.
        """
    ),
    llm=llm,
    tools=[upload_tool, query_tool],
    allow_delegation=False,
)

# Define the task for uploading the document
upload_task = Task(
    description=dedent(
        f"""\
        Upload and process the document or folder at: {document_path}
        Use the UploadDocument tool with this exact path to process this document.
        """
    ),
    expected_output="Document was successfully processed and added to the knowledge base",
    agent=doc_rag_agent,
    allow_delegation=False,
)

# Run the upload task first
print("Uploading document...")
upload_crew = Crew(
    agents=[doc_rag_agent],
    tasks=[upload_task],
    process=Process.sequential,
)
upload_result = upload_crew.kickoff()
print(f"Upload complete: {upload_result.raw if hasattr(upload_result, 'raw') else str(upload_result)}")

# Check if we have a collection ID before proceeding
if not collection_id:
    print("Error: Failed to get a valid collection ID. Please check if the document upload was successful.")
    exit(1)

print(f"Collection ID: {collection_id}")

# Function to determine if a query is complex
def is_complex_query(query):
    """Determine if a query requires complex reasoning."""
    complex_indicators = [
        "compare", "contrast", "difference", "summarize", "analyze", 
        "evaluate", "interpret", "synthesize", "why", "how would", 
        "what if", "explain", "relation", "relationship", "implications"
    ]
    
    # Check if query contains any complex indicators
    if any(indicator in query.lower() for indicator in complex_indicators):
        return True
    
    # Check if query is more than 10 words (likely complex)
    if len(query.split()) > 10:
        return True
        
    return False

# Direct query function
def direct_query(query_text):
    """Query documents directly without using the agent."""
    request = DocumentQueryRequest(query=query_text, collection_id=collection_id)
    start_time = time.time()
    result = query_handler.execute(request, {})
    end_time = time.time()
    
    print(f"\nDirect query completed in {end_time - start_time:.2f} seconds")
    return result.response

# Agent query function
def agent_query(query_text):
    """Query documents using the agent for complex reasoning."""
    query_task = Task(
        description=dedent(
            f"""\
            Based on the uploaded documents, answer this question: {query_text}
            Use the QueryDocument tool to get information from the processed documents.
            """
        ),
        expected_output="Answer to the query based on document content.",
        agent=doc_rag_agent,
        allow_delegation=False,
    )
    
    # Run the query task
    start_time = time.time()
    query_crew = Crew(
        agents=[doc_rag_agent],
        tasks=[query_task],
        process=Process.sequential,
    )
    result = query_crew.kickoff()
    end_time = time.time()
    
    print(f"\nAgent query completed in {end_time - start_time:.2f} seconds")
    return result.raw if hasattr(result, 'raw') else str(result)

# Interactive question loop
print("\n*** Document processed. You can now ask questions. Type 'exit' to quit. ***")
print("*** Simple queries will be processed directly for speed, complex queries will use the agent ***\n")

while True:
    user_query = input("\nAsk a question about your documents: ")
    if user_query.lower() in ['exit', 'quit', 'q']:
        break
    
    # Determine whether to use direct query or agent based on complexity
    if is_complex_query(user_query):
        print("Using agent for complex query...")
        answer = agent_query(user_query)
    else:
        print("Using direct query for speed...")
        answer = direct_query(user_query)
    
    print("\nAnswer:", answer)