from flask import Flask, render_template, jsonify, request
from src.helpers import download_embeddings
from src.main_class import OpenRouterChat
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI # Keep if used by download_embeddings or other parts
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import * 
import os

# --- APP INITIALIZATION ---
app = Flask(__name__)

# --- CONFIGURATION AND RAG SETUP ---
load_dotenv()
load_dotenv(dotenv_path=".env")

# Fetch the keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Set the keys in os.environ (redundant if load_dotenv worked, but safe)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embedding = download_embeddings()

index_name = "educational-chatbot"

vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

ChatModel = OpenRouterChat(api_key=OPENROUTER_API_KEY, model_name="openai/gpt-4o")

# Assuming 'system_prompt' is imported from src.prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}")  
    ] 
)

question_answering_chain = create_stuff_documents_chain(ChatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

# --- FLASK ROUTES ---

# FIX 1: Add the decorator to map the root URL to the index function
@app.route("/")
def index():
    # Make sure your HTML file is named 'chat.html' and is located in a 'templates' folder
    return render_template("chat.html") 

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get message from POST form (preferred) or GET query
    if request.method == "POST":
        # The frontend sends data as 'application/x-www-form-urlencoded', so use request.form
        msg = request.form.get("msg") 
    else: # GET request
        msg = request.args.get("msg")

    if not msg:
        return "Error: No message received.", 400

    input_text = msg
    print("User Input:", input_text)

    # Invoke RAG chain
    try:
        response = rag_chain.invoke({"input": input_text})

        # Process and return the response
        if isinstance(response, dict):
            # Prioritize 'answer' or 'output_text' from the dictionary response
            output = response.get("answer") or response.get("output_text")
            print("Response:", output)
            return str(output)
        else:
            # Handle direct string or non-dict responses
            print("Response:", response)
            return str(response)
    
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        # Return a 500 status code for server-side errors
        return "An internal server error occurred while processing your request.", 500
    

# --- APP RUNNER ---
if __name__ == "__main__":
    # Recommended for deployment/Docker (0.0.0.0) but keep the specific port 8080
    # For local testing, you can use app.run(debug=True)
    print(f"\n* Running on http://0.0.0.0:8080/")
    app.run(host="0.0.0.0", port=8080, debug=True)