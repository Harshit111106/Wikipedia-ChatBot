import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client with the powerful Llama 3.3 70B model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,  # Low temperature means more factual, less creative
)

def generate_answer(query: str, retrieved_chunks: list) -> str:
    """
    Combines the retrieved chunks and the user's query into a prompt,
    sends it to Groq, and returns the final synthesized answer.
    """
    # Join our list of text chunks into a single string block
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    
    # Define the template instructions for the model
    system_prompt = (
        "You are an expert AI assistant providing answers based strictly on the provided context.\n"
        "Analyze the retrieved snippets from Wikipedia carefully and answer the question accurately.\n"
        "If the context does not contain the answer, state clearly that you cannot find the answer based on the document.\n\n"
        "Retrieved Context:\n{context}"
    )
    
    # Create the structured chat prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # Format the prompt with our actual data
    formatted_prompt = prompt_template.format_messages(
        context=context_text,
        question=query
    )
    
    # Invoke the model and return the raw text response
    response = llm.invoke(formatted_prompt)
    return response.content