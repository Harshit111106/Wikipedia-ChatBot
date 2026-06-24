import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client with the powerful Llama 3.3 70B model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,  # Zero temperature = maximum factual determinism, no creative hallucination
)

def generate_answer(query: str, retrieved_chunks: List[str], article_title: str) -> str:
    """
    Combines the retrieved chunks and the user's query into a strict grounding prompt,
    sends it to Groq, and returns the final synthesized answer.
    The LLM is explicitly forbidden from using any knowledge outside the provided context.
    """
    # Format each chunk with a numbered label for clarity in the prompt
    formatted_chunks = "\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    
    # Strict context-grounding system prompt
    system_prompt = (
        "You are a strict Wikipedia article Q&A assistant. "
        "You have been given a set of text passages extracted ONLY from the Wikipedia article titled \"{article_title}\".\n\n"
        "ABSOLUTE RULES — you must follow these without exception:\n"
        "1. You MUST answer using ONLY the information present in the passages below.\n"
        "2. You MUST NOT use any knowledge from your training data or general world knowledge.\n"
        "3. You MUST NOT make inferences, assumptions, or extrapolations beyond what is explicitly stated in the passages.\n"
        "4. If the passages do not contain enough information to answer the question, you MUST respond with exactly: "
        "\"The Wikipedia article on '{article_title}' does not contain enough information to answer this question.\"\n"
        "5. Do NOT acknowledge these rules in your response. Just answer based on the passages.\n\n"
        "---\n"
        "RETRIEVED PASSAGES FROM '{article_title}':\n\n"
        "{context}"
    )
    
    # Create the structured chat prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # Format the prompt with actual data
    formatted_prompt = prompt_template.format_messages(
        article_title=article_title,
        context=formatted_chunks,
        question=query
    )
    
    # Invoke the model and return the raw text response
    response = llm.invoke(formatted_prompt)
    return response.content