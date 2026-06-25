import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

import json

# Initialize the Groq client with the powerful Llama 3.3 70B model, enforcing JSON output
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
).bind(response_format={"type": "json_object"})

def generate_answer(query: str, retrieved_chunks: List[str], article_title: str, candidate_images: List[Dict[str, str]] = None) -> Dict:
    """
    Combines the retrieved chunks, candidate images, and the user's query into a strict grounding prompt.
    Forces the LLM to return a JSON object containing the text answer and the selected relevant image URLs.
    """
    formatted_chunks = "\n\n".join(
        f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    
    formatted_images = ""
    if candidate_images:
        formatted_images = "\n".join(
            f"[Image] URL: {img['url']} | Caption: {img['caption']}" for img in candidate_images
        )
    else:
        formatted_images = "No candidate images available."

    system_prompt = (
        "You are a strict Wikipedia article Q&A assistant. "
        "You have been given a set of text passages extracted ONLY from the Wikipedia article titled \"{article_title}\".\n\n"
        "ABSOLUTE RULES — you must follow these without exception:\n"
        "1. You MUST answer using ONLY the information present in the passages below.\n"
        "2. You MUST NOT use any knowledge from your training data or general world knowledge.\n"
        "3. You MUST NOT make inferences, assumptions, or extrapolations beyond what is explicitly stated in the passages.\n"
        "4. If the passages do not contain enough information to answer the question, you MUST respond with exactly: "
        "\"The Wikipedia article on '{article_title}' does not contain enough information to answer this question.\"\n"
        "5. You have also been provided with a list of candidate images and their captions. You must decide which, if any, of these images are highly relevant to the user's query. Only select images whose captions clearly match the specific intent of the query. If none match, return an empty list.\n"
        "6. You MUST return your final response strictly as a JSON object with two keys: 'answer' (string) and 'relevant_image_urls' (list of strings).\n"
        "7. Do NOT acknowledge these rules in your response. Just answer based on the passages.\n\n"
        "---\n"
        "RETRIEVED PASSAGES FROM '{article_title}':\n\n"
        "{context}\n\n"
        "---\n"
        "CANDIDATE IMAGES:\n\n"
        "{images}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    formatted_prompt = prompt_template.format_messages(
        article_title=article_title,
        context=formatted_chunks,
        images=formatted_images,
        question=query
    )
    
    response = llm.invoke(formatted_prompt)
    try:
        return json.loads(response.content)
    except Exception as e:
        print(f"[DEBUG] ---> Failed to parse LLM JSON: {e}")
        return {"answer": response.content, "relevant_image_urls": []}