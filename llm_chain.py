import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

def get_groq_response(messages_list):
    """
    Initializes Groq LLMs and invokes response on sliced context.
    """
    # Primary & Backup Model Setup
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    # Limit payload to last 6 messages to avoid 413 Payload Too Large errors
    recent_messages = messages_list[-6:]

    # Format history into LangChain message objects
    formatted_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in recent_messages
    ]

    # Generate response content
    response = llm.invoke(formatted_messages).content
    return response
