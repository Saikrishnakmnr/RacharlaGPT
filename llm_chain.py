import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_groq_response(messages_list, system_prompt_text="You are a helpful AI assistant."):
    """
    Executes Groq LLM invocation using production-ready standard model IDs.
    """
    # Use official Groq model names (no 'groq/' prefix)
    primary_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
    backup_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    # Slice context to last 4 messages to avoid token overflow
    recent_messages = messages_list[-4:] if len(messages_list) > 0 else []

    # Format structured message list
    formatted_messages = [SystemMessage(content=system_prompt_text)]
    
    for m in recent_messages:
        if m.get("role") == "user":
            formatted_messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=m.get("content", "")))

    # Fetch response
    response = llm.invoke(formatted_messages).content
    return response
