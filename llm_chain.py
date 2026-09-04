import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_groq_response(messages_list, system_prompt_text="You are a helpful AI assistant."):
    """
    Executes Groq LLM queries with primary (groq/compound) and backup (qwen/qwen3-32b) models.
    Converts and sanitizes input to raw text strings to prevent 413 payload crashes.
    """
    # Initialize requested primary and fallback models
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    # Keep strictly the last 2 turns (1 user prompt + 1 assistant reply max)
    recent_messages = messages_list[-2:] if len(messages_list) > 0 else []

    # Construct clean payload using ONLY standard string content
    formatted_messages = [SystemMessage(content=str(system_prompt_text)[:200])]
    
    for m in recent_messages:
        # Extract raw text string and discard any hidden metadata/reasoning attributes
        raw_text = str(m.get("content", ""))[:500]
        
        if m.get("role") == "user":
            formatted_messages.append(HumanMessage(content=raw_text))
        elif m.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=raw_text))

    # Fetch string response
    response_obj = llm.invoke(formatted_messages)
    return str(response_obj.content)
