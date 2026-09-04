import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_groq_response(messages_list, system_prompt_text="You are a helpful AI assistant."):
    """
    Ultra-lightweight execution engine designed to conserve Groq API credits and avoid 413 errors.
    """
    # 1. Primary and Backup Models
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    # 2. Keep ONLY the last 2 turns (1 user message + 1 previous reply) to save credits
    recent_messages = messages_list[-2:] if len(messages_list) > 0 else []

    # 3. Truncate system prompt to max 250 chars
    formatted_messages = [SystemMessage(content=system_prompt_text[:250])]
    
    for m in recent_messages:
        # Hard-cap every message to 500 characters max to prevent token inflation
        clean_content = str(m.get("content", ""))[:500]
        
        if m.get("role") == "user":
            formatted_messages.append(HumanMessage(content=clean_content))
        elif m.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=clean_content))

    # 4. Invoke LLM and return text
    response = llm.invoke(formatted_messages).content
    return response
