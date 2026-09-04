import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_groq_response(messages_list, system_prompt_text="You are a helpful AI assistant."):
    """
    Initializes Groq models with fallback and processes response using sliced history.
    """
    # 1. Primary & Backup Model Setup with Fallback Protection
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    # 2. Slice context to last 4 turns to avoid 413 Payload Too Large limits
    recent_messages = messages_list[-4:]

    # 3. Construct message array starting with System Persona
    formatted_messages = [SystemMessage(content=system_prompt_text)]
    
    for m in recent_messages:
        if m["role"] == "user":
            formatted_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            formatted_messages.append(AIMessage(content=m["content"]))

    # 4. Invoke model and return generated string
    response = llm.invoke(formatted_messages).content
    return response
