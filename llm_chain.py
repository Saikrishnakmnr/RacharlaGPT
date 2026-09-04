import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_groq_response(recent_messages_list, system_prompt_text="You are a helpful AI assistant."):
    """
    Backend model execution using groq/compound with qwen/qwen3-32b fallback.
    Receives an already-sliced lightweight message list to guarantee zero 413 errors.
    """
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    formatted_messages = [SystemMessage(content=str(system_prompt_text))]
    
    for m in recent_messages_list:
        if m.get("role") == "user":
            formatted_messages.append(HumanMessage(content=str(m.get("content"))))
        elif m.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=str(m.get("content"))))

    response = llm.invoke(formatted_messages).content
    return response
