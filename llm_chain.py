import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

def get_groq_response(user_text, system_prompt_text="You are a helpful AI assistant."):
    """
    Agentic execution engine for groq/compound and qwen/qwen3-32b.
    Strips reasoning metadata to eliminate 413 Payload Too Large errors.
    """
    # 1. Initialize models with reasoning output disabled to prevent token payload bloat
    primary_llm = ChatGroq(
        model_name="groq/compound",
        temperature=0.7,
        model_kwargs={"reasoning_format": "hidden"}
    )
    
    backup_llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.7,
        model_kwargs={"reasoning_format": "hidden"}
    )
    
    llm = primary_llm.with_fallbacks([backup_llm])

    # 2. Direct clean payload with no carried-over internal agent bloat
    formatted_messages = [
        SystemMessage(content=str(system_prompt_text)[:300]),
        HumanMessage(content=str(user_text)[:1000])
    ]

    # 3. Invoke LLM and return clean string output
    response_obj = llm.invoke(formatted_messages)
    return str(response_obj.content)
