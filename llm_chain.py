import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

def get_groq_response(user_text, system_prompt_text="You are a helpful AI assistant."):
    """
    Direct execution engine. Configures reasoning_format directly on ChatGroq
    to prevent Pydantic validation errors and payload bloat.
    """
    # 1. Initialize models with explicit reasoning_format top-level parameter
    primary_llm = ChatGroq(
        model_name="groq/compound",
        temperature=0.7,
        reasoning_format="hidden"
    )
    
    backup_llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.7,
        reasoning_format="hidden"
    )
    
    llm = primary_llm.with_fallbacks([backup_llm])

    # 2. Build clean payload
    formatted_messages = [
        SystemMessage(content=str(system_prompt_text)[:300]),
        HumanMessage(content=str(user_text)[:1000])
    ]

    # 3. Invoke LLM and return direct text content
    response_obj = llm.invoke(formatted_messages)
    return str(response_obj.content)
