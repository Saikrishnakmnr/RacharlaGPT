import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

def get_groq_response(user_text, system_prompt_text="You are a helpful AI assistant."):
    """
    Stateless execution engine. Sends ONLY the single prompt string.
    """
    primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
    backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
    llm = primary_llm.with_fallbacks([backup_llm])

    formatted_messages = [
        SystemMessage(content=str(system_prompt_text)),
        HumanMessage(content=str(user_text))
    ]

    response = llm.invoke(formatted_messages).content
    return str(response)
