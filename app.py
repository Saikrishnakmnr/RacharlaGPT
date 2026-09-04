import os
import streamlit as st
from llm_chain import get_groq_response

st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_groq_response(prompt, system_prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
