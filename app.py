import os
import streamlit as st
from llm_chain import get_groq_response

# 1. Page Configuration
st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

# 2. Secret Verification
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

# 3. Sidebar Configuration
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant powered by Groq."
)

# New Chat button to reset conversational memory
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 4. Initialize Session Memory State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Render existing visual history on UI reloads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Capture User Input & Execute API Request
if prompt := st.chat_input("Type your message..."):
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke backend helper and display output
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_groq_response(st.session_state.messages, system_prompt)
            st.markdown(response)

    # Append response to memory state
    st.session_state.messages.append({"role": "assistant", "content": response})
