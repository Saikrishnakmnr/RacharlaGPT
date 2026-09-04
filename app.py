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

# 3. Sidebar UI
st.sidebar.header("Configuration")
system_prompt = st.sidebar.text_area(
    "System Prompt (AI Persona):",
    value="You are a helpful, witty, and concise AI assistant."
)

# Reset Button to wipe memory
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 4. Session State Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Render visible chat log
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Prompt Logic
if prompt := st.chat_input("Type your message..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant output
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send sanitized copy of memory
                response = get_groq_response(st.session_state.messages, system_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
