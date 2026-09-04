import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Page Config
st.set_page_config(page_title="RacharlaGPT", page_icon="🤖")
st.title("RacharlaGPT")

# 2. Secret Verification
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing from Streamlit secrets!")
    st.stop()

# 3. Sidebar UI Configuration
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

# 5. Render visible chat log on UI reloads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Prompt Execution Logic
if prompt := st.chat_input("Type your message..."):
    # Append and display user message locally on screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Primary and Fallback setup
                primary_llm = ChatGroq(model_name="groq/compound", temperature=0.7)
                backup_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.7)
                llm = primary_llm.with_fallbacks([backup_llm])

                # CRITICAL FIX: Send ONLY SystemMessage + current HumanMessage
                # Omitting AIMessage prevents groq/compound's hidden reasoning payload from crashing the API
                formatted_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]

                # Invoke LLM
                response = llm.invoke(formatted_messages).content
                st.markdown(response)

                # Store response in session state for visual screen history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
