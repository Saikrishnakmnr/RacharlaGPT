# ⚡ RacharlaGPT

**RacharlaGPT** is a high-performance, public-facing AI conversational assistant built with **Streamlit**, **LangChain**, and **Groq**. Powered by Groq's ultra-fast LPU inference engine, it delivers sub-second response times, interactive prompt customization, and automated model failovers.

---

## 🌟 Key Features

* **⚡ Ultra-Fast Inference:** Leverages Groq's LPU architecture for high-speed response generation.
* **🛡️ Automated Fallback Architecture:** Features dynamic failover protection using `groq/compound` as the primary engine with automatic switching to `qwen/qwen3.8-27b` if needed.
* **💬 Persistent Conversation Memory:** Utilizes `st.session_state` and LangChain memory placeholders to retain full conversational context.
* **🎭 Dynamic System Prompts:** Customizable sidebar interface to change the AI's persona on the fly.
* **🔒 Production-Grade Security:** Server-side secret management via `st.secrets` ensuring zero client-side API key exposure.

---

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit
* **Orchestration:** LangChain (`langchain-groq`, `langchain-core`)
* **LLM Infrastructure:** Groq API
* **Language:** Python 3.10+

---

## 🚀 Quick Start (Local Setup)

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/RacharlaGPT.git](https://github.com/YOUR_GITHUB_USERNAME/RacharlaGPT.git)
   cd RacharlaGPT
