import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from supabase import create_client, Client

st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "RacharlaGPT"
PRIMARY_MODEL = "openai/gpt-oss-20b"
BACKUP_MODEL = "llama-3.1-8b-instant"
MAX_CONTEXT_MESSAGES = 14

DEFAULT_SYSTEM_PROMPT = """You are RacharlaGPT, a helpful, intelligent, friendly and concise AI assistant.
Give accurate, clear and useful answers.
Use headings, bullet points, numbered steps and tables when they improve readability.
If the user asks for code, provide complete working code and explain important changes briefly.
Do not invent facts. If uncertain, say so.
Be practical and solution-oriented."""


def secret(name):
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        value = st.secrets.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


GROQ_API_KEY = secret("GROQ_API_KEY")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_KEY = secret("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase is not configured.")
    st.info("Add SUPABASE_URL and SUPABASE_KEY in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


@st.cache_resource(show_spinner=False)
def get_supabase(url: str, key: str) -> Client:
    return create_client(url, key)


supabase = get_supabase(SUPABASE_URL, SUPABASE_KEY)


st.markdown("""
<style>
.stApp { background:#fff; }
section[data-testid="stSidebar"] { background:#f7f8fc; border-right:1px solid #e7e9ef; }
.brand { display:flex; align-items:center; gap:12px; margin:8px 0 4px; }
.icon { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center; font-size:23px; background:linear-gradient(135deg,#ff4b4b,#ff9f1c); }
.name { font-size:30px; font-weight:800; color:#172033; letter-spacing:-1px; }
.sub { color:#6b7280; margin:0 0 24px 54px; font-size:14px; }
.auth { max-width:480px; margin:7vh auto; text-align:center; }
.welcome { max-width:760px; margin:10vh auto 3vh; text-align:center; padding:35px 20px; }
.wtitle { font-size:34px; font-weight:800; color:#172033; }
.wtext { color:#6b7280; line-height:1.6; }
@media(max-width:700px){ .name{font-size:25px}.wtitle{font-size:27px} }
</style>
""", unsafe_allow_html=True)


# ---------------- Authentication ----------------

def show_auth():
    st.markdown("""
    <div class="auth">
        <div style="font-size:52px">⚡</div>
        <h1>Welcome to RacharlaGPT</h1>
        <p style="color:#6b7280">Sign in to save your conversations permanently.</p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        login_tab, signup_tab = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit:
                if not email or not password:
                    st.warning("Enter your email and password.")
                else:
                    try:
                        result = supabase.auth.sign_in_with_password({
                            "email": email.strip(),
                            "password": password,
                        })
                        if result.user:
                            st.session_state.auth_user = result.user
                            st.rerun()
                        else:
                            st.error("Sign in failed.")
                    except Exception as exc:
                        st.error(f"Sign in failed: {exc}")

        with signup_tab:
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submit:
                if not email or not password:
                    st.warning("Enter your email and password.")
                elif password != confirm:
                    st.warning("Passwords do not match.")
                elif len(password) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    try:
                        result = supabase.auth.sign_up({
                            "email": email.strip(),
                            "password": password,
                        })
                        if result.session and result.user:
                            st.session_state.auth_user = result.user
                            st.rerun()
                        else:
                            st.success("Account created. Check your email for confirmation, then sign in.")
                    except Exception as exc:
                        st.error(f"Account creation failed: {exc}")


# IMPORTANT: no authenticated user = login screen only.
if "auth_user" not in st.session_state:
    show_auth()
    st.stop()

auth_user = st.session_state.auth_user
USER_ID = str(auth_user.id)


# ---------------- Models ----------------

@st.cache_resource(show_spinner=False)
def get_models(key):
    primary = ChatGroq(model=PRIMARY_MODEL, temperature=0.7, api_key=key)
    backup = ChatGroq(model=BACKUP_MODEL, temperature=0.7, api_key=key)
    return primary, backup


primary_llm, backup_llm = get_models(GROQ_API_KEY)


# ---------------- State ----------------

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "loaded_from_supabase" not in st.session_state:
    st.session_state.loaded_from_supabase = False
if "search" not in st.session_state:
    st.session_state.search = ""


def blank_chat():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "title": "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def load_chats():
    result = (
        supabase.table("chats")
        .select("*")
        .eq("user_id", USER_ID)
        .order("updated_at", desc=True)
        .execute()
    )

    chats = {}
    for row in result.data or []:
        chats[str(row["id"])] = {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "title": row.get("title") or "New Chat",
            "messages": row.get("messages") or [],
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    if not chats:
        chat = blank_chat()
        supabase.table("chats").insert(chat).execute()
        chats[chat["id"]] = chat

    return chats


def save_chat(chat):
    now = datetime.now(timezone.utc).isoformat()
    (
        supabase.table("chats")
        .update({
            "title": chat["title"],
            "messages": chat["messages"],
            "updated_at": now,
        })
        .eq("id", chat["id"])
        .eq("user_id", USER_ID)
        .execute()
    )
    chat["updated_at"] = now


def create_chat():
    chat = blank_chat()
    supabase.table("chats").insert(chat).execute()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.current_chat_id = chat["id"]


def delete_chat(chat_id):
    supabase.table("chats").delete().eq("id", chat_id).eq("user_id", USER_ID).execute()
    st.session_state.chats.pop(chat_id, None)
    if not st.session_state.chats:
        create_chat()
    else:
        ordered = sorted(st.session_state.chats.values(), key=lambda x: x.get("updated_at") or "", reverse=True)
        st.session_state.current_chat_id = ordered[0]["id"]


def delete_all():
    supabase.table("chats").delete().eq("user_id", USER_ID).execute()
    st.session_state.chats = {}
    create_chat()


def title_for(text):
    text = " ".join(text.strip().split())
    return text if len(text) <= 34 else text[:34].rstrip() + "…"


def current_chat():
    if st.session_state.current_chat_id not in st.session_state.chats:
        create_chat()
    return st.session_state.chats[st.session_state.current_chat_id]


def ask(chat):
    recent = chat["messages"][-MAX_CONTEXT_MESSAGES:]
    msgs = [SystemMessage(content=st.session_state.system_prompt)]
    for m in recent:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    try:
        return primary_llm.invoke(msgs), "primary"
    except RateLimitError:
        return backup_llm.invoke(msgs), "backup"


if not st.session_state.loaded_from_supabase:
    try:
        st.session_state.chats = load_chats()
        st.session_state.current_chat_id = next(iter(st.session_state.chats))
        st.session_state.loaded_from_supabase = True
    except Exception as exc:
        st.error("Could not load chats from Supabase.")
        st.code(str(exc))
        st.stop()


# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown("## ⚡ RacharlaGPT")

    if st.button("＋ New Chat", type="primary", use_container_width=True):
        create_chat()
        st.rerun()

    st.caption(f"Signed in: {getattr(auth_user, 'email', 'User')}")

    if st.button("🚪 Sign Out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        for key in ["auth_user", "chats", "current_chat_id", "loaded_from_supabase"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("### YOUR CHATS")
    search = st.text_input("Search", placeholder="🔎 Search your chats...", label_visibility="collapsed")
    st.session_state.search = search

    # Safe rename: only changes the existing chat title in Supabase.
    if st.session_state.current_chat_id in st.session_state.chats:
        with st.expander("✏️ Rename current chat"):
            rename_value = st.text_input(
                "Chat name",
                value=st.session_state.chats[st.session_state.current_chat_id]["title"],
                key="rename_chat_title",
            )
            if st.button("Save name", use_container_width=True):
                rename_value = " ".join(rename_value.strip().split())
                if rename_value:
                    current = st.session_state.chats[st.session_state.current_chat_id]
                    current["title"] = rename_value[:80]
                    save_chat(current)
                    st.rerun()

    ordered = sorted(st.session_state.chats.items(), key=lambda x: x[1].get("updated_at") or "", reverse=True)
    for chat_id, item in ordered:
        q = search.lower().strip()
        matches = not q or q in item["title"].lower() or any(q in str(m.get("content","")).lower() for m in item["messages"])
        if not matches:
            continue

        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button(f"💬 {item['title']}", key=f"open_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{chat_id}", use_container_width=True):
                delete_chat(chat_id)
                st.rerun()

    st.divider()
    if st.button("🗑️ Delete All Chats", use_container_width=True):
        delete_all()
        st.rerun()

    st.divider()
    st.session_state.system_prompt = st.text_area(
        "AI System Prompt",
        value=st.session_state.system_prompt,
        height=150,
    )

    st.caption(f"Primary: {PRIMARY_MODEL}")
    st.caption(f"Fallback: {BACKUP_MODEL}")
    st.caption("☁️ Chats stored in Supabase")


# ---------------- Main UI ----------------

st.markdown("""
<div class="brand">
    <div class="icon">⚡</div>
    <div class="name">RacharlaGPT</div>
</div>
<div class="sub">Fast, intelligent AI conversations powered by Groq</div>
""", unsafe_allow_html=True)

chat = current_chat()

if not chat["messages"]:
    st.markdown("""
    <div class="welcome">
        <div style="font-size:48px">⚡</div>
        <div class="wtitle">Welcome to RacharlaGPT</div>
        <div class="wtext">Your conversations are saved to your account. Ask questions, write code, brainstorm ideas, or learn something new.</div>
    </div>
    """, unsafe_allow_html=True)

for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Message RacharlaGPT...")

if user_input:
    user_input = user_input.strip()
    if user_input:
        chat["messages"].append({"role": "user", "content": user_input})
        if chat["title"] == "New Chat":
            chat["title"] = title_for(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("RacharlaGPT is thinking…"):
                    response, used = ask(chat)
                answer = response.content
                if not isinstance(answer, str):
                    answer = str(answer)
                st.markdown(answer)
                if used == "backup":
                    st.caption(f"Primary model was rate-limited; answered with {BACKUP_MODEL}.")

            chat["messages"].append({"role": "assistant", "content": answer})
            save_chat(chat)

        except RateLimitError:
            with st.chat_message("assistant"):
                st.warning("Groq rate limit reached. Please wait for the quota to reset and try again.")
            chat["messages"].pop()

        except Exception as exc:
            with st.chat_message("assistant"):
                st.error("Something went wrong while contacting Groq. Please try again.")
            chat["messages"].pop()
            print(f"RacharlaGPT error: {type(exc).__name__}: {exc}")

st.markdown(
    '<div style="text-align:center;color:#9aa0ad;font-size:11px;margin-top:12px;">'
    'RacharlaGPT • Powered by Groq • Chats stored in Supabase'
    '</div>',
    unsafe_allow_html=True,
)
