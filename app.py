import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from supabase import create_client, Client


# =========================================================
# RacharlaGPT
# Stable version
# =========================================================

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

MODEL_OPTIONS = [
    "Auto (recommended)",
    PRIMARY_MODEL,
    BACKUP_MODEL,
]


# =========================================================
# DEFAULT SYSTEM PROMPT
# =========================================================

DEFAULT_SYSTEM_PROMPT = """
You are RacharlaGPT, a helpful, intelligent, friendly and concise AI assistant.

IMPORTANT BEHAVIOR:

1. Understand spelling mistakes, typing mistakes, abbreviations,
   and imperfect English. Never stop responding just because the
   user misspelled a word.

2. Interpret the user's intended meaning from context.
   For example, "speciality", "specialty", "speaciality",
   or similar spellings should normally be understood as
   "specialty/speciality".

3. Always try to answer the user's actual question.
   If a question is ambiguous, explain the likely meaning and
   ask a short clarification only when necessary.

4. You know the current date provided below.
   Use it when the user asks:
   - today
   - tomorrow
   - yesterday
   - day of the week
   - current date
   - this week
   - this month
   - relative dates

5. Do not incorrectly claim that you cannot know today's day
   when the current date is explicitly provided in this prompt.

6. Give accurate, clear and useful answers.

7. Use headings, bullet points, numbered steps and tables
   when they improve readability.

8. If the user asks for code, provide complete working code
   and briefly explain important changes.

9. Do not invent facts. If you are uncertain, say so.

10. Be practical and solution-oriented.
"""


# =========================================================
# CURRENT DATE / TIME CONTEXT
# =========================================================

now_local = datetime.now().astimezone()

CURRENT_DATE = now_local.strftime("%Y-%m-%d")
CURRENT_DAY = now_local.strftime("%A")
CURRENT_TIME = now_local.strftime("%I:%M %p")
CURRENT_TIMEZONE = now_local.strftime("%Z")

DATE_CONTEXT = f"""
CURRENT DATE AND TIME CONTEXT:

Date: {CURRENT_DATE}
Day: {CURRENT_DAY}
Time: {CURRENT_TIME}
Timezone: {CURRENT_TIMEZONE}

Use this information for relative-date questions.
"""


# =========================================================
# SECRET HELPER
# =========================================================

def secret(name):
    value = os.environ.get(name)

    if value:
        return value.strip()

    try:
        value = st.secrets.get(name)

        if value:
            return str(value).strip()

    except Exception:
        pass

    return None


GROQ_API_KEY = secret("GROQ_API_KEY")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_KEY = secret("SUPABASE_KEY")


# =========================================================
# CONFIG CHECK
# =========================================================

if not SUPABASE_URL or not SUPABASE_KEY:

    st.error("Supabase is not configured.")

    st.info(
        "Add SUPABASE_URL and SUPABASE_KEY in "
        "Streamlit Cloud → Manage app → Settings → Secrets."
    )

    st.stop()


if not GROQ_API_KEY:

    st.error("GROQ_API_KEY is not configured.")

    st.stop()


os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =========================================================
# SUPABASE CLIENT
# =========================================================

if "supabase_client" not in st.session_state:

    st.session_state.supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )


supabase: Client = st.session_state.supabase_client


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background: #f7f8fc;
        border-right: 1px solid #e7e9ef;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 8px 0 4px;
    }

    .icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
        background: linear-gradient(
            135deg,
            #ff4b4b,
            #ff9f1c
        );
    }

    .name {
        font-size: 30px;
        font-weight: 800;
        color: #172033;
        letter-spacing: -1px;
    }

    .sub {
        color: #6b7280;
        margin: 0 0 24px 54px;
        font-size: 14px;
    }

    .auth {
        max-width: 480px;
        margin: 7vh auto;
        text-align: center;
    }

    .welcome {
        max-width: 760px;
        margin: 10vh auto 3vh;
        text-align: center;
        padding: 35px 20px;
    }

    .wtitle {
        font-size: 34px;
        font-weight: 800;
        color: #172033;
    }

    .wtext {
        color: #6b7280;
        line-height: 1.6;
    }

    @media(max-width:700px) {

        .name {
            font-size: 25px;
        }

        .wtitle {
            font-size: 27px;
        }

        .sub {
            margin-left: 0;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# AUTHENTICATION
# =========================================================

def show_auth():

    st.markdown(
        """
        <div class="auth">
            <div style="font-size:52px">⚡</div>

            <h1>Welcome to RacharlaGPT</h1>

            <p style="color:#6b7280">
                Sign in to save your conversations permanently.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:

        login_tab, signup_tab = st.tabs(
            [
                "🔐 Sign In",
                "✨ Create Account",
            ]
        )

        # =================================================
        # SIGN IN
        # =================================================

        with login_tab:

            with st.form("login_form"):

                email = st.text_input(
                    "Email",
                    key="login_email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )

                submit = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True,
                )

            if submit:

                if not email or not password:

                    st.warning(
                        "Enter your email and password."
                    )

                else:

                    try:

                        result = (
                            supabase
                            .auth
                            .sign_in_with_password(
                                {
                                    "email": email.strip(),
                                    "password": password,
                                }
                            )
                        )

                        if result.user and result.session:

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Sign in failed."
                            )

                    except Exception as exc:

                        st.error(
                            f"Sign in failed: {exc}"
                        )

        # =================================================
        # CREATE ACCOUNT
        # =================================================

        with signup_tab:

            with st.form("signup_form"):

                email = st.text_input(
                    "Email",
                    key="signup_email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="signup_password",
                )

                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="signup_confirm",
                )

                submit = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True,
                )

            if submit:

                if not email or not password:

                    st.warning(
                        "Enter your email and password."
                    )

                elif password != confirm:

                    st.warning(
                        "Passwords do not match."
                    )

                elif len(password) < 6:

                    st.warning(
                        "Password must be at least 6 characters."
                    )

                else:

                    try:

                        result = supabase.auth.sign_up(
                            {
                                "email": email.strip(),
                                "password": password,
                            }
                        )

                        if result.session and result.user:

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Account created. "
                                "Check your email for confirmation, "
                                "then sign in."
                            )

                    except Exception as exc:

                        st.error(
                            f"Account creation failed: {exc}"
                        )


# =========================================================
# AUTH GATE
# =========================================================

if "auth_user" not in st.session_state:

    show_auth()

    st.stop()


auth_user = st.session_state.auth_user

USER_ID = str(auth_user.id)


# =========================================================
# STATE
# =========================================================

if "system_prompt" not in st.session_state:

    st.session_state.system_prompt = (
        DEFAULT_SYSTEM_PROMPT
    )


if "chats" not in st.session_state:

    st.session_state.chats = {}


if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = None


if "loaded_from_supabase" not in st.session_state:

    st.session_state.loaded_from_supabase = False


if "search" not in st.session_state:

    st.session_state.search = ""


if "selected_model" not in st.session_state:

    st.session_state.selected_model = (
        "Auto (recommended)"
    )


if "temperature" not in st.session_state:

    st.session_state.temperature = 0.7


# =========================================================
# CHAT HELPERS
# =========================================================

def blank_chat():

    now = datetime.now(
        timezone.utc
    ).isoformat()

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
        supabase
        .table("chats")
        .select("*")
        .eq("user_id", USER_ID)
        .order("updated_at", desc=True)
        .execute()
    )

    chats = {}

    for row in result.data or []:

        chats[str(row["id"])] = {

            "id": str(row["id"]),

            "user_id": str(
                row["user_id"]
            ),

            "title": (
                row.get("title")
                or "New Chat"
            ),

            "messages": (
                row.get("messages")
                or []
            ),

            "created_at": row.get(
                "created_at"
            ),

            "updated_at": row.get(
                "updated_at"
            ),
        }

    # Create first chat for new account
    if not chats:

        chat = blank_chat()

        supabase.table(
            "chats"
        ).insert(chat).execute()

        chats[chat["id"]] = chat

    return chats


def save_chat(chat):

    now = datetime.now(
        timezone.utc
    ).isoformat()

    (
        supabase
        .table("chats")
        .update(
            {
                "title": chat["title"],
                "messages": chat["messages"],
                "updated_at": now,
            }
        )
        .eq("id", chat["id"])
        .eq("user_id", USER_ID)
        .execute()
    )

    chat["updated_at"] = now


def create_chat():

    chat = blank_chat()

    supabase.table(
        "chats"
    ).insert(chat).execute()

    st.session_state.chats[
        chat["id"]
    ] = chat

    st.session_state.current_chat_id = (
        chat["id"]
    )


def delete_chat(chat_id):

    (
        supabase
        .table("chats")
        .delete()
        .eq("id", chat_id)
        .eq("user_id", USER_ID)
        .execute()
    )

    st.session_state.chats.pop(
        chat_id,
        None,
    )

    if not st.session_state.chats:

        create_chat()

    else:

        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda x: x.get(
                "updated_at"
            ) or "",
            reverse=True,
        )

        st.session_state.current_chat_id = (
            ordered[0]["id"]
        )


def delete_all():

    (
        supabase
        .table("chats")
        .delete()
        .eq("user_id", USER_ID)
        .execute()
    )

    st.session_state.chats = {}

    create_chat()


def title_for(text):

    text = " ".join(
        text.strip().split()
    )

    if not text:

        return "New Chat"

    return (
        text
        if len(text) <= 34
        else text[:34].rstrip() + "…"
    )


def current_chat():

    if (
        st.session_state.current_chat_id
        not in st.session_state.chats
    ):

        create_chat()

    return st.session_state.chats[
        st.session_state.current_chat_id
    ]


# =========================================================
# AI
# =========================================================

def build_messages(chat):

    recent = chat["messages"][
        -MAX_CONTEXT_MESSAGES:
    ]

    system_text = (
        st.session_state.system_prompt
        + "\n\n"
        + DATE_CONTEXT
    )

    messages = [
        SystemMessage(
            content=system_text
        )
    ]

    for message in recent:

        content = str(
            message.get("content", "")
        )

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        else:

            messages.append(
                AIMessage(
                    content=content
                )
            )

    return messages


def make_llm(model_name):

    return ChatGroq(
        model=model_name,
        temperature=(
            st.session_state.temperature
        ),
        api_key=GROQ_API_KEY,
    )


def ask(chat):

    messages = build_messages(chat)

    selected = (
        st.session_state.selected_model
    )

    # -----------------------------------------------------
    # Explicit primary model
    # -----------------------------------------------------

    if selected == PRIMARY_MODEL:

        try:

            llm = make_llm(
                PRIMARY_MODEL
            )

            return (
                llm.invoke(messages),
                "primary",
            )

        except RateLimitError:

            llm = make_llm(
                BACKUP_MODEL
            )

            return (
                llm.invoke(messages),
                "backup",
            )

        except Exception as primary_error:

            try:

                llm = make_llm(
                    BACKUP_MODEL
                )

                return (
                    llm.invoke(messages),
                    "backup",
                )

            except Exception:

                raise primary_error

    # -----------------------------------------------------
    # Explicit backup model
    # -----------------------------------------------------

    if selected == BACKUP_MODEL:

        llm = make_llm(
            BACKUP_MODEL
        )

        return (
            llm.invoke(messages),
            "backup",
        )

    # -----------------------------------------------------
    # Auto
    # -----------------------------------------------------

    try:

        llm = make_llm(
            PRIMARY_MODEL
        )

        return (
            llm.invoke(messages),
            "primary",
        )

    except RateLimitError:

        llm = make_llm(
            BACKUP_MODEL
        )

        return (
            llm.invoke(messages),
            "backup",
        )

    except Exception as primary_error:

        try:

            llm = make_llm(
                BACKUP_MODEL
            )

            return (
                llm.invoke(messages),
                "backup",
            )

        except Exception:

            raise primary_error


# =========================================================
# LOAD FROM SUPABASE
# =========================================================

if not st.session_state.loaded_from_supabase:

    try:

        st.session_state.chats = (
            load_chats()
        )

        ordered = sorted(
            st.session_state.chats.values(),
            key=lambda x: x.get(
                "updated_at"
            ) or "",
            reverse=True,
        )

        if ordered:

            st.session_state.current_chat_id = (
                ordered[0]["id"]
            )

        st.session_state.loaded_from_supabase = (
            True
        )

    except Exception as exc:

        st.error(
            "Could not load chats from Supabase."
        )

        st.code(
            str(exc)
        )

        st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ⚡ RacharlaGPT"
    )

    # -----------------------------------------------------
    # New Chat
    # -----------------------------------------------------

    if st.button(
        "＋ New Chat",
        type="primary",
        use_container_width=True,
    ):

        create_chat()

        st.rerun()

    # -----------------------------------------------------
    # User
    # -----------------------------------------------------

    st.caption(
        "Signed in: "
        + str(
            getattr(
                auth_user,
                "email",
                "User",
            )
        )
    )

    # -----------------------------------------------------
    # Sign Out
    # -----------------------------------------------------

    if st.button(
        "🚪 Sign Out",
        use_container_width=True,
    ):

        try:

            supabase.auth.sign_out()

        except Exception:

            pass

        for key in [
            "auth_user",
            "chats",
            "current_chat_id",
            "loaded_from_supabase",
        ]:

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()

    st.markdown(
        "### YOUR CHATS"
    )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    search = st.text_input(
        "Search",
        placeholder="🔎 Search your chats...",
        label_visibility="collapsed",
    )

    st.session_state.search = search

    # -----------------------------------------------------
    # Rename
    # -----------------------------------------------------

    if (
        st.session_state.current_chat_id
        in st.session_state.chats
    ):

        with st.expander(
            "✏️ Rename current chat"
        ):

            rename_value = st.text_input(
                "Chat name",
                value=(
                    st.session_state
                    .chats[
                        st.session_state
                        .current_chat_id
                    ]["title"]
                ),
                key="rename_chat_title",
            )

            if st.button(
                "Save name",
                use_container_width=True,
            ):

                rename_value = " ".join(
                    rename_value.strip().split()
                )

                if rename_value:

                    current = (
                        st.session_state
                        .chats[
                            st.session_state
                            .current_chat_id
                        ]
                    )

                    current["title"] = (
                        rename_value[:80]
                    )

                    save_chat(
                        current
                    )

                    st.rerun()

    # -----------------------------------------------------
    # Chat list
    # -----------------------------------------------------

    ordered = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1].get(
            "updated_at"
        ) or "",
        reverse=True,
    )

    for chat_id, item in ordered:

        q = search.lower().strip()

        matches = (
            not q
            or q in item["title"].lower()
            or any(
                q
                in str(
                    message.get(
                        "content",
                        "",
                    )
                ).lower()
                for message in item[
                    "messages"
                ]
            )
        )

        if not matches:

            continue

        c1, c2 = st.columns(
            [4, 1]
        )

        with c1:

            if st.button(
                f"💬 {item['title']}",
                key=f"open_{chat_id}",
                use_container_width=True,
            ):

                st.session_state.current_chat_id = (
                    chat_id
                )

                st.rerun()

        with c2:

            if st.button(
                "🗑️",
                key=f"del_{chat_id}",
                use_container_width=True,
            ):

                delete_chat(
                    chat_id
                )

                st.rerun()

    st.divider()

    # =====================================================
    # DOWNLOAD CHAT
    # Plain TXT = maximum device compatibility
    # =====================================================

    current_export = (
        st.session_state.chats.get(
            st.session_state.current_chat_id
        )
    )

    if current_export:

        export_lines = []

        export_lines.append(
            current_export["title"]
        )

        export_lines.append(
            "="
            * len(
                current_export["title"]
            )
        )

        export_lines.append("")

        for message in current_export.get(
            "messages",
            [],
        ):

            if message.get("role") == "user":

                role = "YOU"

            else:

                role = "RACHARLAGPT"

            export_lines.append(
                role
            )

            export_lines.append(
                "-" * len(role)
            )

            export_lines.append(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )

            export_lines.append("")

        export_text = "\n".join(
            export_lines
        )

        safe_filename = (
            current_export["title"]
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace("*", "-")
            .replace("?", "-")
            .replace('"', "-")
            .replace("<", "-")
            .replace(">", "-")
            .replace("|", "-")
            .strip()
        )

        if not safe_filename:

            safe_filename = "RacharlaGPT-chat"

        safe_filename = (
            safe_filename[:60]
        )

        st.download_button(
            "⬇️ Download Chat (.txt)",
            data=export_text,
            file_name=(
                safe_filename
                + ".txt"
            ),
            mime="text/plain; charset=utf-8",
            use_container_width=True,
            on_click="ignore",
        )

        st.caption(
            "TXT format opens on Android, iPhone, "
            "Windows, Mac and other devices."
        )

    # =====================================================
    # DELETE ALL
    # =====================================================

    if st.button(
        "🗑️ Delete All Chats",
        use_container_width=True,
    ):

        delete_all()

        st.rerun()

    st.divider()

    # =====================================================
    # AI SETTINGS
    # =====================================================

    with st.expander(
        "⚙️ AI Settings",
        expanded=False,
    ):

        st.session_state.selected_model = (
            st.selectbox(
                "Model",
                MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(
                    st.session_state.selected_model
                ),
            )
        )

        st.session_state.temperature = (
            st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.2,
                value=float(
                    st.session_state.temperature
                ),
                step=0.1,
                help=(
                    "Lower = more focused. "
                    "Higher = more creative."
                ),
            )
        )

        st.session_state.system_prompt = (
            st.text_area(
                "AI System Prompt",
                value=(
                    st.session_state.system_prompt
                ),
                height=180,
            )
        )

        if st.button(
            "↩️ Reset AI Settings",
            use_container_width=True,
        ):

            st.session_state.selected_model = (
                "Auto (recommended)"
            )

            st.session_state.temperature = 0.7

            st.session_state.system_prompt = (
                DEFAULT_SYSTEM_PROMPT
            )

            st.rerun()

    st.caption(
        f"Primary: {PRIMARY_MODEL}"
    )

    st.caption(
        f"Fallback: {BACKUP_MODEL}"
    )

    st.caption(
        "☁️ Chats stored in Supabase"
    )


# =========================================================
# MAIN UI
# =========================================================

st.markdown(
    """
    <div class="brand">
        <div class="icon">⚡</div>
        <div class="name">RacharlaGPT</div>
    </div>

    <div class="sub">
        Fast, intelligent AI conversations powered by Groq
    </div>
    """,
    unsafe_allow_html=True,
)


chat = current_chat()


# =========================================================
# EMPTY CHAT
# =========================================================

if not chat["messages"]:

    st.markdown(
        """
        <div class="welcome">

            <div style="font-size:48px">
                ⚡
            </div>

            <div class="wtitle">
                Welcome to RacharlaGPT
            </div>

            <div class="wtext">
                Your conversations are saved to your account.
                Ask questions, write code, brainstorm ideas,
                or learn something new.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DISPLAY MESSAGES
# =========================================================

for message in chat["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Message RacharlaGPT..."
)


if user_input:

    user_input = user_input.strip()

    if user_input:

        # -------------------------------------------------
        # Save user message immediately
        # -------------------------------------------------

        chat["messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        # -------------------------------------------------
        # Automatic title
        # -------------------------------------------------

        if chat["title"] == "New Chat":

            chat["title"] = title_for(
                user_input
            )

        # -------------------------------------------------
        # Display user message
        # -------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_input
            )

        # -------------------------------------------------
        # Save BEFORE contacting AI
        # -------------------------------------------------

        try:

            save_chat(chat)

        except Exception as save_error:

            st.warning(
                "Message was received, but "
                "saving to Supabase failed."
            )

            print(
                f"Save error: {save_error}"
            )

        # -------------------------------------------------
        # AI response
        # -------------------------------------------------

        try:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "RacharlaGPT is thinking…"
                ):

                    response, used = ask(
                        chat
                    )

                answer = response.content

                if not isinstance(
                    answer,
                    str,
                ):

                    answer = str(
                        answer
                    )

                answer = answer.strip()

                if not answer:

                    answer = (
                        "I couldn't generate a response "
                        "this time. Please try again."
                    )

                st.markdown(
                    answer
                )

                if used == "backup":

                    st.caption(
                        "Primary model was unavailable; "
                        f"answered with {BACKUP_MODEL}."
                    )

            # ---------------------------------------------
            # Save AI response
            # ---------------------------------------------

            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            try:

                save_chat(
                    chat
                )

            except Exception as save_error:

                st.warning(
                    "The response was generated, "
                    "but saving it to Supabase failed."
                )

                print(
                    f"Response save error: {save_error}"
                )

        # -------------------------------------------------
        # Rate limit
        # -------------------------------------------------

        except RateLimitError:

            with st.chat_message(
                "assistant"
            ):

                st.warning(
                    "Both Groq models are currently "
                    "rate-limited. Please wait a few minutes "
                    "and try again."
                )

            # Keep user's message.
            # It is NOT deleted.

        # -------------------------------------------------
        # Other error
        # -------------------------------------------------

        except Exception as exc:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "RacharlaGPT could not generate "
                    "a response this time."
                )

                st.caption(
                    "Your message has been kept in the chat. "
                    "Please try again."
                )

            print(
                f"RacharlaGPT error: "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9aa0ad;
        font-size:11px;
        margin-top:12px;
    ">
        RacharlaGPT • Powered by Groq • Chats stored in Supabase
    </div>
    """,
    unsafe_allow_html=True,
)
