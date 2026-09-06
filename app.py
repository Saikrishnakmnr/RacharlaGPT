import os
import streamlit as st
from supabase import create_client, Client, ClientOptions

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# CREDENTIALS & SUPABASE CLIENT INITIALIZATION
# ============================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

if "supabase_client" not in st.session_state:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Missing SUPABASE_URL or SUPABASE_ANON_KEY in Streamlit Secrets.")
        st.stop()
    
    # Initialize Supabase Client
    st.session_state.supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

supabase: Client = st.session_state.supabase_client

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ============================================================
# ISSUE RESOLVED BLOCK: SESSION RESTORATION & OAUTH EXCHANGE
# ============================================================
def restore_session():
    """Handles OAuth callback, exchanges codes safely, and keeps user logged in."""
    if st.session_state.get("auth_user"):
        return

    params = st.query_params
    if params.get("logged_out") == "1":
        return

    code = params.get("code")
    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")

    # 1. Direct restore via access/refresh tokens
    if access_token and refresh_token:
        try:
            res = supabase.auth.set_session(access_token, refresh_token)
            if res and res.user:
                st.session_state.auth_user = res.user
                return
        except Exception as exc:
            st.session_state.google_oauth_error = str(exc)

    # 2. PKCE Code Exchange with exception handler to avoid app crashes
    if code:
        try:
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            if res and res.user:
                st.session_state.auth_user = res.user
                # Clean up query params after successful login
                st.query_params.clear()
                st.session_state.pop("google_oauth_error", None)
                st.session_state.pop("google_oauth_url", None)
                st.rerun()
                return
        except Exception as exc:
            # Clear bad code param so the error screen disappears on fresh attempts
            st.session_state.google_oauth_error = str(exc)
            st.query_params.pop("code", None)

    # 3. Active session check fallback
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.auth_user = session.user
    except Exception:
        pass


restore_session()


# ============================================================
# ISSUE RESOLVED BLOCK: GOOGLE OAUTH TRIGGER & LOGOUT
# ============================================================
def login_with_google():
    """Trigger Google OAuth login flow with clean redirect configuration."""
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": "https://racharlagpt.streamlit.app/",
                "queryParams": {
                    "prompt": "select_account"
                }
            }
        })
        if res.url:
            st.session_state.google_oauth_url = res.url
            st.rerun()
    except Exception as exc:
        st.error(f"Failed to initiate login: {exc}")


def logout():
    """Clear user session and update query parameters cleanly."""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.auth_user = None
    st.session_state.pop("google_oauth_url", None)
    st.query_params.clear()
    st.query_params["logged_out"] = "1"
    st.rerun()


# ============================================================
# APPLICATION UI ROUTING
# ============================================================
user = st.session_state.auth_user

if not user:
    st.title("Welcome back 👋")
    st.caption("Sign in to save and access your conversations permanently.")

    # Show Diagnostic / OAuth Error if trigger fails
    if "google_oauth_error" in st.session_state:
        with st.expander("Google sign-in diagnostic", expanded=True):
            st.error(st.session_state.google_oauth_error)
            if st.button("Try Google sign-in again"):
                st.session_state.pop("google_oauth_error", None)
                st.rerun()

    # OAuth Automatic or Manual Redirect
    if "google_oauth_url" in st.session_state:
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={st.session_state.google_oauth_url}">',
            unsafe_allow_html=True
        )
        st.info("Redirecting to Google Sign-In...")
        st.markdown(f"[Click here if not redirected automatically]({st.session_state.google_oauth_url})")
    else:
        st.button("Continue with Google (Gmail)", on_click=login_with_google, type="primary")

else:
    # Sidebar Setup
    with st.sidebar:
        st.title("RacharlaGPT")
        st.write(f"Logged in as: **{getattr(user, 'email', 'User')}**")
        st.button("Log out", on_click=logout)

    # Main Chat View
    st.title("RacharlaGPT")
    st.write("A practical AI assistant for learning, coding, ideas, research, and everyday questions.")
    
    prompt = st.chat_input("Ask RacharlaGPT anything...")
    if prompt:
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(f"Echo response to: {prompt}")
