import streamlit as st
from supabase import create_client, Client

# Page Configuration
st.set_page_config(
    page_title="RacharlaGPT",
    page_icon="🤖",
    layout="wide"
)

# Read Secrets / Config
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

# Initialize Supabase Client in Session State
if "supabase_client" not in st.session_state:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Missing Supabase credentials in .streamlit/secrets.toml")
        st.stop()
    st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = st.session_state.supabase_client

# Initialize User Auth State
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


def restore_session():
    """Restore an existing session or handle OAuth callback tokens."""
    if st.session_state.auth_user:
        return

    params = st.query_params
    if params.get("logged_out") == "1":
        return

    code = params.get("code")
    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")

    # Handle standard PKCE code exchange
    if code:
        try:
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            if res.user:
                st.session_state.auth_user = res.user
                st.query_params.clear()
                st.rerun()
                return
        except Exception as exc:
            st.session_state.google_oauth_error = str(exc)

    # Handle direct access/refresh token restoration
    if access_token and refresh_token:
        try:
            res = supabase.auth.set_session(access_token, refresh_token)
            if res.user:
                st.session_state.auth_user = res.user
                return
        except Exception as exc:
            st.session_state.google_oauth_error = str(exc)

    # Check for active session in memory
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.auth_user = session.user
    except Exception:
        pass


restore_session()


def login_with_google():
    """Trigger Google OAuth login flow."""
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": "https://racharlagpt.streamlit.app/"
            }
        })
        if res.url:
            st.session_state.google_oauth_url = res.url
            st.rerun()
    except Exception as exc:
        st.error(f"Failed to initiate login: {exc}")


def logout():
    """Clear session state and query params on logout."""
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
# APP UI / MAIN ROUTING
# ============================================================

user = st.session_state.auth_user

if not user:
    st.title("Welcome to RacharlaGPT")
    st.write("Please sign in to continue.")

    # Show OAuth Error if present
    if "google_oauth_error" in st.session_state:
        st.error(f"Authentication Error: {st.session_state.google_oauth_error}")
        st.session_state.pop("google_oauth_error")

    # Auto-redirect trigger or manually click button
    if "google_oauth_url" in st.session_state:
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={st.session_state.google_oauth_url}">',
            unsafe_allow_html=True
        )
        st.info("Redirecting to Google Sign-In...")
        if st.button("Click here if not redirected automatically"):
            st.markdown(f"[Login with Google]({st.session_state.google_oauth_url})")
    else:
        st.button("Sign in with Google", on_click=login_with_google, type="primary")

else:
    # Sidebar User Profile
    with st.sidebar:
        user_email = getattr(user, "email", "User")
        st.write(f"Logged in as: **{user_email}**")
        st.button("Log out", on_click=logout)

    # Main App Logic
    st.title("RacharlaGPT Dashboard")
    st.success(f"Welcome back, {user.email}!")
    
    st.write("---")
    prompt = st.chat_input("Ask RacharlaGPT anything...")
    if prompt:
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(f"Echo response to: '{prompt}'")
