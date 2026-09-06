# ============================================================
# SUPABASE CLIENT SETUP (FIXED OAUTH STORAGE)
# ============================================================

if "supabase_client" not in st.session_state:
    # Use standard client settings for web/Streamlit flow
    st.session_state.supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

supabase: Client = st.session_state.supabase_client


# ============================================================
# SESSION PERSISTENCE & GOOGLE OAUTH CALLBACK FIX
# ============================================================

def restore_session():
    """Restore an existing session or complete the Google OAuth flow."""
    if "auth_user" in st.session_state and st.session_state.auth_user:
        return

    params = st.query_params
    if params.get("logged_out") == "1":
        return

    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")
    code = params.get("code")

    # Handle standard Code exchange fallback
    if code:
        try:
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            if res.user and res.session:
                st.session_state.auth_user = res.user
                st.query_params["access_token"] = res.session.access_token
                st.query_params["refresh_token"] = res.session.refresh_token
                st.query_params.pop("code", None)
                st.query_params.pop("oauth_flow", None)
                st.query_params.pop("logged_out", None)
                st.session_state.pop("google_oauth_error", None)
                st.session_state.pop("google_oauth_url", None)
                st.rerun()
                return
        except Exception as exc:
            st.session_state.google_oauth_error = str(exc)

    # Restore via access/refresh tokens
    if access_token and refresh_token:
        try:
            res = supabase.auth.set_session(access_token, refresh_token)
            if res.user:
                st.session_state.auth_user = res.user
                st.query_params.pop("logged_out", None)
                return
        except Exception as exc:
            st.session_state.google_oauth_error = str(exc)

    # Fallback session check
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.auth_user = session.user
            if getattr(session, "access_token", None):
                st.query_params["access_token"] = session.access_token
            if getattr(session, "refresh_token", None):
                st.query_params["refresh_token"] = session.refresh_token
            st.query_params.pop("logged_out", None)
    except Exception:
        pass


restore_session()
