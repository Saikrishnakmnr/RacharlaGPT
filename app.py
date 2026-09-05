def show_auth():

    st.markdown(
        "<div style='text-align:center; font-size:52px;'>⚡</div>",
        unsafe_allow_html=True,
    )

    st.title("Welcome to RacharlaGPT")

    st.markdown(
        "Sign in to save your conversations permanently."
    )

    st.write("")

    _, center, _ = st.columns([1, 2, 1])

    with center:

        login_tab, signup_tab = st.tabs(
            ["🔐 Sign In", "✨ Create Account"]
        )

        # ====================================================
        # SIGN IN
        # ====================================================

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

                email = email.strip()

                if not email or not password:

                    st.warning(
                        "Please enter your email and password."
                    )

                else:

                    try:

                        result = (
                            supabase.auth.sign_in_with_password(
                                {
                                    "email": email,
                                    "password": password,
                                }
                            )
                        )

                        if (
                            result.user
                            and result.session
                        ):

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.success(
                                "Signed in successfully!"
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Sign in failed. "
                                "Supabase did not return a session."
                            )

                    except Exception as exc:

                        error_text = str(exc)

                        st.error(
                            f"Sign in failed: {error_text}"
                        )

        # ====================================================
        # CREATE ACCOUNT
        # ====================================================

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

                email = email.strip()

                if not email or not password:

                    st.warning(
                        "Please enter your email and password."
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
                                "email": email,
                                "password": password,
                            }
                        )

                        if (
                            result.user
                            and result.session
                        ):

                            st.session_state.auth_user = (
                                result.user
                            )

                            st.success(
                                "Account created successfully!"
                            )

                            st.rerun()

                        elif result.user:

                            st.success(
                                "Account created. "
                                "You can now sign in."
                            )

                        else:

                            st.error(
                                "Account creation failed."
                            )

                    except Exception as exc:

                        st.error(
                            f"Account creation failed: {exc}"
                        )
