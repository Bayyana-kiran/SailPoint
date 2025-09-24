"""
Authentication Manager
Handles user authentication, sessions, and security for the local database.
"""

import streamlit as st
import time
from typing import Optional, Dict, Any
from config.local_db import local_db
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages user authentication and sessions."""

    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        """Initialize authentication-related session state."""
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = 'login'  # 'login' or 'signup'

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return st.session_state.user is not None

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user data."""
        return st.session_state.user

    def login(self, username_or_email: str, password: str) -> bool:
        """Attempt to log in a user."""
        user = local_db.authenticate_user(username_or_email, password)
        if user:
            st.session_state.user = user
            # Create or get current chat session
            self._ensure_chat_session()
            logger.info(f"User logged in: {user['username']}")
            return True
        return False

    def signup(self, username: str, email: str, password: str) -> tuple[bool, str]:
        """Create a new user account."""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"

        if len(username) < 3:
            return False, "Username must be at least 3 characters long"

        user_id = local_db.create_user(username, email, password)
        if user_id:
            # Auto-login after signup
            user = local_db.get_user_by_id(user_id)
            if user:
                st.session_state.user = user
                self._ensure_chat_session()
                logger.info(f"User signed up: {username}")
                return True, "Account created successfully!"
        return False, "Username or email already exists"

    def logout(self):
        """Log out the current user."""
        if st.session_state.user:
            logger.info(f"User logged out: {st.session_state.user['username']}")
        st.session_state.user = None
        st.session_state.current_session_id = None
        # Clear other session data
        if 'messages' in st.session_state:
            st.session_state.messages = []
        if 'query_history' in st.session_state:
            st.session_state.query_history = []

    def _ensure_chat_session(self):
        """Ensure the user has an active chat session."""
        if not self.is_authenticated():
            return

        user_id = st.session_state.user['id']

        # If no current session, create one
        if st.session_state.current_session_id is None:
            session_id = local_db.create_chat_session(user_id)
            st.session_state.current_session_id = session_id
        else:
            # Verify the session exists and belongs to the user
            sessions = local_db.get_user_sessions(user_id, 1)
            if not sessions or sessions[0]['id'] != st.session_state.current_session_id:
                # Session doesn't exist or doesn't belong to user, create new one
                session_id = local_db.create_chat_session(user_id)
                st.session_state.current_session_id = session_id

    def save_message(self, role: str, content: str, query: str = None,
                    results: list = None, error: bool = False):
        """Save a message to the current chat session."""
        if not self.is_authenticated() or st.session_state.current_session_id is None:
            return

        local_db.save_message(
            session_id=st.session_state.current_session_id,
            user_id=st.session_state.user['id'],
            role=role,
            content=content,
            query=query,
            results=results,
            error=error
        )

    def load_chat_history(self, limit: int = 50) -> list:
        """Load chat history for the current session."""
        if not self.is_authenticated() or st.session_state.current_session_id is None:
            return []

        return local_db.get_chat_history(st.session_state.current_session_id, limit)

    def get_user_sessions(self, limit: int = 10) -> list:
        """Get user's chat sessions."""
        if not self.is_authenticated():
            return []

        return local_db.get_user_sessions(st.session_state.user['id'], limit)

    def switch_session(self, session_id: int):
        """Switch to a different chat session."""
        if not self.is_authenticated():
            return False

        # Verify the session belongs to the user
        sessions = local_db.get_user_sessions(st.session_state.user['id'])
        session_ids = [s['id'] for s in sessions]

        if session_id in session_ids:
            st.session_state.current_session_id = session_id
            return True
        return False

    def create_new_session(self, session_name: str = None) -> int:
        """Create a new chat session for the current user."""
        if not self.is_authenticated():
            return None

        session_id = local_db.create_chat_session(st.session_state.user['id'], session_name)
        st.session_state.current_session_id = session_id
        return session_id

    def delete_session(self, session_id: int) -> bool:
        """Delete a chat session."""
        if not self.is_authenticated():
            return False

        return local_db.delete_chat_session(session_id, st.session_state.user['id'])

    def get_user_preference(self, key: str, default: str = None) -> str:
        """Get a user preference."""
        if not self.is_authenticated():
            return default

        return local_db.get_user_preference(st.session_state.user['id'], key) or default

    def set_user_preference(self, key: str, value: str):
        """Set a user preference."""
        if not self.is_authenticated():
            return

        local_db.update_user_preference(st.session_state.user['id'], key, value)

# Global auth manager instance
auth_manager = AuthManager()