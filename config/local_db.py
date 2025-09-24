"""
Local Database Configuration
Manages local SQLite database for user credentials and chat history using SQLAlchemy ORM.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError
import logging
import secrets
import hashlib

logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_chatbot.db')}"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    messages = relationship("ChatMessage", back_populates="user")

class ChatSession(Base):
    """Chat session model."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    query = Column(Text, nullable=True)  # SQL query if applicable
    results = Column(Text, nullable=True)  # JSON string of query results
    error = Column(Boolean, default=False)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User", back_populates="messages")

class UserPreference(Base):
    """User preferences model."""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    preference_key = Column(String, nullable=False)
    preference_value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

class LocalDatabase:
    """SQLite database manager for local user data and chat history using SQLAlchemy."""

    def __init__(self):
        pass

    def get_db(self) -> Session:
        """Get database session."""
        return SessionLocal()

    # User Management Methods
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Create a new user account."""
        try:
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)

            with self.get_db() as db:
                user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    salt=salt
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Created user: {username} (ID: {user.id})")
                return user.id

        except IntegrityError:
            logger.error(f"User creation failed: {username} or {email} already exists")
            return None

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user and return user data if successful."""
        with self.get_db() as db:
            user = db.query(User).filter(
                ((User.username == username_or_email) | (User.email == username_or_email)) &
                (User.is_active == True)
            ).first()

            if user and self._verify_password(password, user.password_hash, user.salt):
                # Update last login
                user.last_login = datetime.utcnow()
                db.commit()

                return {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_active': user.is_active
                }
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user data by ID."""
        with self.get_db() as db:
            user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
            if user:
                return {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'is_active': user.is_active
                }
            return None

    def update_user_preference(self, user_id: int, key: str, value: str):
        """Update or create a user preference."""
        with self.get_db() as db:
            preference = db.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.preference_key == key
            ).first()

            if preference:
                preference.preference_value = value
                preference.updated_at = datetime.utcnow()
            else:
                preference = UserPreference(
                    user_id=user_id,
                    preference_key=key,
                    preference_value=value
                )
                db.add(preference)

            db.commit()

    def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a user preference value."""
        with self.get_db() as db:
            preference = db.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.preference_key == key
            ).first()

            return preference.preference_value if preference else None

    # Chat Management Methods
    def create_chat_session(self, user_id: int, session_name: str = None) -> int:
        """Create a new chat session for a user."""
        with self.get_db() as db:
            session = ChatSession(
                user_id=user_id,
                session_name=session_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id

    def save_message(self, session_id: int, user_id: int, role: str, content: str,
                    query: str = None, results: List[Dict] = None, error: bool = False):
        """Save a chat message to the database."""
        import json

        with self.get_db() as db:
            # Convert results to JSON if provided
            results_json = json.dumps(results) if results else None

            message = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                query=query,
                results=results_json,
                error=error
            )
            db.add(message)

            # Update session's last message time and message count
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.last_message_at = datetime.utcnow()
                session.message_count += 1

            db.commit()

    def get_chat_history(self, session_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        import json

        with self.get_db() as db:
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp.asc()).limit(limit).all()

            result = []
            for msg in messages:
                message_data = {
                    'id': msg.id,
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else None,
                    'query': msg.query,
                    'error': msg.error
                }

                # Parse results JSON if present
                if msg.results:
                    try:
                        message_data['results'] = json.loads(msg.results)
                    except:
                        message_data['results'] = None
                else:
                    message_data['results'] = None

                result.append(message_data)

            return result

    def get_user_sessions(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat sessions for a user."""
        with self.get_db() as db:
            sessions = db.query(ChatSession).filter(
                ChatSession.user_id == user_id
            ).order_by(ChatSession.last_message_at.desc()).limit(limit).all()

            return [{
                'id': s.id,
                'session_name': s.session_name,
                'created_at': s.created_at.isoformat() if s.created_at else None,
                'last_message_at': s.last_message_at.isoformat() if s.last_message_at else None,
                'message_count': s.message_count
            } for s in sessions]

    def delete_chat_session(self, session_id: int, user_id: int) -> bool:
        """Delete a chat session (only if owned by user)."""
        with self.get_db() as db:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()

            if session:
                db.delete(session)
                db.commit()
                return True
            return False

    # Utility Methods
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt using SHA-256."""
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    def _verify_password(self, password: str, hash: str, salt: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(password, salt) == hash

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_db() as db:
            active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
            total_sessions = db.query(func.count(ChatSession.id)).scalar()
            total_messages = db.query(func.count(ChatMessage.id)).scalar()

            # Messages in last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)
            messages_last_24h = db.query(func.count(ChatMessage.id)).filter(
                ChatMessage.timestamp >= yesterday
            ).scalar()

            return {
                'active_users': active_users or 0,
                'total_sessions': total_sessions or 0,
                'total_messages': total_messages or 0,
                'messages_last_24h': messages_last_24h or 0
            }

# Global instance
local_db = LocalDatabase()