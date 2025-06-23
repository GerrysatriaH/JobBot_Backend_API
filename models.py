from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum

class SenderEnum(str, enum.Enum):
    user = "user"
    bot = "bot"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    new_password_temp = Column(String, nullable=True)

    chats = relationship(
        "HistoryChat",
        back_populates="user",
        cascade="all, delete-orphan"
    )

class HistoryChat(Base):
    __tablename__ = "history_chat"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey(
        "users.id", ondelete="CASCADE", onupdate="CASCADE"
    ))
    created_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="chats")
    messages = relationship(
        "MessagesChat",
        back_populates="chat",
        cascade="all, delete-orphan"
    )

class MessagesChat(Base):
    __tablename__ = "messages_chat"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey(
        "history_chat.id", ondelete="CASCADE", onupdate="CASCADE"
    ))
    sender = Column(Enum(SenderEnum))
    message = Column(Text)
    is_file = Column(Boolean, default=False)
    file_name = Column(String(255), nullable=True)
    file_url = Column(Text, nullable=True)
    timestamp = Column(TIMESTAMP, server_default=func.now())

    chat = relationship("HistoryChat", back_populates="messages")
