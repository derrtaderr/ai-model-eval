"""
Enhanced authentication models for multi-tenancy and RBAC.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Table, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from database.models import Base


class UserRole(str, Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"      # Platform administrator
    TEAM_ADMIN = "team_admin"        # Team/Organization administrator
    EVALUATOR = "evaluator"          # Can create and manage evaluations
    VIEWER = "viewer"                # Read-only access
    API_USER = "api_user"            # API-only access


class TeamTier(str, Enum):
    """Team subscription tiers."""
    FREE = "free"
    PROFESSIONAL = "professional" 
    ENTERPRISE = "enterprise"


# Association table for many-to-many relationship between users and teams
user_team_association = Table(
    'user_teams',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('team_id', UUID(as_uuid=True), ForeignKey('teams.id'), primary_key=True),
    Column('role', String(50), nullable=False, default=UserRole.VIEWER),
    Column('joined_at', DateTime, default=datetime.utcnow),
    Column('invited_by', UUID(as_uuid=True), ForeignKey('users.id'), nullable=True),
    Column('is_active', Boolean, default=True)
)


class Team(Base):
    """Team/Organization model for multi-tenancy."""
    __tablename__ = "teams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    tier = Column(String(50), default=TeamTier.FREE, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Subscription info
    max_users = Column(Integer, default=5)  # Based on tier
    max_evaluations_per_month = Column(Integer, default=1000)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    members = relationship("User", secondary=user_team_association, back_populates="teams")
    invitations = relationship("TeamInvitation", back_populates="team", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="team", cascade="all, delete-orphan")


class TeamInvitation(Base):
    """Team invitations for user onboarding."""
    __tablename__ = "team_invitations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(50), default=UserRole.VIEWER, nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    
    # Status tracking
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    invited_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    is_used = Column(Boolean, default=False)
    
    # Relationships
    team = relationship("Team", back_populates="invitations")
    inviter = relationship("User", foreign_keys=[invited_by])


class APIKey(Base):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for identification
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Permissions
    scopes = Column(Text, nullable=False, default="read")  # JSON array of scopes
    
    # Status and limits
    is_active = Column(Boolean, default=True)
    rate_limit_per_hour = Column(Integer, default=1000)
    
    # Usage tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User")
    team = relationship("Team", back_populates="api_keys")


# Pydantic models for API

class TeamCreate(BaseModel):
    """Schema for creating a new team."""
    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., min_length=2, max_length=100, regex=r'^[a-z0-9-]+$')
    description: Optional[str] = Field(None, max_length=1000)
    
    @validator('slug')
    def slug_must_be_lowercase(cls, v):
        return v.lower()


class TeamResponse(BaseModel):
    """Schema for team response."""
    id: str
    name: str
    slug: str
    description: Optional[str]
    tier: str
    is_active: bool
    max_users: int
    max_evaluations_per_month: int
    member_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserTeamRole(BaseModel):
    """Schema for user's role in a team."""
    user_id: str
    team_id: str
    role: UserRole
    joined_at: datetime
    is_active: bool


class TeamInvitationCreate(BaseModel):
    """Schema for creating team invitations."""
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    role: UserRole = Field(default=UserRole.VIEWER)


class TeamInvitationResponse(BaseModel):
    """Schema for team invitation response."""
    id: str
    email: str
    role: str
    invited_at: datetime
    expires_at: datetime
    is_used: bool
    team_name: str
    inviter_email: str
    
    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    """Schema for creating API keys."""
    name: str = Field(..., min_length=1, max_length=255)
    scopes: List[str] = Field(default=["read"])
    rate_limit_per_hour: int = Field(default=1000, ge=100, le=10000)


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    id: str
    name: str
    key_prefix: str
    scopes: List[str]
    is_active: bool
    rate_limit_per_hour: int
    created_at: datetime
    last_used_at: Optional[datetime]
    usage_count: int
    
    class Config:
        from_attributes = True


class APIKeyCreateResponse(BaseModel):
    """Schema for API key creation response (includes full key)."""
    api_key: APIKeyResponse
    secret_key: str  # Only returned once during creation


class UserProfile(BaseModel):
    """Enhanced user profile schema."""
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    teams: List[TeamResponse]
    current_team_role: Optional[UserRole]
    
    class Config:
        from_attributes = True


class AuthContext(BaseModel):
    """Authentication context for requests."""
    user_id: str
    email: str
    team_id: Optional[str] = None
    team_role: Optional[UserRole] = None
    scopes: List[str] = Field(default_factory=list)
    is_api_key: bool = False 