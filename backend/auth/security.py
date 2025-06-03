"""
Enhanced authentication and security utilities for the LLM Evaluation Platform.
Supports JWT-based auth, RBAC, multi-tenancy, and API key authentication.
"""

import secrets
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from decouple import config
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from database.connection import get_db
from database.models import User
from .models import (
    Team, TeamInvitation, APIKey, UserRole, TeamTier, AuthContext,
    user_team_association
)


# Security configuration
SECRET_KEY = config("SECRET_KEY", default=secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
REFRESH_TOKEN_EXPIRE_DAYS = config("REFRESH_TOKEN_EXPIRE_DAYS", default=7, cast=int)
API_KEY_PREFIX = "llmeval"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security scheme
security = HTTPBearer()


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    team_role: Optional[str] = None
    scopes: List[str] = []


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# PASSWORD AND TOKEN UTILITIES
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None,
    token_type: str = "access"
) -> str:
    """Create a JWT token (access or refresh)."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        if token_type == "refresh":
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "type": token_type,
        "iat": datetime.utcnow()
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify a JWT token and return token data."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            return None
        
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        team_id: str = payload.get("team_id")
        team_role: str = payload.get("team_role")
        scopes: List[str] = payload.get("scopes", [])
        
        if email is None or user_id is None:
            return None
            
        return TokenData(
            email=email,
            user_id=user_id,
            team_id=team_id,
            team_role=team_role,
            scopes=scopes
        )
    except JWTError:
        return None


# ============================================================================
# API KEY UTILITIES
# ============================================================================

def generate_api_key() -> tuple[str, str]:
    """Generate an API key and return (full_key, hash)."""
    random_part = secrets.token_urlsafe(32)
    full_key = f"{API_KEY_PREFIX}_{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, key_hash


def verify_api_key(api_key: str) -> str:
    """Verify API key format and return hash."""
    if not api_key.startswith(f"{API_KEY_PREFIX}_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )
    
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_api_key_context(
    api_key_hash: str, 
    db: AsyncSession
) -> Optional[AuthContext]:
    """Get authentication context from API key."""
    try:
        # Get API key with relationships
        stmt = select(APIKey).options(
            selectinload(APIKey.user),
            selectinload(APIKey.team)
        ).where(
            and_(
                APIKey.key_hash == api_key_hash,
                APIKey.is_active == True
            )
        )
        
        result = await db.execute(stmt)
        api_key_obj = result.scalar_one_or_none()
        
        if not api_key_obj:
            return None
        
        # Update usage tracking
        api_key_obj.usage_count += 1
        api_key_obj.last_used_at = datetime.utcnow()
        await db.commit()
        
        # Parse scopes
        scopes = json.loads(api_key_obj.scopes) if api_key_obj.scopes else ["read"]
        
        return AuthContext(
            user_id=str(api_key_obj.user_id),
            email=api_key_obj.user.email,
            team_id=str(api_key_obj.team_id),
            team_role=UserRole.API_USER,
            scopes=scopes,
            is_api_key=True
        )
        
    except Exception:
        return None


# ============================================================================
# RBAC UTILITIES  
# ============================================================================

def check_permission(
    required_permission: str,
    user_role: UserRole,
    scopes: List[str] = None
) -> bool:
    """Check if user has required permission."""
    # Super admin has all permissions
    if user_role == UserRole.SUPER_ADMIN:
        return True
    
    # For API keys, check scopes
    if scopes:
        return required_permission in scopes
    
    # Role-based permissions
    permissions = {
        UserRole.TEAM_ADMIN: [
            "read", "write", "delete", "manage_team", "manage_users", 
            "manage_api_keys", "view_analytics"
        ],
        UserRole.EVALUATOR: [
            "read", "write", "create_evaluations", "view_own_data"
        ],
        UserRole.VIEWER: [
            "read", "view_own_data"
        ],
        UserRole.API_USER: []  # API users rely on scopes
    }
    
    user_permissions = permissions.get(user_role, [])
    return required_permission in user_permissions


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get auth context from kwargs (injected by auth dependency)
            auth_context = kwargs.get('auth_context')
            if not auth_context:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not check_permission(permission, auth_context.team_role, auth_context.scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# TEAM/TENANT UTILITIES
# ============================================================================

async def get_user_team_role(
    user_id: str, 
    team_id: str, 
    db: AsyncSession
) -> Optional[UserRole]:
    """Get user's role in a specific team."""
    try:
        stmt = select(user_team_association).where(
            and_(
                user_team_association.c.user_id == user_id,
                user_team_association.c.team_id == team_id,
                user_team_association.c.is_active == True
            )
        )
        
        result = await db.execute(stmt)
        row = result.first()
        
        if row:
            return UserRole(row.role)
        return None
        
    except Exception:
        return None


async def ensure_data_isolation(
    auth_context: AuthContext,
    resource_team_id: Optional[str] = None
) -> bool:
    """Ensure user can only access data from their team."""
    # Super admins can access all data
    if auth_context.team_role == UserRole.SUPER_ADMIN:
        return True
    
    # If no team context, deny access
    if not auth_context.team_id:
        return False
    
    # If resource has no team (legacy data), allow for now
    if not resource_team_id:
        return True
    
    # Check team isolation
    return auth_context.team_id == resource_team_id


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_auth_context(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> AuthContext:
    """Get current authentication context from JWT or API key."""
    token = credentials.credentials
    
    # Try API key first (format: llmeval_...)
    if token.startswith(f"{API_KEY_PREFIX}_"):
        api_key_hash = verify_api_key(token)
        auth_context = await get_api_key_context(api_key_hash, db)
        
        if not auth_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return auth_context
    
    # Try JWT token
    token_data = verify_token(token, "access")
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    try:
        stmt = select(User).where(
            and_(
                User.id == token_data.user_id,
                User.is_active == True
            )
        )
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Get team role if team context exists
        team_role = None
        if token_data.team_id:
            team_role = await get_user_team_role(
                token_data.user_id, 
                token_data.team_id, 
                db
            )
        
        return AuthContext(
            user_id=token_data.user_id,
            email=token_data.email,
            team_id=token_data.team_id,
            team_role=team_role or UserRole.VIEWER,
            scopes=token_data.scopes,
            is_api_key=False
        )
        
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


# Legacy compatibility
async def get_current_user_email(
    auth_context: AuthContext = Depends(get_current_auth_context)
) -> str:
    """Get current user email (legacy compatibility)."""
    return auth_context.email


# RBAC dependencies
def require_role(required_role: UserRole):
    """Dependency to require specific role."""
    async def check_role(
        auth_context: AuthContext = Depends(get_current_auth_context)
    ) -> AuthContext:
        # Super admin bypasses role checks
        if auth_context.team_role == UserRole.SUPER_ADMIN:
            return auth_context
        
        # Check role hierarchy
        role_hierarchy = {
            UserRole.SUPER_ADMIN: 5,
            UserRole.TEAM_ADMIN: 4,
            UserRole.EVALUATOR: 3,
            UserRole.VIEWER: 2,
            UserRole.API_USER: 1
        }
        
        user_level = role_hierarchy.get(auth_context.team_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required"
            )
        
        return auth_context
    
    return check_role


def require_scope(required_scope: str):
    """Dependency to require specific scope."""
    async def check_scope(
        auth_context: AuthContext = Depends(get_current_auth_context)
    ) -> AuthContext:
        if not check_permission(required_scope, auth_context.team_role, auth_context.scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required"
            )
        
        return auth_context
    
    return check_scope


# Team isolation dependency
def ensure_team_access():
    """Dependency to ensure team-level data isolation."""
    async def check_team_access(
        request: Request,
        auth_context: AuthContext = Depends(get_current_auth_context)
    ) -> AuthContext:
        # Extract team_id from path parameters if present
        path_params = request.path_params
        resource_team_id = path_params.get('team_id')
        
        if not await ensure_data_isolation(auth_context, resource_team_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient permissions for this team"
            )
        
        return auth_context
    
    return check_team_access 