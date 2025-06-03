"""
Authentication API endpoints for user management, team operations, and API keys.
Provides registration, login, team management, and API key CRUD operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from auth.models import (
    UserRole, TeamCreate, TeamResponse, TeamInvitationCreate, 
    TeamInvitationResponse, APIKeyCreate, APIKeyCreateResponse, 
    APIKeyResponse, UserProfile, AuthContext
)
from auth.security import (
    UserCreate, UserLogin, UserResponse, Token,
    get_current_auth_context, require_role, require_scope,
    ensure_team_access, security
)
from auth.service import AuthService

router = APIRouter()

# ============================================================================
# USER AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user with automatic team creation."""
    try:
        user, team = await AuthService.register_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            db=db,
            create_default_team=True
        )
        
        # Create tokens with team context
        tokens = await AuthService.create_user_tokens(
            user=user,
            team_id=str(team.id) if team else None,
            db=db
        )
        
        return {
            "message": "User registered successfully",
            "user": UserResponse(
                id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_admin=user.is_admin,
                created_at=user.created_at
            ),
            "team": TeamResponse(
                id=str(team.id),
                name=team.name,
                slug=team.slug,
                description=team.description,
                tier=team.tier,
                is_active=team.is_active,
                max_users=team.max_users,
                max_evaluations_per_month=team.max_evaluations_per_month,
                member_count=1,
                created_at=team.created_at
            ) if team else None,
            "tokens": tokens
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login_user(
    user_data: UserLogin,
    team_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT tokens."""
    try:
        # Authenticate user
        user = await AuthService.authenticate_user(
            email=user_data.email,
            password=user_data.password,
            db=db
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Create tokens
        tokens = await AuthService.create_user_tokens(
            user=user,
            team_id=team_id,
            db=db
        )
        
        return Token(**tokens)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    auth_context: AuthContext = Depends(get_current_auth_context),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's profile with team information."""
    return await AuthService.get_user_profile(auth_context.user_id, db)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    from auth.security import verify_token
    
    # Verify refresh token
    token_data = verify_token(credentials.credentials, "refresh")
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    from database.models import User
    from sqlalchemy import select, and_
    
    stmt = select(User).where(
        and_(User.id == token_data.user_id, User.is_active == True)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new tokens
    tokens = await AuthService.create_user_tokens(
        user=user,
        team_id=token_data.team_id,
        db=db
    )
    
    return Token(**tokens)


# ============================================================================
# TEAM MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/teams", response_model=TeamResponse)
async def create_team(
    team_data: TeamCreate,
    auth_context: AuthContext = Depends(require_role(UserRole.EVALUATOR)),
    db: AsyncSession = Depends(get_db)
):
    """Create a new team."""
    return await AuthService.create_team(
        team_data=team_data,
        creator_id=auth_context.user_id,
        db=db
    )


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    auth_context: AuthContext = Depends(ensure_team_access()),
    db: AsyncSession = Depends(get_db)
):
    """Get team details."""
    from auth.models import Team
    from sqlalchemy import select
    
    stmt = select(Team).where(Team.id == team_id)
    result = await db.execute(stmt)
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found"
        )
    
    # Get member count
    from sqlalchemy import func
    from auth.models import user_team_association
    
    member_count_stmt = select(func.count()).select_from(
        user_team_association
    ).where(user_team_association.c.team_id == team_id)
    
    member_count_result = await db.execute(member_count_stmt)
    member_count = member_count_result.scalar()
    
    return TeamResponse(
        id=str(team.id),
        name=team.name,
        slug=team.slug,
        description=team.description,
        tier=team.tier,
        is_active=team.is_active,
        max_users=team.max_users,
        max_evaluations_per_month=team.max_evaluations_per_month,
        member_count=member_count,
        created_at=team.created_at
    )


@router.get("/teams/{team_id}/members")
async def get_team_members(
    team_id: str,
    auth_context: AuthContext = Depends(ensure_team_access()),
    db: AsyncSession = Depends(get_db)
):
    """Get team members with their roles."""
    return await AuthService.get_team_members(team_id, db)


@router.post("/teams/{team_id}/invitations", response_model=TeamInvitationResponse)
async def create_team_invitation(
    team_id: str,
    invitation_data: TeamInvitationCreate,
    auth_context: AuthContext = Depends(require_role(UserRole.TEAM_ADMIN)),
    db: AsyncSession = Depends(get_db)
):
    """Create a team invitation."""
    # Ensure user can manage this team
    if auth_context.team_id != team_id and auth_context.team_role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot invite users to this team"
        )
    
    return await AuthService.create_team_invitation(
        team_id=team_id,
        invitation_data=invitation_data,
        inviter_id=auth_context.user_id,
        db=db
    )


@router.get("/teams/{team_id}/invitations")
async def list_team_invitations(
    team_id: str,
    auth_context: AuthContext = Depends(require_role(UserRole.TEAM_ADMIN)),
    db: AsyncSession = Depends(get_db)
):
    """List pending team invitations."""
    from auth.models import TeamInvitation
    from sqlalchemy import select, and_
    
    stmt = select(TeamInvitation).where(
        and_(
            TeamInvitation.team_id == team_id,
            TeamInvitation.is_used == False,
            TeamInvitation.expires_at > datetime.utcnow()
        )
    ).order_by(TeamInvitation.invited_at.desc())
    
    result = await db.execute(stmt)
    invitations = result.scalars().all()
    
    return [
        TeamInvitationResponse(
            id=str(inv.id),
            email=inv.email,
            role=inv.role,
            invited_at=inv.invited_at,
            expires_at=inv.expires_at,
            is_used=inv.is_used,
            team_name="",  # Could load team name if needed
            inviter_email=""  # Could load inviter email if needed
        )
        for inv in invitations
    ]


@router.post("/invitations/{invitation_token}/accept")
async def accept_team_invitation(
    invitation_token: str,
    auth_context: AuthContext = Depends(get_current_auth_context),
    db: AsyncSession = Depends(get_db)
):
    """Accept a team invitation."""
    from auth.models import TeamInvitation
    from sqlalchemy import select, and_
    
    # Get invitation
    stmt = select(TeamInvitation).where(
        and_(
            TeamInvitation.token == invitation_token,
            TeamInvitation.is_used == False,
            TeamInvitation.expires_at > datetime.utcnow()
        )
    )
    result = await db.execute(stmt)
    invitation = result.scalar_one_or_none()
    
    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired invitation"
        )
    
    # Check if invitation is for current user
    if invitation.email != auth_context.email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This invitation is not for you"
        )
    
    # Add user to team
    success = await AuthService.add_user_to_team(
        user_id=auth_context.user_id,
        team_id=str(invitation.team_id),
        role=UserRole(invitation.role),
        invited_by=str(invitation.invited_by),
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add user to team"
        )
    
    # Mark invitation as used
    invitation.is_used = True
    invitation.accepted_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Invitation accepted successfully"}


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/teams/{team_id}/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    team_id: str,
    key_data: APIKeyCreate,
    auth_context: AuthContext = Depends(require_role(UserRole.EVALUATOR)),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key for the team."""
    # Ensure user can create API keys for this team
    if auth_context.team_id != team_id and auth_context.team_role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create API keys for this team"
        )
    
    return await AuthService.create_api_key(
        key_data=key_data,
        user_id=auth_context.user_id,
        team_id=team_id,
        db=db
    )


@router.get("/teams/{team_id}/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    team_id: str,
    auth_context: AuthContext = Depends(require_role(UserRole.VIEWER)),
    db: AsyncSession = Depends(get_db)
):
    """List API keys for the team."""
    # Ensure user can view API keys for this team
    if auth_context.team_id != team_id and auth_context.team_role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view API keys for this team"
        )
    
    # Non-admins can only see their own keys
    user_id = None if auth_context.team_role in [UserRole.TEAM_ADMIN, UserRole.SUPER_ADMIN] else auth_context.user_id
    
    return await AuthService.list_api_keys(
        team_id=team_id,
        user_id=user_id,
        db=db
    )


@router.delete("/teams/{team_id}/api-keys/{key_id}")
async def revoke_api_key(
    team_id: str,
    key_id: str,
    auth_context: AuthContext = Depends(require_role(UserRole.EVALUATOR)),
    db: AsyncSession = Depends(get_db)
):
    """Revoke an API key."""
    # Ensure user can manage API keys for this team
    if auth_context.team_id != team_id and auth_context.team_role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot revoke API keys for this team"
        )
    
    # Check if key belongs to user (if not admin)
    if auth_context.team_role not in [UserRole.TEAM_ADMIN, UserRole.SUPER_ADMIN]:
        from auth.models import APIKey
        from sqlalchemy import select, and_
        
        stmt = select(APIKey).where(
            and_(
                APIKey.id == key_id,
                APIKey.user_id == auth_context.user_id,
                APIKey.team_id == team_id
            )
        )
        result = await db.execute(stmt)
        key = result.scalar_one_or_none()
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or not owned by you"
            )
    
    success = await AuthService.revoke_api_key(key_id, team_id, db)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return {"message": "API key revoked successfully"}


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/check-auth")
async def check_authentication(
    auth_context: AuthContext = Depends(get_current_auth_context)
):
    """Check authentication status and return user context."""
    return {
        "authenticated": True,
        "user_id": auth_context.user_id,
        "email": auth_context.email,
        "team_id": auth_context.team_id,
        "role": auth_context.team_role.value if auth_context.team_role else None,
        "scopes": auth_context.scopes,
        "is_api_key": auth_context.is_api_key
    }


@router.get("/permissions")
async def get_user_permissions(
    auth_context: AuthContext = Depends(get_current_auth_context)
):
    """Get user's permissions based on role and scopes."""
    from auth.security import check_permission
    
    # Define all possible permissions
    all_permissions = [
        "read", "write", "delete", "manage_team", "manage_users", 
        "manage_api_keys", "view_analytics", "create_evaluations", "view_own_data"
    ]
    
    permissions = {
        permission: check_permission(permission, auth_context.team_role, auth_context.scopes)
        for permission in all_permissions
    }
    
    return {
        "role": auth_context.team_role.value if auth_context.team_role else None,
        "scopes": auth_context.scopes,
        "permissions": permissions
    } 