"""
Authentication service for user registration, team management, and API operations.
Handles user lifecycle, team invitations, and API key management.
"""

import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update, delete
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status

from database.models import User
from .models import (
    Team, TeamInvitation, APIKey, UserRole, TeamTier, 
    user_team_association, TeamCreate, TeamResponse,
    TeamInvitationCreate, TeamInvitationResponse,
    APIKeyCreate, APIKeyCreateResponse, APIKeyResponse,
    UserProfile, AuthContext
)
from .security import (
    get_password_hash, verify_password, 
    create_access_token, generate_api_key,
    get_user_team_role
)


class AuthService:
    """Centralized authentication and authorization service."""
    
    @staticmethod
    async def register_user(
        email: str,
        password: str,
        full_name: Optional[str],
        db: AsyncSession,
        create_default_team: bool = True
    ) -> Tuple[User, Optional[Team]]:
        """Register a new user and optionally create a default team."""
        try:
            # Check if user already exists
            stmt = select(User).where(User.email == email)
            result = await db.execute(stmt)
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Create user
            hashed_password = get_password_hash(password)
            user = User(
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                is_active=True,
                is_admin=False
            )
            
            db.add(user)
            await db.flush()  # Get user ID
            
            team = None
            if create_default_team:
                # Create default team
                team_slug = f"team-{user.id}".replace("-", "")[:20]
                team = Team(
                    name=f"{full_name or email.split('@')[0]}'s Team",
                    slug=team_slug,
                    tier=TeamTier.FREE,
                    max_users=5,
                    max_evaluations_per_month=1000
                )
                
                db.add(team)
                await db.flush()  # Get team ID
                
                # Add user to team as admin
                await AuthService.add_user_to_team(
                    user.id, team.id, UserRole.TEAM_ADMIN, user.id, db
                )
            
            await db.commit()
            return user, team
            
        except Exception as e:
            await db.rollback()
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Registration failed: {str(e)}"
            )
    
    @staticmethod
    async def authenticate_user(
        email: str,
        password: str,
        db: AsyncSession
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            stmt = select(User).where(
                and_(User.email == email, User.is_active == True)
            )
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user or not verify_password(password, user.hashed_password):
                return None
            
            return user
            
        except Exception:
            return None
    
    @staticmethod
    async def create_user_tokens(
        user: User,
        team_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Create access and refresh tokens for user."""
        try:
            # Get user's role in team if specified
            team_role = None
            if team_id and db:
                team_role = await get_user_team_role(str(user.id), team_id, db)
            
            # Token payload
            token_data = {
                "sub": user.email,
                "user_id": str(user.id),
                "team_id": team_id,
                "team_role": team_role.value if team_role else None,
                "scopes": ["read", "write"]  # Default scopes for JWT
            }
            
            # Create tokens
            access_token = create_access_token(
                token_data, 
                token_type="access"
            )
            refresh_token = create_access_token(
                token_data,
                token_type="refresh"
            )
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": 30 * 60  # 30 minutes
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Token creation failed: {str(e)}"
            )
    
    @staticmethod
    async def get_user_profile(
        user_id: str,
        db: AsyncSession
    ) -> UserProfile:
        """Get comprehensive user profile with teams."""
        try:
            # Get user with teams
            stmt = select(User).options(
                selectinload(User.teams)
            ).where(User.id == user_id)
            
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get team data with member counts
            teams = []
            current_team_role = None
            
            for team in user.teams:
                # Get member count
                member_count_stmt = select(func.count()).select_from(
                    user_team_association
                ).where(
                    and_(
                        user_team_association.c.team_id == team.id,
                        user_team_association.c.is_active == True
                    )
                )
                member_count_result = await db.execute(member_count_stmt)
                member_count = member_count_result.scalar()
                
                # Get user's role in this team
                role = await get_user_team_role(user_id, str(team.id), db)
                
                team_response = TeamResponse(
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
                teams.append(team_response)
                
                # Set current team role (first active team)
                if not current_team_role and role:
                    current_team_role = role
            
            return UserProfile(
                id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_admin=user.is_admin,
                created_at=user.created_at,
                teams=teams,
                current_team_role=current_team_role
            )
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user profile: {str(e)}"
            )
    
    # ========================================================================
    # TEAM MANAGEMENT
    # ========================================================================
    
    @staticmethod
    async def create_team(
        team_data: TeamCreate,
        creator_id: str,
        db: AsyncSession
    ) -> TeamResponse:
        """Create a new team."""
        try:
            # Check slug uniqueness
            stmt = select(Team).where(Team.slug == team_data.slug)
            result = await db.execute(stmt)
            existing_team = result.scalar_one_or_none()
            
            if existing_team:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Team slug already exists"
                )
            
            # Create team
            team = Team(
                name=team_data.name,
                slug=team_data.slug,
                description=team_data.description,
                tier=TeamTier.FREE,
                max_users=5,
                max_evaluations_per_month=1000
            )
            
            db.add(team)
            await db.flush()
            
            # Add creator as team admin
            await AuthService.add_user_to_team(
                creator_id, team.id, UserRole.TEAM_ADMIN, creator_id, db
            )
            
            await db.commit()
            
            return TeamResponse(
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
            )
            
        except Exception as e:
            await db.rollback()
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Team creation failed: {str(e)}"
            )
    
    @staticmethod
    async def add_user_to_team(
        user_id: str,
        team_id: str,
        role: UserRole,
        invited_by: str,
        db: AsyncSession
    ) -> bool:
        """Add user to team with specific role."""
        try:
            # Insert into association table
            stmt = user_team_association.insert().values(
                user_id=user_id,
                team_id=team_id,
                role=role.value,
                joined_at=datetime.utcnow(),
                invited_by=invited_by,
                is_active=True
            )
            
            await db.execute(stmt)
            return True
            
        except Exception:
            return False
    
    @staticmethod
    async def create_team_invitation(
        team_id: str,
        invitation_data: TeamInvitationCreate,
        inviter_id: str,
        db: AsyncSession
    ) -> TeamInvitationResponse:
        """Create a team invitation."""
        try:
            # Check if user is already a team member
            user_stmt = select(User).where(User.email == invitation_data.email)
            user_result = await db.execute(user_stmt)
            existing_user = user_result.scalar_one_or_none()
            
            if existing_user:
                # Check if already in team
                member_stmt = select(user_team_association).where(
                    and_(
                        user_team_association.c.user_id == existing_user.id,
                        user_team_association.c.team_id == team_id,
                        user_team_association.c.is_active == True
                    )
                )
                member_result = await db.execute(member_stmt)
                if member_result.first():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="User is already a team member"
                    )
            
            # Generate invitation token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            invitation = TeamInvitation(
                team_id=team_id,
                email=invitation_data.email,
                role=invitation_data.role.value,
                token=token,
                invited_by=inviter_id,
                expires_at=expires_at
            )
            
            db.add(invitation)
            await db.flush()
            
            # Get team and inviter info for response
            team_stmt = select(Team).where(Team.id == team_id)
            team_result = await db.execute(team_stmt)
            team = team_result.scalar_one()
            
            inviter_stmt = select(User).where(User.id == inviter_id)
            inviter_result = await db.execute(inviter_stmt)
            inviter = inviter_result.scalar_one()
            
            await db.commit()
            
            return TeamInvitationResponse(
                id=str(invitation.id),
                email=invitation.email,
                role=invitation.role,
                invited_at=invitation.invited_at,
                expires_at=invitation.expires_at,
                is_used=invitation.is_used,
                team_name=team.name,
                inviter_email=inviter.email
            )
            
        except Exception as e:
            await db.rollback()
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invitation creation failed: {str(e)}"
            )
    
    # ========================================================================
    # API KEY MANAGEMENT
    # ========================================================================
    
    @staticmethod
    async def create_api_key(
        key_data: APIKeyCreate,
        user_id: str,
        team_id: str,
        db: AsyncSession
    ) -> APIKeyCreateResponse:
        """Create a new API key."""
        try:
            # Generate API key
            full_key, key_hash = generate_api_key()
            key_prefix = full_key[:12]  # llmeval_xxxx
            
            # Create API key record
            api_key = APIKey(
                name=key_data.name,
                key_hash=key_hash,
                key_prefix=key_prefix,
                user_id=user_id,
                team_id=team_id,
                scopes=json.dumps(key_data.scopes),
                rate_limit_per_hour=key_data.rate_limit_per_hour
            )
            
            db.add(api_key)
            await db.flush()
            await db.commit()
            
            # Return response with full key (only time it's shown)
            api_key_response = APIKeyResponse(
                id=str(api_key.id),
                name=api_key.name,
                key_prefix=api_key.key_prefix,
                scopes=key_data.scopes,
                is_active=api_key.is_active,
                rate_limit_per_hour=api_key.rate_limit_per_hour,
                created_at=api_key.created_at,
                last_used_at=api_key.last_used_at,
                usage_count=api_key.usage_count
            )
            
            return APIKeyCreateResponse(
                api_key=api_key_response,
                secret_key=full_key
            )
            
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"API key creation failed: {str(e)}"
            )
    
    @staticmethod
    async def list_api_keys(
        team_id: str,
        user_id: Optional[str],
        db: AsyncSession
    ) -> List[APIKeyResponse]:
        """List API keys for team or user."""
        try:
            stmt = select(APIKey).where(APIKey.team_id == team_id)
            
            if user_id:
                stmt = stmt.where(APIKey.user_id == user_id)
            
            stmt = stmt.order_by(APIKey.created_at.desc())
            
            result = await db.execute(stmt)
            api_keys = result.scalars().all()
            
            return [
                APIKeyResponse(
                    id=str(key.id),
                    name=key.name,
                    key_prefix=key.key_prefix,
                    scopes=json.loads(key.scopes),
                    is_active=key.is_active,
                    rate_limit_per_hour=key.rate_limit_per_hour,
                    created_at=key.created_at,
                    last_used_at=key.last_used_at,
                    usage_count=key.usage_count
                )
                for key in api_keys
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list API keys: {str(e)}"
            )
    
    @staticmethod
    async def revoke_api_key(
        key_id: str,
        team_id: str,
        db: AsyncSession
    ) -> bool:
        """Revoke (deactivate) an API key."""
        try:
            stmt = update(APIKey).where(
                and_(
                    APIKey.id == key_id,
                    APIKey.team_id == team_id
                )
            ).values(is_active=False)
            
            result = await db.execute(stmt)
            await db.commit()
            
            return result.rowcount > 0
            
        except Exception:
            await db.rollback()
            return False
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    async def get_team_members(
        team_id: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get all members of a team with their roles."""
        try:
            stmt = select(
                User.id,
                User.email,
                User.full_name,
                User.is_active,
                user_team_association.c.role,
                user_team_association.c.joined_at
            ).select_from(
                User.__table__.join(
                    user_team_association,
                    User.id == user_team_association.c.user_id
                )
            ).where(
                and_(
                    user_team_association.c.team_id == team_id,
                    user_team_association.c.is_active == True
                )
            ).order_by(user_team_association.c.joined_at)
            
            result = await db.execute(stmt)
            members = result.fetchall()
            
            return [
                {
                    "user_id": str(member.id),
                    "email": member.email,
                    "full_name": member.full_name,
                    "is_active": member.is_active,
                    "role": member.role,
                    "joined_at": member.joined_at.isoformat()
                }
                for member in members
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get team members: {str(e)}"
            ) 