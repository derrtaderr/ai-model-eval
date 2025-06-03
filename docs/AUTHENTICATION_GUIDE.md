# Authentication System Guide

## Overview

The LLM Evaluation Platform features a comprehensive authentication system supporting:

- **JWT-based Authentication** - Secure token-based auth with refresh tokens
- **Multi-tenancy** - Team-based data isolation and resource management
- **Role-Based Access Control (RBAC)** - Granular permissions system
- **API Key Authentication** - Programmatic access with scoped permissions
- **Team Management** - User invitations and role management

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Database      │
│   (React)       │    │   (FastAPI)     │    │   (PostgreSQL)  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Login Form    │◄──►│ • Auth Router   │◄──►│ • Users Table   │
│ • Token Storage │    │ • JWT Middleware│    │ • Teams Table   │
│ • Protected     │    │ • RBAC System   │    │ • User-Team     │
│   Routes        │    │ • API Keys      │    │   Association   │
│ • Team Mgmt     │    │ • Team Isolation│    │ • API Keys      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. User Registration

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true,
    "is_admin": false,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "team": {
    "id": "456e7890-e89b-12d3-a456-426614174001",
    "name": "John Doe's Team",
    "slug": "team123e4567e89b12d3a456",
    "tier": "free",
    "max_users": 5,
    "member_count": 1
  },
  "tokens": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

### 2. User Login

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

### 3. Making Authenticated Requests

```bash
curl -X GET "http://localhost:8000/api/v1/auth/profile" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Authentication Methods

### JWT Tokens (Users)

**Access Token:**
- Short-lived (30 minutes default)
- Contains user ID, email, team context, and role
- Used for API authentication

**Refresh Token:**
- Long-lived (7 days default)
- Used to obtain new access tokens
- Stored securely client-side

**Token Structure:**
```json
{
  "sub": "user@example.com",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "team_id": "456e7890-e89b-12d3-a456-426614174001",
  "team_role": "team_admin",
  "scopes": ["read", "write"],
  "exp": 1642176000,
  "type": "access"
}
```

### API Keys (Programmatic Access)

API keys follow the format: `llmeval_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

```bash
# Create API key
curl -X POST "http://localhost:8000/api/v1/auth/teams/{team_id}/api-keys" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "scopes": ["read", "write", "create_evaluations"],
    "rate_limit_per_hour": 1000
  }'

# Use API key
curl -X GET "http://localhost:8000/api/v1/traces" \
  -H "Authorization: Bearer llmeval_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Role-Based Access Control (RBAC)

### User Roles

| Role | Description | Permissions |
|------|-------------|-------------|
| `super_admin` | Platform administrator | All permissions across all teams |
| `team_admin` | Team administrator | Manage team, users, API keys, view analytics |
| `evaluator` | Can create evaluations | Read, write, create evaluations, view own data |
| `viewer` | Read-only access | Read, view own data |
| `api_user` | API-only access | Based on API key scopes |

### Permission System

```python
# Available permissions
permissions = [
    "read",                 # View data
    "write",               # Create/update data  
    "delete",              # Delete data
    "manage_team",         # Team settings
    "manage_users",        # Invite/remove users
    "manage_api_keys",     # Create/revoke API keys
    "view_analytics",      # Access analytics
    "create_evaluations",  # Create evaluations
    "view_own_data"       # View user's own data only
]
```

### Checking Permissions

```bash
curl -X GET "http://localhost:8000/api/v1/auth/permissions" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Response:**
```json
{
  "role": "team_admin",
  "scopes": ["read", "write"],
  "permissions": {
    "read": true,
    "write": true,
    "delete": true,
    "manage_team": true,
    "manage_users": true,
    "manage_api_keys": true,
    "view_analytics": true,
    "create_evaluations": true,
    "view_own_data": true
  }
}
```

## Team Management

### Team Structure

- **Teams** provide data isolation between organizations
- **Members** have specific roles within teams
- **Invitations** enable controlled team onboarding
- **API Keys** are scoped to teams

### Team Operations

#### Create Team
```bash
curl -X POST "http://localhost:8000/api/v1/auth/teams" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Company",
    "slug": "my-company",
    "description": "Company evaluation team"
  }'
```

#### Invite User to Team
```bash
curl -X POST "http://localhost:8000/api/v1/auth/teams/{team_id}/invitations" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "role": "evaluator"
  }'
```

#### Accept Team Invitation
```bash
curl -X POST "http://localhost:8000/api/v1/auth/invitations/{invitation_token}/accept" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### List Team Members
```bash
curl -X GET "http://localhost:8000/api/v1/auth/teams/{team_id}/members" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Data Isolation

### Team-Level Isolation

All data is isolated by team:

- **Traces** include `team_id` field
- **Evaluations** are scoped to team
- **Test Cases** belong to teams  
- **Experiments** are team-specific
- **API Keys** provide team-scoped access

### Automatic Filtering

The system automatically filters data based on user's team context:

```python
# All queries automatically include team filtering
traces = db.query(Trace).filter(Trace.team_id == current_user.team_id)
```

### Cross-Team Access

Only `super_admin` users can access data across teams.

## API Key Management

### Scopes

API keys support granular scopes:

- `read` - Read access to team data
- `write` - Create/update team data
- `delete` - Delete team data
- `create_evaluations` - Specific evaluation creation
- `manage_team` - Team management operations
- `view_analytics` - Access to analytics data

### Rate Limiting

API keys support configurable rate limits:
- Default: 1000 requests/hour
- Range: 100-10,000 requests/hour
- Enforced per API key

### Key Management Operations

```bash
# List API keys
curl -X GET "http://localhost:8000/api/v1/auth/teams/{team_id}/api-keys" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Revoke API key  
curl -X DELETE "http://localhost:8000/api/v1/auth/teams/{team_id}/api-keys/{key_id}" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Frontend Integration

### React Authentication Hook

```typescript
import { useState, useEffect } from 'react';

interface AuthContext {
  user: User | null;
  team: Team | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

export const useAuth = (): AuthContext => {
  const [user, setUser] = useState<User | null>(null);
  const [team, setTeam] = useState<Team | null>(null);

  const login = async (email: string, password: string) => {
    const response = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    if (response.ok) {
      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      
      // Get user profile
      await fetchProfile();
    }
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setTeam(null);
  };

  const fetchProfile = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    const response = await fetch('/api/v1/auth/profile', {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.ok) {
      const profile = await response.json();
      setUser(profile);
      setTeam(profile.teams[0]); // Set first team as current
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  return {
    user,
    team,
    login,
    logout,
    isAuthenticated: !!user
  };
};
```

### Protected Route Component

```typescript
import { useAuth } from './useAuth';
import { Navigate } from 'react-router-dom';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredRole 
}) => {
  const { isAuthenticated, user } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  if (requiredRole && user?.current_team_role !== requiredRole) {
    return <Navigate to="/unauthorized" />;
  }

  return <>{children}</>;
};
```

## Security Best Practices

### Password Requirements
- Minimum 8 characters
- BCrypt hashing with salt
- No password in logs or responses

### Token Security
- Short-lived access tokens (30 minutes)
- Secure refresh token rotation
- HTTPS required in production
- HttpOnly cookies recommended for refresh tokens

### API Key Security
- SHA-256 hashed storage
- Rate limiting per key
- Scope-based permissions
- Regular rotation recommended

### Team Isolation
- All data queries filtered by team
- No cross-team data leakage
- Super admin oversight for compliance

## Environment Configuration

```env
# JWT Configuration
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database (required for auth)
DATABASE_URL=postgresql://user:pass@localhost/llmeval

# Optional: Custom endpoints
AUTH_CALLBACK_URL=http://localhost:3000/auth/callback
```

## Testing Authentication

### Unit Tests

```python
import pytest
from auth.service import AuthService

@pytest.mark.asyncio
async def test_user_registration():
    user, team = await AuthService.register_user(
        email="test@example.com",
        password="testpass123",
        full_name="Test User",
        db=db_session
    )
    assert user.email == "test@example.com"
    assert team.name == "Test User's Team"

@pytest.mark.asyncio  
async def test_api_key_creation():
    api_key = await AuthService.create_api_key(
        key_data={"name": "Test Key", "scopes": ["read"]},
        user_id=user_id,
        team_id=team_id,
        db=db_session
    )
    assert api_key.secret_key.startswith("llmeval_")
```

### Integration Tests

```bash
# Test registration flow
./scripts/test_auth_flow.sh

# Test API key authentication
./scripts/test_api_keys.sh

# Test team isolation
./scripts/test_team_isolation.sh
```

## Troubleshooting

### Common Issues

**"Invalid credentials" error:**
- Verify email/password combination
- Check user is active in database
- Ensure case-sensitive email matching

**"Token expired" error:**
- Use refresh token to get new access token
- Check token expiration settings
- Verify system clock accuracy

**"Permission denied" error:**
- Check user's role and permissions
- Verify team membership
- Ensure proper API key scopes

**"Team not found" error:**
- Verify team ID in requests
- Check user's team membership
- Ensure proper team isolation

### Debug Mode

```python
# Enable auth debugging
import logging
logging.getLogger('auth').setLevel(logging.DEBUG)
```

### Health Checks

```bash
# Check authentication status
curl -X GET "http://localhost:8000/api/v1/auth/check-auth" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Verify permissions
curl -X GET "http://localhost:8000/api/v1/auth/permissions" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

This authentication system provides enterprise-grade security with multi-tenancy, RBAC, and comprehensive API access control, enabling secure collaboration while maintaining strict data isolation between teams. 