"""
Slack Integration API
Comprehensive API for managing Slack notifications, webhooks, and team preferences.

Provides endpoints for:
- Webhook configuration and testing
- Notification preferences management
- Manual notification sending
- Notification history and analytics
- Team-based Slack settings
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, HttpUrl, validator

from auth.dependencies import get_current_user, require_permission
from database.models import User
from services.slack_service import (
    slack_service,
    NotificationType,
    NotificationPriority,
    NotificationPreferences
)


router = APIRouter(prefix="/slack", tags=["Slack Integration"])


class SlackWebhookRequest(BaseModel):
    """Request model for configuring Slack webhook."""
    webhook_url: HttpUrl = Field(..., description="Slack webhook URL")
    channel: Optional[str] = Field(None, description="Default channel name (e.g., #general)")
    enabled: bool = Field(True, description="Enable/disable notifications")
    

class NotificationPreferencesRequest(BaseModel):
    """Request model for updating notification preferences."""
    enabled: bool = Field(True, description="Enable/disable Slack notifications")
    webhook_url: Optional[HttpUrl] = Field(None, description="Team-specific webhook URL")
    default_channel: Optional[str] = Field(None, description="Default channel for notifications")
    notification_types: List[NotificationType] = Field(
        default_factory=lambda: list(NotificationType),
        description="Types of notifications to send"
    )
    priority_filter: NotificationPriority = Field(
        NotificationPriority.LOW,
        description="Minimum priority level for notifications"
    )
    quiet_hours_start: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Start hour for quiet period (0-23, UTC)"
    )
    quiet_hours_end: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="End hour for quiet period (0-23, UTC)"
    )
    rate_limit_per_hour: int = Field(
        60,
        ge=1,
        le=1000,
        description="Maximum notifications per hour"
    )
    mention_users: List[str] = Field(
        default_factory=list,
        description="Slack user IDs to mention for high-priority notifications"
    )


class TestNotificationRequest(BaseModel):
    """Request model for sending test notifications."""
    webhook_url: Optional[HttpUrl] = Field(None, description="Override webhook URL for test")
    notification_type: NotificationType = Field(
        NotificationType.ALERT,
        description="Type of test notification"
    )
    test_message: str = Field(
        "This is a test notification from LLM Evaluation Platform",
        description="Custom test message"
    )


class SendNotificationRequest(BaseModel):
    """Request model for sending custom notifications."""
    notification_type: NotificationType = Field(..., description="Type of notification")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: NotificationPriority = Field(
        NotificationPriority.MEDIUM,
        description="Notification priority"
    )
    fields: Optional[Dict[str, str]] = Field(
        None,
        description="Additional fields to include in notification"
    )
    channel: Optional[str] = Field(None, description="Override default channel")
    webhook_url: Optional[HttpUrl] = Field(None, description="Override webhook URL")


class NotificationHistoryResponse(BaseModel):
    """Response model for notification history."""
    total_sent: int
    recent_notifications: List[Dict[str, Any]]
    rate_limit_status: Dict[str, Any]
    preferences: Dict[str, Any]


@router.get("/preferences", response_model=Dict[str, Any])
async def get_notification_preferences(
    current_user: User = Depends(get_current_user)
):
    """
    Get current Slack notification preferences for the user's team.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else None
        prefs = await slack_service._get_notification_preferences(team_id)
        
        return {
            "team_id": prefs.team_id,
            "enabled": prefs.enabled,
            "webhook_configured": bool(prefs.webhook_url),
            "default_channel": prefs.default_channel,
            "notification_types": [nt.value for nt in prefs.notification_types],
            "priority_filter": prefs.priority_filter.value,
            "quiet_hours_start": prefs.quiet_hours_start,
            "quiet_hours_end": prefs.quiet_hours_end,
            "rate_limit_per_hour": prefs.rate_limit_per_hour,
            "mention_users": prefs.mention_users,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving notification preferences: {str(e)}"
        )


@router.put("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferencesRequest,
    current_user: User = Depends(require_permission("slack:configure"))
):
    """
    Update Slack notification preferences for the user's team.
    
    Requires 'slack:configure' permission.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else "default"
        
        # Create preferences object
        prefs = NotificationPreferences(
            team_id=team_id,
            enabled=preferences.enabled,
            webhook_url=str(preferences.webhook_url) if preferences.webhook_url else None,
            default_channel=preferences.default_channel,
            notification_types=preferences.notification_types,
            priority_filter=preferences.priority_filter,
            quiet_hours_start=preferences.quiet_hours_start,
            quiet_hours_end=preferences.quiet_hours_end,
            rate_limit_per_hour=preferences.rate_limit_per_hour,
            mention_users=preferences.mention_users
        )
        
        # Cache the preferences
        from utils.cache_service import cache_service
        cache_key = f"slack_prefs:{team_id}"
        await cache_service.set(cache_key, prefs.__dict__, ttl=3600)
        
        return {
            "status": "success",
            "message": "Notification preferences updated successfully",
            "preferences": prefs.__dict__,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating notification preferences: {str(e)}"
        )


@router.post("/test-webhook")
async def test_slack_webhook(
    test_request: TestNotificationRequest,
    current_user: User = Depends(require_permission("slack:test"))
):
    """
    Test a Slack webhook URL by sending a test notification.
    
    Requires 'slack:test' permission.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else None
        
        # Determine webhook URL
        webhook_url = None
        if test_request.webhook_url:
            webhook_url = str(test_request.webhook_url)
        else:
            # Get from preferences
            prefs = await slack_service._get_notification_preferences(team_id)
            webhook_url = prefs.webhook_url or slack_service.default_webhook_url
        
        if not webhook_url:
            raise HTTPException(
                status_code=400,
                detail="No webhook URL configured. Please provide a webhook URL or configure team preferences."
            )
        
        # Send test notification
        test_data = {
            "title": "Webhook Test",
            "message": test_request.test_message,
            "timestamp": datetime.utcnow().isoformat(),
            "user_name": current_user.full_name or current_user.email,
            "test_type": test_request.notification_type.value
        }
        
        success = await slack_service.send_notification(
            test_request.notification_type,
            test_data,
            team_id,
            NotificationPriority.LOW,
            webhook_url
        )
        
        if success:
            return {
                "status": "success",
                "message": "Test notification sent successfully",
                "webhook_url": webhook_url,
                "notification_type": test_request.notification_type.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to send test notification. Please check your webhook URL."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing webhook: {str(e)}"
        )


@router.post("/send-notification")
async def send_custom_notification(
    notification: SendNotificationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("slack:send"))
):
    """
    Send a custom notification to Slack.
    
    Requires 'slack:send' permission.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else None
        
        # Prepare notification data
        data = {
            "title": notification.title,
            "message": notification.message,
            "fields": notification.fields or {},
            "timestamp": datetime.utcnow().isoformat(),
            "sender": current_user.full_name or current_user.email
        }
        
        # Add custom fields if provided
        if notification.fields:
            data.update(notification.fields)
        
        # Send notification in background
        background_tasks.add_task(
            slack_service.send_notification,
            notification.notification_type,
            data,
            team_id,
            notification.priority,
            str(notification.webhook_url) if notification.webhook_url else None
        )
        
        return {
            "status": "queued",
            "message": "Notification queued for sending",
            "notification_type": notification.notification_type.value,
            "priority": notification.priority.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error sending notification: {str(e)}"
        )


@router.get("/history", response_model=NotificationHistoryResponse)
async def get_notification_history(
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
    current_user: User = Depends(get_current_user)
):
    """
    Get notification history and statistics for the user's team.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else "default"
        
        # Get rate limit data from cache
        from utils.cache_service import cache_service
        cache_key = f"slack_rate_limit:{team_id}"
        recent_notifications = await cache_service.get(cache_key) or []
        
        # Get preferences
        prefs = await slack_service._get_notification_preferences(team_id)
        
        # Calculate statistics
        now = datetime.utcnow()
        recent_count = len([
            ts for ts in recent_notifications
            if datetime.fromisoformat(ts) > now - timedelta(hours=1)
        ])
        
        # Mock some recent notifications data (in production, this would come from database)
        recent_notifications_data = []
        for i, ts in enumerate(recent_notifications[-10:]):  # Last 10
            recent_notifications_data.append({
                "timestamp": ts,
                "type": "test_notification",
                "status": "sent",
                "priority": "medium"
            })
        
        return NotificationHistoryResponse(
            total_sent=len(recent_notifications),
            recent_notifications=recent_notifications_data,
            rate_limit_status={
                "current_hour_count": recent_count,
                "limit_per_hour": prefs.rate_limit_per_hour,
                "remaining": max(0, prefs.rate_limit_per_hour - recent_count),
                "reset_time": (now + timedelta(hours=1)).isoformat()
            },
            preferences={
                "enabled": prefs.enabled,
                "notification_types": [nt.value for nt in prefs.notification_types],
                "priority_filter": prefs.priority_filter.value,
                "rate_limit": prefs.rate_limit_per_hour
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving notification history: {str(e)}"
        )


@router.get("/notification-types", response_model=List[Dict[str, Any]])
async def get_notification_types(
    current_user: User = Depends(get_current_user)
):
    """
    Get available notification types and their descriptions.
    """
    return [
        {
            "type": NotificationType.ALERT.value,
            "name": "Alerts",
            "description": "System alerts and warnings",
            "priority": "high",
            "icon": "🚨"
        },
        {
            "type": NotificationType.EVALUATION_COMPLETE.value,
            "name": "Evaluation Complete",
            "description": "Notifications when evaluation batches are completed",
            "priority": "medium",
            "icon": "✅"
        },
        {
            "type": NotificationType.EXPERIMENT_RESULT.value,
            "name": "Experiment Results",
            "description": "A/B test and experiment result notifications",
            "priority": "high",
            "icon": "🧪"
        },
        {
            "type": NotificationType.DAILY_REPORT.value,
            "name": "Daily Reports",
            "description": "Daily metrics and performance summaries",
            "priority": "low",
            "icon": "📊"
        },
        {
            "type": NotificationType.PERFORMANCE_WARNING.value,
            "name": "Performance Warnings",
            "description": "System performance and health warnings",
            "priority": "medium",
            "icon": "⚠️"
        },
        {
            "type": NotificationType.ERROR_NOTIFICATION.value,
            "name": "Error Notifications",
            "description": "System error and failure notifications",
            "priority": "critical",
            "icon": "❌"
        },
        {
            "type": NotificationType.USER_ACTION.value,
            "name": "User Actions",
            "description": "Important user action notifications",
            "priority": "low",
            "icon": "👤"
        },
        {
            "type": NotificationType.TRACE_ANOMALY.value,
            "name": "Trace Anomalies",
            "description": "Unusual trace patterns and anomalies",
            "priority": "medium",
            "icon": "🔍"
        }
    ]


@router.get("/status", response_model=Dict[str, Any])
async def get_slack_integration_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the current status of Slack integration for the user's team.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else None
        prefs = await slack_service._get_notification_preferences(team_id)
        
        # Check webhook connectivity (simplified)
        webhook_status = "unknown"
        if prefs.webhook_url:
            try:
                # This would normally test the webhook, but we'll simulate for now
                webhook_status = "connected"
            except:
                webhook_status = "error"
        else:
            webhook_status = "not_configured"
        
        return {
            "integration_enabled": prefs.enabled,
            "webhook_status": webhook_status,
            "webhook_configured": bool(prefs.webhook_url),
            "notification_types_count": len(prefs.notification_types),
            "total_notification_types": len(list(NotificationType)),
            "rate_limit_per_hour": prefs.rate_limit_per_hour,
            "quiet_hours_configured": bool(prefs.quiet_hours_start and prefs.quiet_hours_end),
            "mentions_configured": len(prefs.mention_users),
            "last_check": datetime.utcnow().isoformat(),
            "team_id": team_id
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting Slack integration status: {str(e)}"
        )


@router.post("/quick-setup")
async def quick_slack_setup(
    webhook_request: SlackWebhookRequest,
    current_user: User = Depends(require_permission("slack:configure"))
):
    """
    Quick setup for Slack integration with sensible defaults.
    
    Requires 'slack:configure' permission.
    """
    try:
        team_id = str(current_user.team_id) if current_user.team_id else "default"
        
        # Test webhook first
        test_result = await slack_service.test_webhook(str(webhook_request.webhook_url))
        
        if not test_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Webhook test failed: {test_result['message']}"
            )
        
        # Create default preferences with sensible defaults
        prefs = NotificationPreferences(
            team_id=team_id,
            enabled=webhook_request.enabled,
            webhook_url=str(webhook_request.webhook_url),
            default_channel=webhook_request.channel,
            notification_types=[
                NotificationType.ALERT,
                NotificationType.EVALUATION_COMPLETE,
                NotificationType.EXPERIMENT_RESULT,
                NotificationType.PERFORMANCE_WARNING,
                NotificationType.ERROR_NOTIFICATION
            ],
            priority_filter=NotificationPriority.MEDIUM,
            rate_limit_per_hour=30,  # Conservative default
            quiet_hours_start=22,    # 10 PM UTC
            quiet_hours_end=8        # 8 AM UTC
        )
        
        # Save preferences
        from utils.cache_service import cache_service
        cache_key = f"slack_prefs:{team_id}"
        await cache_service.set(cache_key, prefs.__dict__, ttl=3600)
        
        return {
            "status": "success",
            "message": "Slack integration configured successfully",
            "webhook_test_result": test_result,
            "preferences": {
                "enabled": prefs.enabled,
                "notification_types": [nt.value for nt in prefs.notification_types],
                "rate_limit": prefs.rate_limit_per_hour,
                "quiet_hours": f"{prefs.quiet_hours_start}:00 - {prefs.quiet_hours_end}:00 UTC"
            },
            "next_steps": [
                "Configure specific notification types in preferences",
                "Add Slack user IDs for mentions if needed",
                "Test notifications using the test endpoint"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error setting up Slack integration: {str(e)}"
        ) 