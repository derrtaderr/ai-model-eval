"""
Slack Integration Service
Comprehensive Slack integration for sending notifications about system events, alerts, and user activities.

Features:
- Rich message formatting with attachments
- Multiple notification types and templates
- Team-based notification preferences
- Event-driven notification system
- Rate limiting and error handling
- Webhook URL management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from decouple import config

from config.settings import get_settings
from utils.cache_service import cache_service


logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications that can be sent to Slack."""
    ALERT = "alert"
    EVALUATION_COMPLETE = "evaluation_complete"
    EXPERIMENT_RESULT = "experiment_result"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE_WARNING = "performance_warning"
    ERROR_NOTIFICATION = "error_notification"
    DAILY_REPORT = "daily_report"
    WEEKLY_SUMMARY = "weekly_summary"
    USER_ACTION = "user_action"
    TRACE_ANOMALY = "trace_anomaly"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SlackColor(str, Enum):
    """Standard Slack colors for message attachments."""
    GOOD = "good"  # Green
    WARNING = "warning"  # Yellow/Orange
    DANGER = "danger"  # Red
    BLUE = "#36a64f"
    PURPLE = "#9c27b0"
    GRAY = "#808080"


@dataclass
class SlackAttachment:
    """Slack message attachment structure."""
    color: str
    title: Optional[str] = None
    title_link: Optional[str] = None
    text: Optional[str] = None
    fields: List[Dict[str, Any]] = field(default_factory=list)
    footer: Optional[str] = None
    footer_icon: Optional[str] = None
    ts: Optional[int] = None
    
    def add_field(self, title: str, value: str, short: bool = True):
        """Add a field to the attachment."""
        self.fields.append({
            "title": title,
            "value": value,
            "short": short
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Slack API format."""
        result = {"color": self.color}
        
        if self.title:
            result["title"] = self.title
        if self.title_link:
            result["title_link"] = self.title_link
        if self.text:
            result["text"] = self.text
        if self.fields:
            result["fields"] = self.fields
        if self.footer:
            result["footer"] = self.footer
        if self.footer_icon:
            result["footer_icon"] = self.footer_icon
        if self.ts:
            result["ts"] = self.ts
        
        return result


@dataclass
class SlackMessage:
    """Slack message structure."""
    text: str
    channel: Optional[str] = None
    username: Optional[str] = None
    icon_emoji: Optional[str] = None
    icon_url: Optional[str] = None
    attachments: List[SlackAttachment] = field(default_factory=list)
    blocks: Optional[List[Dict[str, Any]]] = None
    thread_ts: Optional[str] = None
    
    def add_attachment(self, attachment: SlackAttachment):
        """Add an attachment to the message."""
        self.attachments.append(attachment)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Slack API format."""
        result = {"text": self.text}
        
        if self.channel:
            result["channel"] = self.channel
        if self.username:
            result["username"] = self.username
        if self.icon_emoji:
            result["icon_emoji"] = self.icon_emoji
        if self.icon_url:
            result["icon_url"] = self.icon_url
        if self.attachments:
            result["attachments"] = [att.to_dict() for att in self.attachments]
        if self.blocks:
            result["blocks"] = self.blocks
        if self.thread_ts:
            result["thread_ts"] = self.thread_ts
        
        return result


@dataclass
class NotificationPreferences:
    """Team notification preferences."""
    team_id: str
    enabled: bool = True
    webhook_url: Optional[str] = None
    default_channel: Optional[str] = None
    notification_types: List[NotificationType] = field(default_factory=list)
    priority_filter: NotificationPriority = NotificationPriority.LOW
    quiet_hours_start: Optional[int] = None  # Hour of day (0-23)
    quiet_hours_end: Optional[int] = None
    rate_limit_per_hour: int = 60
    mention_users: List[str] = field(default_factory=list)  # Slack user IDs


class SlackNotificationService:
    """Service for managing Slack notifications."""
    
    def __init__(self):
        self.settings = get_settings()
        self.default_webhook_url = getattr(self.settings, 'SLACK_WEBHOOK_URL', None)
        self.default_bot_token = getattr(self.settings, 'SLACK_BOT_TOKEN', None)
        self.app_name = "LLM Evaluation Platform"
        self.app_icon = ":robot_face:"
        self.base_url = getattr(self.settings, 'APP_BASE_URL', 'http://localhost:3000')
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Template cache
        self._template_cache: Dict[str, Dict[str, Any]] = {}
    
    async def send_notification(
        self,
        notification_type: NotificationType,
        data: Dict[str, Any],
        team_id: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        custom_webhook_url: Optional[str] = None
    ) -> bool:
        """
        Send a notification to Slack.
        
        Args:
            notification_type: Type of notification
            data: Data for the notification template
            team_id: Team ID for notification preferences
            priority: Priority level
            custom_webhook_url: Override webhook URL
            
        Returns:
            True if notification was sent successfully
        """
        try:
            # Get notification preferences
            prefs = await self._get_notification_preferences(team_id)
            
            # Check if notifications are enabled and type is allowed
            if not prefs.enabled or notification_type not in prefs.notification_types:
                logger.debug(f"Notification {notification_type} skipped for team {team_id}")
                return False
            
            # Check priority filter
            if self._priority_level(priority) < self._priority_level(prefs.priority_filter):
                logger.debug(f"Notification {notification_type} filtered by priority for team {team_id}")
                return False
            
            # Check quiet hours
            if self._is_quiet_hours(prefs):
                logger.debug(f"Notification {notification_type} skipped due to quiet hours for team {team_id}")
                return False
            
            # Check rate limiting
            if not await self._check_rate_limit(team_id or "default", prefs.rate_limit_per_hour):
                logger.warning(f"Rate limit exceeded for team {team_id}")
                return False
            
            # Determine webhook URL
            webhook_url = custom_webhook_url or prefs.webhook_url or self.default_webhook_url
            if not webhook_url:
                logger.error(f"No webhook URL configured for team {team_id}")
                return False
            
            # Generate message
            message = await self._create_message(notification_type, data, prefs, priority)
            
            # Send message
            success = await self._send_slack_message(webhook_url, message)
            
            if success:
                # Update rate limiting
                await self._record_notification(team_id or "default")
                logger.info(f"Slack notification sent: {notification_type} for team {team_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def send_alert(
        self,
        title: str,
        message: str,
        team_id: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.HIGH,
        fields: Optional[Dict[str, str]] = None,
        link_url: Optional[str] = None
    ) -> bool:
        """Send a generic alert notification."""
        data = {
            "title": title,
            "message": message,
            "fields": fields or {},
            "link_url": link_url,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_notification(
            NotificationType.ALERT,
            data,
            team_id,
            priority
        )
    
    async def send_evaluation_complete(
        self,
        trace_count: int,
        acceptance_rate: float,
        team_id: Optional[str] = None,
        evaluator_name: Optional[str] = None
    ) -> bool:
        """Send evaluation completion notification."""
        data = {
            "trace_count": trace_count,
            "acceptance_rate": acceptance_rate,
            "evaluator_name": evaluator_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_notification(
            NotificationType.EVALUATION_COMPLETE,
            data,
            team_id,
            NotificationPriority.MEDIUM
        )
    
    async def send_experiment_result(
        self,
        experiment_name: str,
        variant_results: Dict[str, Any],
        significance: float,
        team_id: Optional[str] = None
    ) -> bool:
        """Send experiment result notification."""
        data = {
            "experiment_name": experiment_name,
            "variant_results": variant_results,
            "significance": significance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_notification(
            NotificationType.EXPERIMENT_RESULT,
            data,
            team_id,
            NotificationPriority.HIGH
        )
    
    async def send_daily_report(
        self,
        metrics: Dict[str, Any],
        team_id: Optional[str] = None
    ) -> bool:
        """Send daily metrics report."""
        data = {
            "metrics": metrics,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_notification(
            NotificationType.DAILY_REPORT,
            data,
            team_id,
            NotificationPriority.LOW
        )
    
    async def test_webhook(self, webhook_url: str) -> Dict[str, Any]:
        """Test a Slack webhook URL."""
        try:
            test_message = SlackMessage(
                text="🔔 Test notification from LLM Evaluation Platform",
                username=self.app_name,
                icon_emoji=self.app_icon
            )
            
            test_attachment = SlackAttachment(
                color=SlackColor.GOOD,
                title="Webhook Test Successful",
                text="Your Slack integration is working correctly!",
                footer=self.app_name,
                ts=int(datetime.utcnow().timestamp())
            )
            
            test_attachment.add_field("Status", "✅ Connected", True)
            test_attachment.add_field("Time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), True)
            
            test_message.add_attachment(test_attachment)
            
            success = await self._send_slack_message(webhook_url, test_message)
            
            return {
                "success": success,
                "message": "Test notification sent successfully" if success else "Failed to send test notification",
                "webhook_url": webhook_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Webhook test failed: {str(e)}",
                "webhook_url": webhook_url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _create_message(
        self,
        notification_type: NotificationType,
        data: Dict[str, Any],
        prefs: NotificationPreferences,
        priority: NotificationPriority
    ) -> SlackMessage:
        """Create a Slack message from notification data."""
        # Get template
        template = self._get_notification_template(notification_type)
        
        # Create base message
        message = SlackMessage(
            text=template["text"].format(**data),
            channel=prefs.default_channel,
            username=self.app_name,
            icon_emoji=self.app_icon
        )
        
        # Create attachment
        attachment = SlackAttachment(
            color=self._get_priority_color(priority),
            title=template.get("title", "").format(**data),
            title_link=self._generate_link(notification_type, data),
            text=template.get("description", "").format(**data),
            footer=self.app_name,
            ts=int(datetime.utcnow().timestamp())
        )
        
        # Add fields from template
        for field_template in template.get("fields", []):
            attachment.add_field(
                title=field_template["title"],
                value=field_template["value"].format(**data),
                short=field_template.get("short", True)
            )
        
        # Add mentions if configured
        if prefs.mention_users and priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
            mentions = " ".join([f"<@{user_id}>" for user_id in prefs.mention_users])
            attachment.text = f"{mentions}\n\n{attachment.text}"
        
        message.add_attachment(attachment)
        
        return message
    
    def _get_notification_template(self, notification_type: NotificationType) -> Dict[str, Any]:
        """Get notification template for a specific type."""
        if notification_type.value in self._template_cache:
            return self._template_cache[notification_type.value]
        
        templates = {
            NotificationType.ALERT: {
                "text": "🚨 Alert: {title}",
                "title": "{title}",
                "description": "{message}",
                "fields": [
                    {"title": "Time", "value": "{timestamp}", "short": True},
                    {"title": "Priority", "value": "High", "short": True}
                ]
            },
            NotificationType.EVALUATION_COMPLETE: {
                "text": "✅ Evaluation completed",
                "title": "Evaluation Batch Completed",
                "description": "{evaluator_name} has finished evaluating {trace_count} traces.",
                "fields": [
                    {"title": "Traces Evaluated", "value": "{trace_count}", "short": True},
                    {"title": "Acceptance Rate", "value": "{acceptance_rate:.1%}", "short": True},
                    {"title": "Evaluator", "value": "{evaluator_name}", "short": True},
                    {"title": "Completed", "value": "{timestamp}", "short": True}
                ]
            },
            NotificationType.EXPERIMENT_RESULT: {
                "text": "🧪 Experiment results available",
                "title": "Experiment: {experiment_name}",
                "description": "A/B test results are ready for review with {significance:.2%} statistical significance.",
                "fields": [
                    {"title": "Experiment", "value": "{experiment_name}", "short": True},
                    {"title": "Significance", "value": "{significance:.2%}", "short": True},
                    {"title": "Results", "value": "View detailed results in dashboard", "short": False}
                ]
            },
            NotificationType.DAILY_REPORT: {
                "text": "📊 Daily metrics report",
                "title": "Daily Report - {date}",
                "description": "Your daily LLM evaluation metrics summary.",
                "fields": [
                    {"title": "Total Traces", "value": "{metrics[total_traces]}", "short": True},
                    {"title": "Evaluations", "value": "{metrics[evaluations_count]}", "short": True},
                    {"title": "Avg Response Time", "value": "{metrics[avg_response_time]:.1f}ms", "short": True},
                    {"title": "Success Rate", "value": "{metrics[success_rate]:.1%}", "short": True}
                ]
            },
            NotificationType.PERFORMANCE_WARNING: {
                "text": "⚠️ Performance warning",
                "title": "Performance Alert",
                "description": "System performance metrics indicate potential issues.",
                "fields": [
                    {"title": "Metric", "value": "{metric_name}", "short": True},
                    {"title": "Current Value", "value": "{current_value}", "short": True},
                    {"title": "Threshold", "value": "{threshold}", "short": True},
                    {"title": "Time", "value": "{timestamp}", "short": True}
                ]
            },
            NotificationType.ERROR_NOTIFICATION: {
                "text": "❌ System error detected",
                "title": "Error Notification",
                "description": "An error has been detected in the system.",
                "fields": [
                    {"title": "Error Type", "value": "{error_type}", "short": True},
                    {"title": "Component", "value": "{component}", "short": True},
                    {"title": "Message", "value": "{error_message}", "short": False},
                    {"title": "Time", "value": "{timestamp}", "short": True}
                ]
            }
        }
        
        template = templates.get(notification_type, {
            "text": "📢 Notification",
            "title": "System Notification",
            "description": "A system event has occurred.",
            "fields": []
        })
        
        self._template_cache[notification_type.value] = template
        return template
    
    def _get_priority_color(self, priority: NotificationPriority) -> str:
        """Get Slack color for priority level."""
        color_map = {
            NotificationPriority.LOW: SlackColor.GRAY,
            NotificationPriority.MEDIUM: SlackColor.BLUE,
            NotificationPriority.HIGH: SlackColor.WARNING,
            NotificationPriority.CRITICAL: SlackColor.DANGER
        }
        return color_map.get(priority, SlackColor.BLUE)
    
    def _generate_link(self, notification_type: NotificationType, data: Dict[str, Any]) -> Optional[str]:
        """Generate a link to the relevant page in the application."""
        if notification_type == NotificationType.EVALUATION_COMPLETE:
            return f"{self.base_url}/evaluations"
        elif notification_type == NotificationType.EXPERIMENT_RESULT:
            return f"{self.base_url}/experiments"
        elif notification_type in [NotificationType.ALERT, NotificationType.PERFORMANCE_WARNING]:
            return f"{self.base_url}/analytics"
        else:
            return f"{self.base_url}/dashboard"
    
    async def _send_slack_message(self, webhook_url: str, message: SlackMessage) -> bool:
        """Send message to Slack webhook."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=message.to_dict(),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"Slack webhook error: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False
    
    async def _get_notification_preferences(self, team_id: Optional[str]) -> NotificationPreferences:
        """Get notification preferences for a team."""
        if not team_id:
            # Default preferences
            return NotificationPreferences(
                team_id="default",
                enabled=True,
                notification_types=list(NotificationType),
                priority_filter=NotificationPriority.LOW,
                rate_limit_per_hour=60
            )
        
        # Try to get from cache
        cache_key = f"slack_prefs:{team_id}"
        cached_prefs = await cache_service.get(cache_key)
        
        if cached_prefs:
            return NotificationPreferences(**cached_prefs)
        
        # For now, return default preferences
        # In a full implementation, this would query the database
        prefs = NotificationPreferences(
            team_id=team_id,
            enabled=True,
            notification_types=list(NotificationType),
            priority_filter=NotificationPriority.LOW,
            rate_limit_per_hour=60
        )
        
        # Cache preferences
        await cache_service.set(cache_key, prefs.__dict__, ttl=3600)
        
        return prefs
    
    def _priority_level(self, priority: NotificationPriority) -> int:
        """Get numeric priority level."""
        levels = {
            NotificationPriority.LOW: 1,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.HIGH: 3,
            NotificationPriority.CRITICAL: 4
        }
        return levels.get(priority, 1)
    
    def _is_quiet_hours(self, prefs: NotificationPreferences) -> bool:
        """Check if current time is within quiet hours."""
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return False
        
        current_hour = datetime.utcnow().hour
        
        if prefs.quiet_hours_start <= prefs.quiet_hours_end:
            # Same day range (e.g., 22:00 to 06:00)
            return prefs.quiet_hours_start <= current_hour <= prefs.quiet_hours_end
        else:
            # Overnight range (e.g., 22:00 to 06:00)
            return current_hour >= prefs.quiet_hours_start or current_hour <= prefs.quiet_hours_end
    
    async def _check_rate_limit(self, identifier: str, limit_per_hour: int) -> bool:
        """Check if rate limit allows sending notification."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Get recent notifications from cache
        cache_key = f"slack_rate_limit:{identifier}"
        recent_notifications = await cache_service.get(cache_key) or []
        
        # Convert to datetime objects and filter
        recent_notifications = [
            datetime.fromisoformat(ts) for ts in recent_notifications
            if datetime.fromisoformat(ts) > hour_ago
        ]
        
        return len(recent_notifications) < limit_per_hour
    
    async def _record_notification(self, identifier: str):
        """Record a notification for rate limiting."""
        now = datetime.utcnow()
        
        cache_key = f"slack_rate_limit:{identifier}"
        recent_notifications = await cache_service.get(cache_key) or []
        
        # Add current notification
        recent_notifications.append(now.isoformat())
        
        # Keep only last hour
        hour_ago = now - timedelta(hours=1)
        recent_notifications = [
            ts for ts in recent_notifications
            if datetime.fromisoformat(ts) > hour_ago
        ]
        
        # Update cache
        await cache_service.set(cache_key, recent_notifications, ttl=3600)


# Global service instance
slack_service = SlackNotificationService() 