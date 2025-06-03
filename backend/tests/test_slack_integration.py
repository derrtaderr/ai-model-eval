"""
Comprehensive test suite for Slack integration functionality.

Tests cover:
- Slack service functionality
- Notification creation and sending
- API endpoint testing
- Template system
- Rate limiting
- Error handling
- Integration scenarios
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from services.slack_service import (
    SlackNotificationService,
    NotificationType,
    NotificationPriority,
    SlackMessage,
    SlackAttachment,
    NotificationPreferences,
    slack_service
)
from api.slack import router as slack_router
from main import app


@pytest.fixture
def slack_service_instance():
    """Create a fresh Slack service instance for testing."""
    service = SlackNotificationService()
    service.default_webhook_url = "https://hooks.slack.com/test/webhook"
    return service


@pytest.fixture
def test_preferences():
    """Create test notification preferences."""
    return NotificationPreferences(
        team_id="test_team",
        enabled=True,
        webhook_url="https://hooks.slack.com/test/webhook",
        default_channel="#test",
        notification_types=[
            NotificationType.ALERT,
            NotificationType.EVALUATION_COMPLETE,
            NotificationType.EXPERIMENT_RESULT
        ],
        priority_filter=NotificationPriority.LOW,
        rate_limit_per_hour=60
    )


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestSlackMessage:
    """Test Slack message and attachment structures."""
    
    def test_slack_attachment_creation(self):
        """Test creating and converting Slack attachments."""
        attachment = SlackAttachment(color="good", title="Test Title")
        attachment.add_field("Field 1", "Value 1", True)
        attachment.add_field("Field 2", "Value 2", False)
        
        result = attachment.to_dict()
        
        assert result["color"] == "good"
        assert result["title"] == "Test Title"
        assert len(result["fields"]) == 2
        assert result["fields"][0]["title"] == "Field 1"
        assert result["fields"][0]["value"] == "Value 1"
        assert result["fields"][0]["short"] is True
        assert result["fields"][1]["short"] is False
    
    def test_slack_message_creation(self):
        """Test creating and converting Slack messages."""
        message = SlackMessage(
            text="Test message",
            channel="#general",
            username="Test Bot"
        )
        
        attachment = SlackAttachment(color="warning", title="Warning")
        message.add_attachment(attachment)
        
        result = message.to_dict()
        
        assert result["text"] == "Test message"
        assert result["channel"] == "#general"
        assert result["username"] == "Test Bot"
        assert len(result["attachments"]) == 1
        assert result["attachments"][0]["color"] == "warning"


class TestNotificationTemplates:
    """Test notification template system."""
    
    def test_alert_template(self, slack_service_instance):
        """Test alert notification template."""
        template = slack_service_instance._get_notification_template(NotificationType.ALERT)
        
        assert "🚨" in template["text"]
        assert template["title"] == "{title}"
        assert template["description"] == "{message}"
        assert len(template["fields"]) >= 2
    
    def test_evaluation_complete_template(self, slack_service_instance):
        """Test evaluation complete notification template."""
        template = slack_service_instance._get_notification_template(
            NotificationType.EVALUATION_COMPLETE
        )
        
        assert "✅" in template["text"]
        assert "Evaluation" in template["title"]
        assert len(template["fields"]) >= 4
        
        # Check field templates contain expected placeholders
        field_values = [field["value"] for field in template["fields"]]
        assert any("{trace_count}" in value for value in field_values)
        assert any("{acceptance_rate" in value for value in field_values)
    
    def test_experiment_result_template(self, slack_service_instance):
        """Test experiment result notification template."""
        template = slack_service_instance._get_notification_template(
            NotificationType.EXPERIMENT_RESULT
        )
        
        assert "🧪" in template["text"]
        assert "Experiment" in template["title"]
        assert "{experiment_name}" in template["title"]
        assert "{significance" in template["description"]
    
    def test_template_caching(self, slack_service_instance):
        """Test that templates are cached properly."""
        # Clear cache first
        slack_service_instance._template_cache.clear()
        
        # Get template twice
        template1 = slack_service_instance._get_notification_template(NotificationType.ALERT)
        template2 = slack_service_instance._get_notification_template(NotificationType.ALERT)
        
        # Should be same object (cached)
        assert template1 is template2
        assert NotificationType.ALERT.value in slack_service_instance._template_cache


class TestNotificationPreferences:
    """Test notification preferences and filtering."""
    
    def test_priority_filtering(self, slack_service_instance):
        """Test priority level filtering."""
        assert slack_service_instance._priority_level(NotificationPriority.LOW) == 1
        assert slack_service_instance._priority_level(NotificationPriority.MEDIUM) == 2
        assert slack_service_instance._priority_level(NotificationPriority.HIGH) == 3
        assert slack_service_instance._priority_level(NotificationPriority.CRITICAL) == 4
    
    def test_quiet_hours_same_day(self, slack_service_instance):
        """Test quiet hours calculation for same-day range."""
        prefs = NotificationPreferences(
            team_id="test",
            quiet_hours_start=9,  # 9 AM
            quiet_hours_end=17    # 5 PM
        )
        
        with patch('services.slack_service.datetime') as mock_datetime:
            # Test during quiet hours (noon)
            mock_datetime.utcnow.return_value.hour = 12
            assert slack_service_instance._is_quiet_hours(prefs) is True
            
            # Test outside quiet hours (evening)
            mock_datetime.utcnow.return_value.hour = 20
            assert slack_service_instance._is_quiet_hours(prefs) is False
    
    def test_quiet_hours_overnight(self, slack_service_instance):
        """Test quiet hours calculation for overnight range."""
        prefs = NotificationPreferences(
            team_id="test",
            quiet_hours_start=22,  # 10 PM
            quiet_hours_end=8      # 8 AM
        )
        
        with patch('services.slack_service.datetime') as mock_datetime:
            # Test during overnight quiet hours (2 AM)
            mock_datetime.utcnow.return_value.hour = 2
            assert slack_service_instance._is_quiet_hours(prefs) is True
            
            # Test during overnight quiet hours (11 PM)
            mock_datetime.utcnow.return_value.hour = 23
            assert slack_service_instance._is_quiet_hours(prefs) is True
            
            # Test outside quiet hours (noon)
            mock_datetime.utcnow.return_value.hour = 12
            assert slack_service_instance._is_quiet_hours(prefs) is False
    
    def test_no_quiet_hours(self, slack_service_instance):
        """Test behavior when no quiet hours are configured."""
        prefs = NotificationPreferences(
            team_id="test",
            quiet_hours_start=None,
            quiet_hours_end=None
        )
        
        assert slack_service_instance._is_quiet_hours(prefs) is False


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_empty(self, slack_service_instance):
        """Test rate limiting with no previous notifications."""
        with patch.object(slack_service_instance, '_check_rate_limit') as mock_check:
            mock_check.return_value = True
            
            result = await slack_service_instance._check_rate_limit("test_team", 60)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, slack_service_instance):
        """Test rate limiting when limit is exceeded."""
        # Mock cache service to return many recent notifications
        recent_notifications = [
            (datetime.utcnow() - timedelta(minutes=i)).isoformat()
            for i in range(61)  # 61 notifications in last hour
        ]
        
        with patch('utils.cache_service.cache_service') as mock_cache:
            mock_cache.get.return_value = recent_notifications
            
            result = await slack_service_instance._check_rate_limit("test_team", 60)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limit_within_bounds(self, slack_service_instance):
        """Test rate limiting when within bounds."""
        # Mock cache service to return few recent notifications
        recent_notifications = [
            (datetime.utcnow() - timedelta(minutes=i)).isoformat()
            for i in range(30)  # 30 notifications in last hour
        ]
        
        with patch('utils.cache_service.cache_service') as mock_cache:
            mock_cache.get.return_value = recent_notifications
            
            result = await slack_service_instance._check_rate_limit("test_team", 60)
            assert result is True


class TestMessageCreation:
    """Test message creation from templates and data."""
    
    @pytest.mark.asyncio
    async def test_create_alert_message(self, slack_service_instance, test_preferences):
        """Test creating alert message."""
        data = {
            "title": "Test Alert",
            "message": "This is a test alert message",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = await slack_service_instance._create_message(
            NotificationType.ALERT,
            data,
            test_preferences,
            NotificationPriority.HIGH
        )
        
        assert isinstance(message, SlackMessage)
        assert "Test Alert" in message.text
        assert message.channel == test_preferences.default_channel
        assert message.username == slack_service_instance.app_name
        assert len(message.attachments) == 1
        
        attachment = message.attachments[0]
        assert attachment.title == "Test Alert"
        assert attachment.text == "This is a test alert message"
    
    @pytest.mark.asyncio
    async def test_create_evaluation_message(self, slack_service_instance, test_preferences):
        """Test creating evaluation complete message."""
        data = {
            "trace_count": 150,
            "acceptance_rate": 0.85,
            "evaluator_name": "Test Evaluator",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = await slack_service_instance._create_message(
            NotificationType.EVALUATION_COMPLETE,
            data,
            test_preferences,
            NotificationPriority.MEDIUM
        )
        
        assert isinstance(message, SlackMessage)
        assert len(message.attachments) == 1
        
        attachment = message.attachments[0]
        assert len(attachment.fields) >= 4
        
        # Check that data was properly formatted
        field_values = [field["value"] for field in attachment.fields]
        assert "150" in str(field_values)  # trace_count
        assert "85.0%" in str(field_values) or "0.85" in str(field_values)  # acceptance_rate
    
    @pytest.mark.asyncio
    async def test_message_with_mentions(self, slack_service_instance):
        """Test message creation with user mentions for high priority."""
        prefs = NotificationPreferences(
            team_id="test",
            mention_users=["U123456", "U789012"],
            default_channel="#alerts"
        )
        
        data = {
            "title": "Critical Alert",
            "message": "System is down",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = await slack_service_instance._create_message(
            NotificationType.ALERT,
            data,
            prefs,
            NotificationPriority.CRITICAL
        )
        
        attachment = message.attachments[0]
        assert "<@U123456>" in attachment.text
        assert "<@U789012>" in attachment.text


class TestNotificationSending:
    """Test actual notification sending functionality."""
    
    @pytest.mark.asyncio
    async def test_send_slack_message_success(self, slack_service_instance):
        """Test successful Slack message sending."""
        message = SlackMessage(text="Test message")
        webhook_url = "https://hooks.slack.com/test"
        
        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await slack_service_instance._send_slack_message(webhook_url, message)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_slack_message_failure(self, slack_service_instance):
        """Test failed Slack message sending."""
        message = SlackMessage(text="Test message")
        webhook_url = "https://hooks.slack.com/test"
        
        # Mock httpx response with error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await slack_service_instance._send_slack_message(webhook_url, message)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_send_slack_message_exception(self, slack_service_instance):
        """Test Slack message sending with network exception."""
        message = SlackMessage(text="Test message")
        webhook_url = "https://hooks.slack.com/test"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Network error")
            
            result = await slack_service_instance._send_slack_message(webhook_url, message)
            assert result is False


class TestConvenienceMethods:
    """Test convenience methods for common notifications."""
    
    @pytest.mark.asyncio
    async def test_send_alert(self, slack_service_instance):
        """Test send_alert convenience method."""
        with patch.object(slack_service_instance, 'send_notification') as mock_send:
            mock_send.return_value = True
            
            result = await slack_service_instance.send_alert(
                "Test Alert",
                "Alert message",
                "test_team",
                NotificationPriority.HIGH
            )
            
            assert result is True
            mock_send.assert_called_once()
            
            # Check call arguments
            call_args = mock_send.call_args
            assert call_args[0][0] == NotificationType.ALERT
            assert call_args[0][2] == "test_team"
            assert call_args[0][3] == NotificationPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_send_evaluation_complete(self, slack_service_instance):
        """Test send_evaluation_complete convenience method."""
        with patch.object(slack_service_instance, 'send_notification') as mock_send:
            mock_send.return_value = True
            
            result = await slack_service_instance.send_evaluation_complete(
                trace_count=100,
                acceptance_rate=0.9,
                team_id="test_team",
                evaluator_name="John Doe"
            )
            
            assert result is True
            mock_send.assert_called_once()
            
            call_args = mock_send.call_args
            assert call_args[0][0] == NotificationType.EVALUATION_COMPLETE
            data = call_args[0][1]
            assert data["trace_count"] == 100
            assert data["acceptance_rate"] == 0.9
            assert data["evaluator_name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_send_experiment_result(self, slack_service_instance):
        """Test send_experiment_result convenience method."""
        with patch.object(slack_service_instance, 'send_notification') as mock_send:
            mock_send.return_value = True
            
            variant_results = {"control": 0.15, "variant": 0.18}
            
            result = await slack_service_instance.send_experiment_result(
                "Test Experiment",
                variant_results,
                0.95,
                "test_team"
            )
            
            assert result is True
            mock_send.assert_called_once()
            
            call_args = mock_send.call_args
            assert call_args[0][0] == NotificationType.EXPERIMENT_RESULT
            data = call_args[0][1]
            assert data["experiment_name"] == "Test Experiment"
            assert data["variant_results"] == variant_results
            assert data["significance"] == 0.95


class TestWebhookTesting:
    """Test webhook testing functionality."""
    
    @pytest.mark.asyncio
    async def test_test_webhook_success(self, slack_service_instance):
        """Test successful webhook testing."""
        webhook_url = "https://hooks.slack.com/test"
        
        with patch.object(slack_service_instance, '_send_slack_message') as mock_send:
            mock_send.return_value = True
            
            result = await slack_service_instance.test_webhook(webhook_url)
            
            assert result["success"] is True
            assert result["webhook_url"] == webhook_url
            assert "Test notification sent successfully" in result["message"]
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_webhook_failure(self, slack_service_instance):
        """Test failed webhook testing."""
        webhook_url = "https://hooks.slack.com/test"
        
        with patch.object(slack_service_instance, '_send_slack_message') as mock_send:
            mock_send.return_value = False
            
            result = await slack_service_instance.test_webhook(webhook_url)
            
            assert result["success"] is False
            assert result["webhook_url"] == webhook_url
            assert "Failed to send test notification" in result["message"]
    
    @pytest.mark.asyncio
    async def test_test_webhook_exception(self, slack_service_instance):
        """Test webhook testing with exception."""
        webhook_url = "https://hooks.slack.com/test"
        
        with patch.object(slack_service_instance, '_send_slack_message') as mock_send:
            mock_send.side_effect = Exception("Network error")
            
            result = await slack_service_instance.test_webhook(webhook_url)
            
            assert result["success"] is False
            assert "Webhook test failed" in result["message"]
            assert "Network error" in result["message"]


class TestIntegrationScenarios:
    """Test full integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_notification_flow(self, slack_service_instance):
        """Test complete notification flow with all checks."""
        # Mock all dependencies
        with patch.object(slack_service_instance, '_get_notification_preferences') as mock_prefs, \
             patch.object(slack_service_instance, '_check_rate_limit') as mock_rate, \
             patch.object(slack_service_instance, '_send_slack_message') as mock_send, \
             patch.object(slack_service_instance, '_record_notification') as mock_record:
            
            # Setup mocks
            prefs = NotificationPreferences(
                team_id="test",
                enabled=True,
                webhook_url="https://hooks.slack.com/test",
                notification_types=[NotificationType.ALERT],
                priority_filter=NotificationPriority.LOW
            )
            mock_prefs.return_value = prefs
            mock_rate.return_value = True
            mock_send.return_value = True
            
            # Send notification
            result = await slack_service_instance.send_notification(
                NotificationType.ALERT,
                {"title": "Test", "message": "Test message", "timestamp": datetime.utcnow().isoformat()},
                "test_team",
                NotificationPriority.MEDIUM
            )
            
            # Verify flow
            assert result is True
            mock_prefs.assert_called_once()
            mock_rate.assert_called_once()
            mock_send.assert_called_once()
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_notification_filtered_by_type(self, slack_service_instance):
        """Test notification filtering by type."""
        with patch.object(slack_service_instance, '_get_notification_preferences') as mock_prefs:
            prefs = NotificationPreferences(
                team_id="test",
                enabled=True,
                notification_types=[NotificationType.ALERT],  # Only alerts enabled
                priority_filter=NotificationPriority.LOW
            )
            mock_prefs.return_value = prefs
            
            # Try to send evaluation notification (not in allowed types)
            result = await slack_service_instance.send_notification(
                NotificationType.EVALUATION_COMPLETE,
                {"trace_count": 100, "acceptance_rate": 0.9, "timestamp": datetime.utcnow().isoformat()},
                "test_team",
                NotificationPriority.MEDIUM
            )
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_notification_filtered_by_priority(self, slack_service_instance):
        """Test notification filtering by priority."""
        with patch.object(slack_service_instance, '_get_notification_preferences') as mock_prefs:
            prefs = NotificationPreferences(
                team_id="test",
                enabled=True,
                notification_types=[NotificationType.ALERT],
                priority_filter=NotificationPriority.HIGH  # Only high+ priority
            )
            mock_prefs.return_value = prefs
            
            # Try to send low priority notification
            result = await slack_service_instance.send_notification(
                NotificationType.ALERT,
                {"title": "Test", "message": "Test", "timestamp": datetime.utcnow().isoformat()},
                "test_team",
                NotificationPriority.LOW  # Below threshold
            )
            
            assert result is False


# API Tests
class TestSlackAPI:
    """Test Slack API endpoints."""
    
    def test_get_notification_types(self, client):
        """Test getting available notification types."""
        # Mock authentication
        with patch('auth.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = MagicMock(team_id="test_team")
            
            response = client.get("/api/v1/slack/notification-types")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure of first item
            first_type = data[0]
            assert "type" in first_type
            assert "name" in first_type
            assert "description" in first_type
            assert "priority" in first_type
            assert "icon" in first_type
    
    def test_get_preferences_unauthorized(self, client):
        """Test getting preferences without authentication."""
        response = client.get("/api/v1/slack/preferences")
        assert response.status_code == 401
    
    def test_quick_setup_invalid_webhook(self, client):
        """Test quick setup with invalid webhook URL."""
        with patch('auth.dependencies.require_permission') as mock_auth:
            mock_auth.return_value = MagicMock(team_id="test_team")
            
            response = client.post("/api/v1/slack/quick-setup", json={
                "webhook_url": "invalid-url",
                "enabled": True
            })
            
            assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 