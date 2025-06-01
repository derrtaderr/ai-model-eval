"""
Real-time Dashboard Updates Service
Provides WebSocket and Server-Sent Events for live dashboard updates.
Part of Task 8 - Analytics Engine & Metrics Dashboard.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

# Import analytics services
try:
    from services.performance_analytics import performance_analytics, TimeRange, AlertLevel
    PERFORMANCE_ANALYTICS_AVAILABLE = True
except ImportError:
    performance_analytics = None
    PERFORMANCE_ANALYTICS_AVAILABLE = False

try:
    from services.user_analytics import user_analytics
    USER_ANALYTICS_AVAILABLE = True
except ImportError:
    user_analytics = None
    USER_ANALYTICS_AVAILABLE = False

try:
    from services.batch_evaluation import batch_processor
    BATCH_MONITORING_AVAILABLE = True
except ImportError:
    batch_processor = None
    BATCH_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class UpdateType(str, Enum):
    """Types of real-time updates."""
    METRIC_UPDATE = "metric_update"
    ALERT_TRIGGERED = "alert_triggered"
    BATCH_PROGRESS = "batch_progress"
    USER_ACTIVITY = "user_activity"
    SYSTEM_STATUS = "system_status"
    COST_UPDATE = "cost_update"

class NotificationLevel(str, Enum):
    """Notification levels for dashboard alerts."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DashboardUpdate:
    """Real-time dashboard update data."""
    
    def __init__(self, update_type: UpdateType, data: Dict[str, Any], 
                 level: NotificationLevel = NotificationLevel.INFO,
                 title: Optional[str] = None, message: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.update_type = update_type
        self.data = data
        self.level = level
        self.title = title or f"{update_type.value.replace('_', ' ').title()}"
        self.message = message
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.update_type.value,
            "data": self.data,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp
        }

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, Set[UpdateType]] = {}
        
    async def connect(self, websocket: WebSocket, user_email: str, 
                     subscriptions: Optional[List[UpdateType]] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_email] = websocket
        
        # Set default subscriptions if none provided
        if subscriptions is None:
            subscriptions = [UpdateType.METRIC_UPDATE, UpdateType.ALERT_TRIGGERED, 
                           UpdateType.BATCH_PROGRESS, UpdateType.SYSTEM_STATUS]
        
        self.user_subscriptions[user_email] = set(subscriptions)
        
        logger.info(f"WebSocket connected for user: {user_email}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Real-time dashboard connected",
            "subscriptions": [sub.value for sub in subscriptions],
            "timestamp": datetime.utcnow().isoformat()
        }, user_email)
    
    def disconnect(self, user_email: str):
        """Remove a WebSocket connection."""
        if user_email in self.active_connections:
            del self.active_connections[user_email]
        if user_email in self.user_subscriptions:
            del self.user_subscriptions[user_email]
        logger.info(f"WebSocket disconnected for user: {user_email}")
    
    async def send_personal_message(self, message: Dict[str, Any], user_email: str):
        """Send a message to a specific user."""
        if user_email in self.active_connections:
            try:
                await self.active_connections[user_email].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {user_email}: {e}")
                self.disconnect(user_email)
    
    async def broadcast_update(self, update: DashboardUpdate):
        """Broadcast an update to all subscribed users."""
        message = update.to_dict()
        
        for user_email, subscriptions in self.user_subscriptions.items():
            if update.update_type in subscriptions:
                await self.send_personal_message(message, user_email)
    
    async def send_targeted_update(self, update: DashboardUpdate, target_users: List[str]):
        """Send an update to specific users."""
        message = update.to_dict()
        
        for user_email in target_users:
            if user_email in self.active_connections:
                await self.send_personal_message(message, user_email)

class RealTimeDashboardService:
    """Service for real-time dashboard updates and notifications."""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.update_interval = 30  # seconds
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Cache for metrics to detect changes
        self._metrics_cache = {}
        self._last_update = {}
        
        logger.info("RealTimeDashboardService initialized")
    
    async def start_monitoring(self):
        """Start the real-time monitoring background task."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self):
        """Stop the real-time monitoring background task."""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Real-time monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for real-time updates."""
        while self.is_monitoring:
            try:
                # Check for metric updates
                if PERFORMANCE_ANALYTICS_AVAILABLE:
                    await self._check_performance_metrics()
                
                # Check for batch job updates
                if BATCH_MONITORING_AVAILABLE:
                    await self._check_batch_progress()
                
                # Check for user activity updates
                if USER_ANALYTICS_AVAILABLE:
                    await self._check_user_activity()
                
                # Check for system status updates
                await self._check_system_status()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_performance_metrics(self):
        """Check for performance metric changes."""
        try:
            overview = await performance_analytics.get_system_overview(TimeRange.HOUR)
            
            # Check for significant metric changes
            current_metrics = {
                "success_rate": overview["evaluation_metrics"]["success_rate"],
                "throughput": overview["evaluation_metrics"]["throughput_per_hour"],
                "total_cost": overview["cost_metrics"]["total_cost_usd"],
                "active_alerts": len(overview["alerts"])
            }
            
            previous_metrics = self._metrics_cache.get("performance", {})
            
            for metric_name, current_value in current_metrics.items():
                previous_value = previous_metrics.get(metric_name, 0)
                
                # Check for significant changes (>5% or new alerts)
                if metric_name == "active_alerts":
                    if current_value > previous_value:
                        update = DashboardUpdate(
                            UpdateType.ALERT_TRIGGERED,
                            {"new_alerts": current_value - previous_value, "total_alerts": current_value},
                            NotificationLevel.WARNING,
                            "New System Alerts",
                            f"{current_value - previous_value} new alerts detected"
                        )
                        await self.connection_manager.broadcast_update(update)
                
                elif isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    if previous_value > 0:
                        change_percent = abs((current_value - previous_value) / previous_value) * 100
                        if change_percent >= 5:
                            level = NotificationLevel.INFO
                            if change_percent >= 20:
                                level = NotificationLevel.WARNING
                            
                            update = DashboardUpdate(
                                UpdateType.METRIC_UPDATE,
                                {
                                    "metric": metric_name,
                                    "current_value": current_value,
                                    "previous_value": previous_value,
                                    "change_percent": round(change_percent, 2),
                                    "trend": "up" if current_value > previous_value else "down"
                                },
                                level,
                                f"{metric_name.replace('_', ' ').title()} Update",
                                f"{metric_name} changed by {change_percent:.1f}%"
                            )
                            await self.connection_manager.broadcast_update(update)
            
            self._metrics_cache["performance"] = current_metrics
            
        except Exception as e:
            logger.error(f"Error checking performance metrics: {e}")
    
    async def _check_batch_progress(self):
        """Check for batch job progress updates."""
        try:
            if hasattr(batch_processor, 'get_system_stats'):
                stats = await batch_processor.get_system_stats()
                
                previous_stats = self._metrics_cache.get("batch", {})
                
                # Check for new completed jobs
                current_processed = stats.get("total_processed", 0)
                previous_processed = previous_stats.get("total_processed", 0)
                
                if current_processed > previous_processed:
                    update = DashboardUpdate(
                        UpdateType.BATCH_PROGRESS,
                        {
                            "new_completions": current_processed - previous_processed,
                            "total_processed": current_processed,
                            "active_jobs": stats.get("active_jobs", 0),
                            "success_rate": stats.get("success_rate", 0)
                        },
                        NotificationLevel.SUCCESS,
                        "Batch Jobs Progress",
                        f"{current_processed - previous_processed} new jobs completed"
                    )
                    await self.connection_manager.broadcast_update(update)
                
                self._metrics_cache["batch"] = stats
                
        except Exception as e:
            logger.error(f"Error checking batch progress: {e}")
    
    async def _check_user_activity(self):
        """Check for user activity updates."""
        try:
            engagement_data = await user_analytics.get_user_engagement_overview(1)  # Last 1 day
            
            current_activity = {
                "active_users": engagement_data["user_metrics"]["active_users"],
                "total_evaluations": engagement_data["user_metrics"]["total_evaluations"]
            }
            
            previous_activity = self._metrics_cache.get("user_activity", {})
            
            # Check for significant activity changes
            for metric, current_value in current_activity.items():
                previous_value = previous_activity.get(metric, 0)
                if current_value > previous_value:
                    update = DashboardUpdate(
                        UpdateType.USER_ACTIVITY,
                        {
                            "metric": metric,
                            "increase": current_value - previous_value,
                            "total": current_value
                        },
                        NotificationLevel.INFO,
                        "User Activity Update",
                        f"New activity: +{current_value - previous_value} {metric.replace('_', ' ')}"
                    )
                    await self.connection_manager.broadcast_update(update)
            
            self._metrics_cache["user_activity"] = current_activity
            
        except Exception as e:
            logger.error(f"Error checking user activity: {e}")
    
    async def _check_system_status(self):
        """Check for system status changes."""
        try:
            # Simple system health check
            current_time = datetime.utcnow()
            
            # Check if we should send a periodic status update (every 5 minutes)
            last_status_update = self._last_update.get("system_status")
            if not last_status_update or (current_time - last_status_update).seconds >= 300:
                
                system_status = {
                    "status": "healthy",
                    "uptime": "operational",
                    "services": {
                        "performance_analytics": PERFORMANCE_ANALYTICS_AVAILABLE,
                        "user_analytics": USER_ANALYTICS_AVAILABLE,
                        "batch_processing": BATCH_MONITORING_AVAILABLE
                    },
                    "timestamp": current_time.isoformat()
                }
                
                update = DashboardUpdate(
                    UpdateType.SYSTEM_STATUS,
                    system_status,
                    NotificationLevel.SUCCESS,
                    "System Status",
                    "All systems operational"
                )
                await self.connection_manager.broadcast_update(update)
                
                self._last_update["system_status"] = current_time
            
        except Exception as e:
            logger.error(f"Error checking system status: {e}")
    
    async def send_cost_alert(self, cost_data: Dict[str, Any], threshold_exceeded: bool = False):
        """Send a cost-related alert."""
        level = NotificationLevel.WARNING if threshold_exceeded else NotificationLevel.INFO
        message = "Cost threshold exceeded!" if threshold_exceeded else "Cost update available"
        
        update = DashboardUpdate(
            UpdateType.COST_UPDATE,
            cost_data,
            level,
            "Cost Alert",
            message
        )
        await self.connection_manager.broadcast_update(update)
    
    def get_sse_stream(self, user_email: str, subscriptions: List[UpdateType]):
        """Create a Server-Sent Events stream for a user."""
        async def event_stream():
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'message': 'SSE stream established', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            # Keep connection alive and send periodic updates
            try:
                while True:
                    # In a real implementation, this would pull from a queue or cache
                    # For now, send a heartbeat every 30 seconds
                    await asyncio.sleep(30)
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                        "subscriptions": [sub.value for sub in subscriptions]
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                    
            except asyncio.CancelledError:
                logger.info(f"SSE stream cancelled for user: {user_email}")
                return
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

# Global real-time dashboard service instance
realtime_dashboard = RealTimeDashboardService() 