"""
Analytics Data Models
Comprehensive models for metrics calculation, trend analysis, and alerting.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from database.models import Base

class MetricType(str, Enum):
    """Types of metrics that can be calculated."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    QUALITY = "quality"
    COST = "cost"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    SATISFACTION = "satisfaction"

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"

class TrendDirection(str, Enum):
    """Trend direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"

# Database Models

class MetricDefinition(Base):
    """Defines a metric that can be calculated."""
    __tablename__ = "metric_definitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Metric identification
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    metric_type = Column(String(50), nullable=False)  # MetricType enum
    unit = Column(String(50))  # e.g., "ms", "%", "$", "count"
    
    # Calculation configuration
    calculation_query = Column(Text, nullable=False)  # SQL query or function name
    calculation_interval_minutes = Column(Integer, default=5)  # How often to calculate
    aggregation_method = Column(String(50), default="avg")  # avg, sum, count, min, max
    
    # Display and alerting
    is_active = Column(Boolean, default=True)
    is_alertable = Column(Boolean, default=False)
    display_order = Column(Integer, default=0)
    
    # Thresholds for alerting
    warning_threshold = Column(Float)
    critical_threshold = Column(Float)
    threshold_direction = Column(String(10), default="above")  # above, below
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    metric_values = relationship("MetricValue", back_populates="metric_definition", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="metric_definition")
    
    # Indexes
    __table_args__ = (
        Index("idx_metric_definitions_team_id", "team_id"),
        Index("idx_metric_definitions_name", "team_id", "name"),
        Index("idx_metric_definitions_type", "metric_type"),
        Index("idx_metric_definitions_active", "is_active"),
    )

class MetricValue(Base):
    """Stores calculated metric values over time."""
    __tablename__ = "metric_values"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_definition_id = Column(UUID(as_uuid=True), ForeignKey("metric_definitions.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Metric data
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Contextual data
    dimensions = Column(JSON)  # Additional dimensions like model_name, user_id, etc.
    metric_metadata = Column(JSON)   # Calculation metadata, sample size, etc.
    
    # Relationships
    metric_definition = relationship("MetricDefinition", back_populates="metric_values")
    
    # Indexes for time-series queries
    __table_args__ = (
        Index("idx_metric_values_metric_timestamp", "metric_definition_id", "timestamp"),
        Index("idx_metric_values_team_timestamp", "team_id", "timestamp"),
        Index("idx_metric_values_timestamp", "timestamp"),
    )

class TrendAnalysis(Base):
    """Stores trend analysis results for metrics."""
    __tablename__ = "trend_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_definition_id = Column(UUID(as_uuid=True), ForeignKey("metric_definitions.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Analysis period
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Trend results
    direction = Column(String(20), nullable=False)  # TrendDirection enum
    strength = Column(Float)  # 0-1, how strong the trend is
    confidence = Column(Float)  # 0-1, confidence in the trend
    
    # Statistical data
    slope = Column(Float)  # Rate of change
    r_squared = Column(Float)  # Goodness of fit
    data_points = Column(Integer)  # Number of data points analyzed
    
    # Change metrics
    absolute_change = Column(Float)
    percentage_change = Column(Float)
    volatility = Column(Float)  # Standard deviation
    
    # Analysis metadata
    algorithm_used = Column(String(100))  # Linear regression, etc.
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    metric_definition = relationship("MetricDefinition")
    
    # Indexes
    __table_args__ = (
        Index("idx_trend_analyses_metric_date", "metric_definition_id", "end_date"),
        Index("idx_trend_analyses_team_date", "team_id", "end_date"),
    )

class Alert(Base):
    """Stores alerts triggered by metric thresholds."""
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_definition_id = Column(UUID(as_uuid=True), ForeignKey("metric_definitions.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Alert details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    severity = Column(String(20), nullable=False)  # AlertSeverity enum
    status = Column(String(20), default="active")  # AlertStatus enum
    
    # Trigger information
    triggered_at = Column(DateTime, default=datetime.utcnow)
    trigger_value = Column(Float, nullable=False)
    threshold_value = Column(Float, nullable=False)
    threshold_type = Column(String(20), nullable=False)  # warning, critical
    
    # Resolution information
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    resolved_at = Column(DateTime)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    resolution_notes = Column(Text)
    
    # Notification tracking
    notifications_sent = Column(JSON)  # Track which notifications were sent
    
    # Relationships
    metric_definition = relationship("MetricDefinition", back_populates="alerts")
    acknowledged_by_user = relationship("User", foreign_keys=[acknowledged_by])
    resolved_by_user = relationship("User", foreign_keys=[resolved_by])
    
    # Indexes
    __table_args__ = (
        Index("idx_alerts_team_status", "team_id", "status"),
        Index("idx_alerts_metric_triggered", "metric_definition_id", "triggered_at"),
        Index("idx_alerts_severity_status", "severity", "status"),
        Index("idx_alerts_triggered_at", "triggered_at"),
    )

class DashboardConfig(Base):
    """Stores dashboard configuration for teams."""
    __tablename__ = "dashboard_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Dashboard details
    name = Column(String(255), nullable=False)
    description = Column(Text)
    is_default = Column(Boolean, default=False)
    
    # Layout configuration
    layout_config = Column(JSON, nullable=False)  # Grid layout, widget positions
    widget_configs = Column(JSON, nullable=False)  # Widget settings, metrics shown
    
    # Access control
    is_public = Column(Boolean, default=False)
    shared_with_teams = Column(JSON)  # List of team IDs
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Indexes
    __table_args__ = (
        Index("idx_dashboard_configs_team", "team_id"),
        Index("idx_dashboard_configs_default", "team_id", "is_default"),
    )

# Pydantic Models for API

class MetricDefinitionCreate(BaseModel):
    """Schema for creating a new metric definition."""
    name: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    metric_type: MetricType
    unit: Optional[str] = None
    calculation_query: str = Field(..., min_length=1)
    calculation_interval_minutes: int = Field(default=5, ge=1, le=1440)
    aggregation_method: str = Field(default="avg")
    is_alertable: bool = Field(default=False)
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    threshold_direction: str = Field(default="above")

class MetricDefinitionResponse(BaseModel):
    """Schema for metric definition responses."""
    id: str
    name: str
    display_name: str
    description: Optional[str]
    metric_type: str
    unit: Optional[str]
    calculation_interval_minutes: int
    is_active: bool
    is_alertable: bool
    warning_threshold: Optional[float]
    critical_threshold: Optional[float]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MetricValueResponse(BaseModel):
    """Schema for metric value responses."""
    id: str
    value: float
    timestamp: datetime
    dimensions: Optional[Dict[str, Any]] = None
    metric_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class TrendAnalysisResponse(BaseModel):
    """Schema for trend analysis responses."""
    id: str
    direction: str
    strength: float
    confidence: float
    slope: Optional[float]
    absolute_change: Optional[float]
    percentage_change: Optional[float]
    volatility: Optional[float]
    data_points: int
    start_date: datetime
    end_date: datetime
    analysis_timestamp: datetime

    class Config:
        from_attributes = True

class AlertCreate(BaseModel):
    """Schema for creating alerts."""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    severity: AlertSeverity

class AlertResponse(BaseModel):
    """Schema for alert responses."""
    id: str
    title: str
    description: Optional[str]
    severity: str
    status: str
    triggered_at: datetime
    trigger_value: float
    threshold_value: float
    threshold_type: str
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]

    class Config:
        from_attributes = True

class DashboardMetricsResponse(BaseModel):
    """Schema for dashboard metrics overview."""
    total_metrics: int
    active_alerts: int
    critical_alerts: int
    metrics_calculated_last_hour: int
    avg_response_time_ms: float
    error_rate_percent: float
    top_performing_models: List[Dict[str, Any]]
    recent_trends: List[TrendAnalysisResponse] 