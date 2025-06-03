"""
A/B Testing Data Models
Comprehensive models for experiment management, traffic allocation, and statistical analysis.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

from database.models import Base

class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class VariantType(str, Enum):
    """Types of experiment variants."""
    CONTROL = "control"
    TREATMENT = "treatment"

class AllocationMethod(str, Enum):
    """Traffic allocation methods."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    DETERMINISTIC = "deterministic"
    COHORT_BASED = "cohort_based"

class StatisticalTest(str, Enum):
    """Statistical significance test methods."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    BAYESIAN = "bayesian"

class MetricType(str, Enum):
    """Experiment metric types."""
    CONVERSION_RATE = "conversion_rate"
    LATENCY = "latency"
    ACCURACY = "accuracy"
    USER_SATISFACTION = "user_satisfaction"
    COST_PER_REQUEST = "cost_per_request"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"

# Database Models

class Experiment(Base):
    """Main experiment configuration."""
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    hypothesis = Column(Text)
    
    # Configuration
    status = Column(String(50), default=ExperimentStatus.DRAFT)
    allocation_method = Column(String(50), default=AllocationMethod.RANDOM)
    traffic_percentage = Column(Float, default=100.0)  # Percentage of total traffic
    
    # Timing
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    duration_days = Column(Integer)  # Expected duration
    
    # Statistical configuration
    primary_metric = Column(String(100), nullable=False)
    secondary_metrics = Column(JSON)  # List of additional metrics
    statistical_test = Column(String(50), default=StatisticalTest.T_TEST)
    confidence_level = Column(Float, default=0.95)
    minimum_detectable_effect = Column(Float, default=0.05)  # 5% minimum effect
    
    # Sample size calculation
    required_sample_size = Column(Integer)
    power = Column(Float, default=0.8)  # Statistical power
    
    # Targeting and filters
    targeting_rules = Column(JSON)  # User segments, geographic filters, etc.
    exclusion_rules = Column(JSON)  # Exclusion criteria
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    variants = relationship("ExperimentVariant", back_populates="experiment", cascade="all, delete-orphan")
    assignments = relationship("ParticipantAssignment", back_populates="experiment")
    results = relationship("ExperimentResult", back_populates="experiment")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiments_team_status", "team_id", "status"),
        Index("idx_experiments_dates", "start_date", "end_date"),
        Index("idx_experiments_active", "status", "start_date", "end_date"),
    )

class ExperimentVariant(Base):
    """Experiment variant configurations."""
    __tablename__ = "experiment_variants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    
    # Variant information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    variant_type = Column(String(20), default=VariantType.TREATMENT)
    
    # Traffic allocation
    traffic_weight = Column(Float, default=50.0)  # Percentage allocation
    
    # Configuration
    configuration = Column(JSON, nullable=False)  # Variant-specific settings
    
    # Model/prompt configuration for LLM experiments
    model_name = Column(String(100))
    system_prompt = Column(Text)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    model_parameters = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="variants")
    assignments = relationship("ParticipantAssignment", back_populates="variant")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_variants_experiment", "experiment_id"),
        Index("idx_experiment_variants_active", "experiment_id", "is_active"),
    )

class ParticipantAssignment(Base):
    """Tracks which users are assigned to which experiment variants."""
    __tablename__ = "participant_assignments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    variant_id = Column(UUID(as_uuid=True), ForeignKey("experiment_variants.id"), nullable=False)
    
    # Participant identification
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))  # For logged-in users
    session_id = Column(String(255))  # For anonymous users
    participant_hash = Column(String(255), nullable=False)  # Stable hash for assignment
    
    # Assignment details
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assignment_method = Column(String(50))  # How they were assigned
    
    # Context at assignment
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    referrer = Column(String(500))
    assignment_context = Column(JSON)  # Additional context data
    
    # Relationships
    experiment = relationship("Experiment", back_populates="assignments")
    variant = relationship("ExperimentVariant", back_populates="assignments")
    
    # Indexes
    __table_args__ = (
        Index("idx_participant_assignments_experiment", "experiment_id"),
        Index("idx_participant_assignments_variant", "variant_id"),
        Index("idx_participant_assignments_user", "user_id"),
        Index("idx_participant_assignments_session", "session_id"),
        Index("idx_participant_assignments_hash", "participant_hash"),
        # Ensure one assignment per participant per experiment
        Index("idx_participant_assignments_unique", "experiment_id", "participant_hash", unique=True),
    )

class ExperimentEvent(Base):
    """Tracks events and conversions for experiment analysis."""
    __tablename__ = "experiment_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    variant_id = Column(UUID(as_uuid=True), ForeignKey("experiment_variants.id"), nullable=False)
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("participant_assignments.id"), nullable=False)
    
    # Event details
    event_type = Column(String(100), nullable=False)  # conversion, interaction, error, etc.
    event_value = Column(Float)  # Numeric value (latency, cost, score, etc.)
    event_metadata = Column(JSON)  # Additional event data
    
    # Timing
    occurred_at = Column(DateTime, default=datetime.utcnow)
    
    # Context
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id"))  # Link to specific trace
    
    # Relationships
    assignment = relationship("ParticipantAssignment")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_events_experiment", "experiment_id"),
        Index("idx_experiment_events_variant", "variant_id"),
        Index("idx_experiment_events_assignment", "assignment_id"),
        Index("idx_experiment_events_type_time", "event_type", "occurred_at"),
        Index("idx_experiment_events_trace", "trace_id"),
    )

class ExperimentResult(Base):
    """Stores calculated experiment results and statistical analysis."""
    __tablename__ = "experiment_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    
    # Analysis details
    metric_name = Column(String(100), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Sample sizes
    control_sample_size = Column(Integer)
    treatment_sample_size = Column(Integer)
    total_sample_size = Column(Integer)
    
    # Metric values
    control_mean = Column(Float)
    treatment_mean = Column(Float)
    control_std = Column(Float)
    treatment_std = Column(Float)
    
    # Effect size
    absolute_effect = Column(Float)
    relative_effect = Column(Float)  # Percentage change
    
    # Statistical significance
    p_value = Column(Float)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    is_significant = Column(Boolean)
    
    # Test details
    statistical_test_used = Column(String(50))
    test_statistic = Column(Float)
    degrees_of_freedom = Column(Integer)
    
    # Bayesian analysis (if applicable)
    bayesian_probability = Column(Float)  # Probability of treatment being better
    credible_interval_lower = Column(Float)
    credible_interval_upper = Column(Float)
    
    # Variance and power analysis
    pooled_variance = Column(Float)
    effect_size_cohens_d = Column(Float)
    observed_power = Column(Float)
    
    # Additional analysis
    analysis_metadata = Column(JSON)  # Additional analysis details
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index("idx_experiment_results_experiment", "experiment_id"),
        Index("idx_experiment_results_metric", "experiment_id", "metric_name"),
        Index("idx_experiment_results_significance", "is_significant", "p_value"),
    )

# Pydantic Models for API

class ExperimentCreate(BaseModel):
    """Schema for creating a new experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    hypothesis: Optional[str] = None
    primary_metric: str = Field(..., min_length=1)
    secondary_metrics: Optional[List[str]] = []
    traffic_percentage: float = Field(default=100.0, ge=1.0, le=100.0)
    duration_days: Optional[int] = Field(None, ge=1, le=365)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    minimum_detectable_effect: float = Field(default=0.05, ge=0.01, le=1.0)
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    targeting_rules: Optional[Dict[str, Any]] = None
    exclusion_rules: Optional[Dict[str, Any]] = None

class VariantCreate(BaseModel):
    """Schema for creating experiment variants."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    variant_type: VariantType = VariantType.TREATMENT
    traffic_weight: float = Field(..., ge=0.1, le=100.0)
    configuration: Dict[str, Any] = Field(..., min_length=1)
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    model_parameters: Optional[Dict[str, Any]] = None

class ExperimentResponse(BaseModel):
    """Schema for experiment responses."""
    id: str
    name: str
    description: Optional[str]
    status: str
    primary_metric: str
    traffic_percentage: float
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    created_at: datetime
    required_sample_size: Optional[int]
    variants_count: int = 0

    class Config:
        from_attributes = True

class VariantResponse(BaseModel):
    """Schema for variant responses."""
    id: str
    name: str
    description: Optional[str]
    variant_type: str
    traffic_weight: float
    configuration: Dict[str, Any]
    model_name: Optional[str]
    participants_count: int = 0

    class Config:
        from_attributes = True

class AssignmentResponse(BaseModel):
    """Schema for participant assignment responses."""
    experiment_id: str
    variant_id: str
    variant_name: str
    assigned_at: datetime
    configuration: Dict[str, Any]

class ExperimentResultResponse(BaseModel):
    """Schema for experiment result responses."""
    id: str
    metric_name: str
    control_sample_size: int
    treatment_sample_size: int
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    p_value: float
    is_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
    analysis_date: datetime

    class Config:
        from_attributes = True

class ExperimentAnalytics(BaseModel):
    """Schema for experiment analytics dashboard."""
    experiment_id: str
    experiment_name: str
    status: str
    total_participants: int
    conversion_rate_control: float
    conversion_rate_treatment: float
    statistical_significance: bool
    confidence_level: float
    days_running: int
    projected_end_date: Optional[datetime]
    results: List[ExperimentResultResponse] 