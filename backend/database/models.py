"""
Database models for the LLM Evaluation Platform.
Based on the schema defined in the PRD:
- traces, evaluations, test_cases, experiments
- Enhanced with authentication models for multi-tenancy
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

# Import auth models and enums
from auth.models import (
    Team, TeamInvitation, APIKey, UserRole, TeamTier,
    user_team_association
)


class TraceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships for multi-tenancy
    teams = relationship("Team", secondary=user_team_association, back_populates="members")
    
    # Existing relationships
    traces = relationship("Trace", back_populates="user")
    evaluations = relationship("Evaluation", foreign_keys="Evaluation.evaluator_id", back_populates="evaluator")
    test_cases = relationship("TestCase", back_populates="creator")
    test_runs = relationship("TestRun", back_populates="executor")
    experiments = relationship("Experiment", back_populates="creator")
    trace_tags = relationship("TraceTag", back_populates="creator")
    filter_presets = relationship("FilterPreset", back_populates="user")


class Trace(Base):
    __tablename__ = "traces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    model_name = Column(String(255), nullable=False, index=True)
    system_prompt = Column(Text, nullable=True)
    user_input = Column(Text, nullable=False)
    model_output = Column(Text, nullable=False)
    trace_metadata = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    token_count = Column(JSON, nullable=True)  # {"input": 123, "output": 456}
    cost_usd = Column(Float, nullable=True)
    status = Column(String(50), default=TraceStatus.COMPLETED, nullable=False, index=True)
    langsmith_run_id = Column(String(255), nullable=True, unique=True, index=True)
    
    # Relationships
    user = relationship("User", back_populates="traces")
    team = relationship("Team")  # New relationship
    evaluations = relationship("Evaluation", back_populates="trace", cascade="all, delete-orphan")
    trace_tags = relationship("TraceTag", back_populates="trace", cascade="all, delete-orphan")
    
    # Performance indexes for filtering
    __table_args__ = (
        # Single column indexes for common filters
        Index('idx_traces_session_id', 'session_id'),
        Index('idx_traces_user_id', 'user_id'),
        Index('idx_traces_team_id', 'team_id'),  # New index for multi-tenancy
        Index('idx_traces_model_name', 'model_name'),
        Index('idx_traces_timestamp', 'timestamp'),
        Index('idx_traces_latency_ms', 'latency_ms'),
        Index('idx_traces_cost_usd', 'cost_usd'),
        
        # Composite indexes for multi-dimensional filtering
        Index('idx_traces_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_traces_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_traces_team_timestamp', 'team_id', 'timestamp'),  # New composite index
        Index('idx_traces_model_timestamp', 'model_name', 'timestamp'),
        Index('idx_traces_session_model', 'session_id', 'model_name'),
        Index('idx_traces_user_model', 'user_id', 'model_name'),
        Index('idx_traces_team_model', 'team_id', 'model_name'),  # New composite index
        
        # Performance range indexes
        Index('idx_traces_latency_timestamp', 'latency_ms', 'timestamp'),
        Index('idx_traces_cost_timestamp', 'cost_usd', 'timestamp'),
        
        # Team isolation indexes
        Index('idx_traces_team_user', 'team_id', 'user_id'),
        Index('idx_traces_team_session', 'team_id', 'session_id'),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id", ondelete="CASCADE"), nullable=False, index=True)
    evaluator_type = Column(String(50), nullable=False)  # "human", "model", "automated"
    evaluator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    score = Column(Float, nullable=True)  # Numeric score if applicable
    label = Column(String(50), nullable=True)  # "accepted", "rejected", custom labels
    critique = Column(Text, nullable=True)  # Detailed feedback
    eval_metadata = Column(JSON, nullable=True)  # Additional evaluation data
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    trace = relationship("Trace", back_populates="evaluations")
    evaluator = relationship("User", foreign_keys=[evaluator_id], back_populates="evaluations")
    team = relationship("Team")  # New relationship

    # Performance indexes for evaluation filtering
    __table_args__ = (
        Index('idx_evaluations_trace_id', 'trace_id'),
        Index('idx_evaluations_type', 'evaluator_type'),
        Index('idx_evaluations_score', 'score'),
        Index('idx_evaluations_evaluator', 'evaluator_id'),
        Index('idx_evaluations_team_id', 'team_id'),  # New index for multi-tenancy
        Index('idx_evaluations_date', 'evaluated_at'),
        
        # Composite indexes for evaluation analysis
        Index('idx_evaluations_trace_type', 'trace_id', 'evaluator_type'),
        Index('idx_evaluations_type_score', 'evaluator_type', 'score'),
        Index('idx_evaluations_evaluator_date', 'evaluator_id', 'evaluated_at'),
        Index('idx_evaluations_team_date', 'team_id', 'evaluated_at'),  # New composite index
        Index('idx_evaluations_type_date', 'evaluator_type', 'evaluated_at'),
        
        # Team isolation indexes
        Index('idx_evaluations_team_evaluator', 'team_id', 'evaluator_id'),
    )


class TestCase(Base):
    __tablename__ = "test_cases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    input_data = Column(JSON, nullable=False)  # Test input
    expected_output = Column(JSON, nullable=True)  # Expected result
    assertion_type = Column(String(100), nullable=False)  # "contains", "sentiment", etc.
    assertion_config = Column(JSON, nullable=True)  # Configuration for assertion
    tags = Column(JSON, nullable=True)  # ["regression", "critical", etc.]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    
    # Relationships
    creator = relationship("User", back_populates="test_cases")
    team = relationship("Team")  # New relationship
    test_runs = relationship("TestRun", back_populates="test_case")
    
    # Indexes for multi-tenancy
    __table_args__ = (
        Index('idx_test_cases_team_id', 'team_id'),
        Index('idx_test_cases_team_creator', 'team_id', 'created_by'),
    )


class TestRun(Base):
    __tablename__ = "test_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_case_id = Column(UUID(as_uuid=True), ForeignKey("test_cases.id"), nullable=False)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id"), nullable=True)
    status = Column(String(50), nullable=False)  # "passed", "failed", "error"
    result = Column(JSON, nullable=True)  # Test execution result
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)
    executed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    
    # Relationships
    test_case = relationship("TestCase", back_populates="test_runs")
    trace = relationship("Trace")
    executor = relationship("User", back_populates="test_runs")
    team = relationship("Team")  # New relationship
    
    # Indexes for multi-tenancy
    __table_args__ = (
        Index('idx_test_runs_team_id', 'team_id'),
        Index('idx_test_runs_team_executor', 'team_id', 'executed_by'),
    )


class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default=ExperimentStatus.DRAFT, nullable=False, index=True)
    config = Column(JSON, nullable=False)  # Experiment configuration
    metrics = Column(JSON, nullable=True)  # Results and metrics
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    
    # Relationships
    creator = relationship("User", back_populates="experiments")
    team = relationship("Team")  # New relationship
    
    # Indexes for multi-tenancy
    __table_args__ = (
        Index('idx_experiments_team_id', 'team_id'),
        Index('idx_experiments_team_creator', 'team_id', 'created_by'),
    )


class TraceTag(Base):
    __tablename__ = "trace_tags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id", ondelete="CASCADE"), nullable=False, index=True)
    tag_type = Column(String(50), nullable=False, index=True)  # "tool", "scenario", "topic", etc.
    tag_value = Column(String(255), nullable=False, index=True)
    confidence_score = Column(Float, nullable=True)  # For auto-generated tags
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    
    # Relationships
    trace = relationship("Trace", back_populates="trace_tags")
    creator = relationship("User", back_populates="trace_tags")
    team = relationship("Team")  # New relationship
    
    # Performance indexes for tag filtering
    __table_args__ = (
        Index('idx_trace_tags_trace_id', 'trace_id'),
        Index('idx_trace_tags_name', 'tag_type'),
        Index('idx_trace_tags_value', 'tag_value'),
        Index('idx_trace_tags_type', 'tag_type'),
        Index('idx_trace_tags_team_id', 'team_id'),  # New index for multi-tenancy
        
        # Composite indexes for tag queries
        Index('idx_trace_tags_trace_name', 'trace_id', 'tag_type'),
        Index('idx_trace_tags_name_value', 'tag_type', 'tag_value'),
        Index('idx_trace_tags_team_type', 'team_id', 'tag_type'),  # New composite index
        
        # Unique constraint for preventing duplicate tags
        Index('idx_trace_tags_unique', 'trace_id', 'tag_type', 'tag_value', unique=True),
    )


class FilterPreset(Base):
    __tablename__ = "filter_presets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True, index=True)  # New for multi-tenancy
    filter_config = Column(JSON, nullable=False)  # Complete filter configuration
    is_public = Column(Boolean, default=False)  # Whether preset can be shared
    is_default = Column(Boolean, default=False)  # Whether this is user's default preset
    usage_count = Column(Integer, default=0)  # Track how often preset is used
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="filter_presets")
    team = relationship("Team")  # New relationship
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_filter_preset_user_name', 'user_id', 'name'),
        Index('idx_filter_preset_team_id', 'team_id'),  # New index for multi-tenancy
        Index('idx_filter_preset_public', 'is_public'),
        Index('idx_filter_preset_user_default', 'user_id', 'is_default'),
        Index('idx_filter_preset_team_public', 'team_id', 'is_public'),  # New composite index
    )


# Import comprehensive A/B testing models (these will replace the basic Experiment model above)
try:
    from experiments.models import (
        Experiment as ABExperiment,
        ExperimentVariant,
        ParticipantAssignment, 
        ExperimentEvent,
        ExperimentResult
    )
except ImportError:
    # Models not yet available during initial setup
    pass 