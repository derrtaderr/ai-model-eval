"""
Database models for the LLM Evaluation Platform.
Based on the schema defined in the PRD:
- traces, evaluations, test_cases, experiments
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class TraceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_REVIEW = "in_review"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
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


class Trace(Base):
    __tablename__ = "traces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    system_prompt = Column(Text, nullable=True)
    user_input = Column(Text, nullable=False)
    model_output = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    token_count = Column(JSON, nullable=True)  # {"input": 123, "output": 456}
    cost_usd = Column(Float, nullable=True)
    status = Column(String(50), default=TraceStatus.COMPLETED, nullable=False, index=True)
    langsmith_run_id = Column(String(255), nullable=True, unique=True, index=True)
    
    # Relationships
    user = relationship("User")
    evaluations = relationship("Evaluation", back_populates="trace", cascade="all, delete-orphan")
    trace_tags = relationship("TraceTag", back_populates="trace", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trace_model_timestamp', 'model_name', 'timestamp'),
        Index('idx_trace_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_trace_session_timestamp', 'session_id', 'timestamp'),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id"), nullable=False, index=True)
    evaluator_type = Column(String(50), nullable=False)  # "human", "model", "automated"
    evaluator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    score = Column(Float, nullable=True)  # Numeric score if applicable
    label = Column(String(50), nullable=True)  # "accepted", "rejected", custom labels
    critique = Column(Text, nullable=True)  # Detailed feedback
    metadata = Column(JSON, nullable=True)  # Additional evaluation data
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    trace = relationship("Trace", back_populates="evaluations")
    evaluator = relationship("User", foreign_keys=[evaluator_id])


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
    
    # Relationships
    creator = relationship("User")
    test_runs = relationship("TestRun", back_populates="test_case")


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
    
    # Relationships
    test_case = relationship("TestCase", back_populates="test_runs")
    trace = relationship("Trace")
    executor = relationship("User")


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
    
    # Relationships
    creator = relationship("User")


class TraceTag(Base):
    __tablename__ = "trace_tags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.id"), nullable=False, index=True)
    tag_type = Column(String(50), nullable=False, index=True)  # "tool", "scenario", "topic", etc.
    tag_value = Column(String(255), nullable=False, index=True)
    confidence_score = Column(Float, nullable=True)  # For auto-generated tags
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trace = relationship("Trace", back_populates="trace_tags")
    creator = relationship("User")
    
    # Composite index for efficient filtering
    __table_args__ = (
        Index('idx_trace_tag_type_value', 'tag_type', 'tag_value'),
        Index('idx_trace_tag_trace_type', 'trace_id', 'tag_type'),
    ) 