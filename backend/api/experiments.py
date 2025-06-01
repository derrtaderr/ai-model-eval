"""
A/B Testing Framework API for LLM Evaluation Platform.
Provides comprehensive experiment management, traffic routing, and statistical analysis.
"""

import hashlib
import json
import logging
import math
import random
import statistics
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4

import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Request, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict, validator
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.connection import get_db
from database.models import Trace, Evaluation
from auth.security import get_current_user_email

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# EXPERIMENT ENUMS AND SCHEMAS
# ============================================================================

class ExperimentStatus(str, Enum):
    """Experiment status values."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class SegmentationType(str, Enum):
    """User segmentation types."""
    RANDOM = "random"
    COHORT = "cohort"
    ATTRIBUTE = "attribute"
    CUSTOM = "custom"

class MetricType(str, Enum):
    """Metric types for tracking."""
    CONVERSION_RATE = "conversion_rate"
    SATISFACTION_SCORE = "satisfaction_score"
    COMPLETION_RATE = "completion_rate"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    CUSTOM = "custom"

class StoppingRuleType(str, Enum):
    """Automated stopping rule types."""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    MINIMUM_EFFECT_SIZE = "minimum_effect_size"
    TIME_BASED = "time_based"
    SAMPLE_SIZE = "sample_size"

class ExperimentVariant(BaseModel):
    """Schema for experiment variants (control/treatment)."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(..., description="Unique variant identifier")
    name: str = Field(..., description="Human-readable variant name")
    description: Optional[str] = Field(None, description="Variant description")
    traffic_percentage: float = Field(..., ge=0, le=100, description="Traffic allocation percentage")
    model_config_override: Optional[Dict[str, Any]] = Field(None, description="Model configuration overrides")
    prompt_template_override: Optional[str] = Field(None, description="Custom prompt template")
    is_control: bool = Field(False, description="Whether this is the control group")

class SegmentationCriteria(BaseModel):
    """Schema for user segmentation criteria."""
    model_config = ConfigDict(protected_namespaces=())
    
    type: SegmentationType = Field(..., description="Type of segmentation")
    criteria: Dict[str, Any] = Field(..., description="Segmentation criteria")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage of users to include")

class MetricDefinition(BaseModel):
    """Schema for experiment metric definitions."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str = Field(..., description="Unique metric identifier")
    name: str = Field(..., description="Human-readable metric name")
    type: MetricType = Field(..., description="Type of metric")
    description: Optional[str] = Field(None, description="Metric description")
    aggregation_method: str = Field("mean", description="Aggregation method (mean, sum, count)")
    is_primary: bool = Field(False, description="Whether this is the primary metric")
    target_value: Optional[float] = Field(None, description="Target value for the metric")

class StoppingRule(BaseModel):
    """Schema for automated stopping rules."""
    model_config = ConfigDict(protected_namespaces=())
    
    type: StoppingRuleType = Field(..., description="Type of stopping rule")
    threshold: float = Field(..., description="Threshold value")
    enabled: bool = Field(True, description="Whether the rule is enabled")
    description: Optional[str] = Field(None, description="Rule description")

class ExperimentCreate(BaseModel):
    """Schema for creating new experiments."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    hypothesis: str = Field(..., description="Experiment hypothesis")
    variants: List[ExperimentVariant] = Field(..., min_items=2, description="Experiment variants")
    metrics: List[MetricDefinition] = Field(..., min_items=1, description="Metrics to track")
    segmentation: Optional[SegmentationCriteria] = Field(None, description="User segmentation criteria")
    stopping_rules: Optional[List[StoppingRule]] = Field(None, description="Automated stopping rules")
    target_sample_size: Optional[int] = Field(None, ge=1, description="Target sample size")
    estimated_duration_days: Optional[int] = Field(None, ge=1, description="Estimated duration in days")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Statistical confidence level")
    minimum_effect_size: Optional[float] = Field(None, ge=0, description="Minimum detectable effect size")
    
    @validator('variants')
    def validate_variants(cls, v):
        """Validate experiment variants."""
        total_percentage = sum(variant.traffic_percentage for variant in v)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError("Variant traffic percentages must sum to 100")
        
        control_count = sum(1 for variant in v if variant.is_control)
        if control_count != 1:
            raise ValueError("Exactly one variant must be marked as control")
        
        return v

class ExperimentResponse(BaseModel):
    """Schema for experiment responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    name: str
    description: Optional[str]
    hypothesis: str
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    metrics: List[MetricDefinition]
    segmentation: Optional[SegmentationCriteria]
    stopping_rules: Optional[List[StoppingRule]]
    target_sample_size: Optional[int]
    current_sample_size: int
    estimated_duration_days: Optional[int]
    confidence_level: float
    minimum_effect_size: Optional[float]
    created_at: str
    started_at: Optional[str]
    ended_at: Optional[str]
    created_by: str
    statistical_power: Optional[float]
    significance_reached: bool

class SampleSizeRequest(BaseModel):
    """Schema for sample size calculation requests."""
    model_config = ConfigDict(protected_namespaces=())
    
    baseline_conversion_rate: float = Field(..., ge=0, le=1, description="Expected baseline conversion rate")
    minimum_effect_size: float = Field(..., ge=0, description="Minimum detectable effect size")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Statistical confidence level")
    statistical_power: float = Field(0.8, ge=0.5, le=0.99, description="Statistical power")
    two_tailed: bool = Field(True, description="Whether to use two-tailed test")

class UserAssignment(BaseModel):
    """Schema for user assignment to experiments."""
    model_config = ConfigDict(protected_namespaces=())
    
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: Optional[str] = None
    user_attributes: Optional[Dict[str, Any]] = None

# ============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# ============================================================================

# Storage for experiments, assignments, and metrics
EXPERIMENTS = {}
USER_ASSIGNMENTS = {}
EXPERIMENT_METRICS = {}

# ============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ============================================================================

class StatisticalAnalyzer:
    """Statistical analysis utilities for A/B testing."""
    
    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        effect_size: float,
        confidence_level: float = 0.95,
        power: float = 0.8,
        two_tailed: bool = True
    ) -> int:
        """Calculate required sample size for A/B test."""
        
        alpha = 1 - confidence_level
        beta = 1 - power
        
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        # For conversion rate tests
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        
        # Pooled standard error
        p_pooled = (p1 + p2) / 2
        se_pooled = math.sqrt(2 * p_pooled * (1 - p_pooled))
        
        # Sample size calculation
        n = ((z_alpha + z_beta) * se_pooled / abs(p2 - p1)) ** 2
        
        return math.ceil(n)
    
    @staticmethod
    def t_test_two_sample(
        control_data: List[float],
        treatment_data: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform two-sample t-test."""
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                "error": "Insufficient data for statistical analysis",
                "p_value": None,
                "confidence_interval": None,
                "effect_size": None
            }
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
        
        # Calculate effect size (Cohen's d)
        control_mean = statistics.mean(control_data)
        treatment_mean = statistics.mean(treatment_data)
        
        pooled_std = math.sqrt(
            ((len(control_data) - 1) * statistics.stdev(control_data) ** 2 +
             (len(treatment_data) - 1) * statistics.stdev(treatment_data) ** 2) /
            (len(control_data) + len(treatment_data) - 2)
        )
        
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference in means
        alpha = 1 - confidence_level
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        se_diff = pooled_std * math.sqrt(1/len(control_data) + 1/len(treatment_data))
        mean_diff = treatment_mean - control_mean
        margin_error = t_critical * se_diff
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "degrees_freedom": df,
            "effect_size": effect_size,
            "mean_difference": mean_diff,
            "confidence_interval": [ci_lower, ci_upper],
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "control_size": len(control_data),
            "treatment_size": len(treatment_data),
            "significant": p_value < (1 - confidence_level)
        }
    
    @staticmethod
    def chi_square_test(
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform chi-square test for conversion rates."""
        
        # Create contingency table
        observed = np.array([
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions]
        ])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
        
        # Calculate conversion rates
        control_rate = control_conversions / control_total if control_total > 0 else 0
        treatment_rate = treatment_conversions / treatment_total if treatment_total > 0 else 0
        
        # Calculate confidence interval for difference in proportions
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        
        p1, p2 = control_rate, treatment_rate
        n1, n2 = control_total, treatment_total
        
        se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0
        diff = p2 - p1
        margin_error = z_critical * se_diff
        
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        return {
            "chi2_statistic": chi2_stat,
            "p_value": p_value,
            "degrees_freedom": dof,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "rate_difference": diff,
            "confidence_interval": [ci_lower, ci_upper],
            "relative_improvement": (diff / p1 * 100) if p1 > 0 else 0,
            "significant": p_value < (1 - confidence_level)
        }

# ============================================================================
# TRAFFIC ROUTING UTILITIES
# ============================================================================

class TrafficRouter:
    """Traffic routing and user assignment utilities."""
    
    @staticmethod
    def hash_user_id(user_id: str, experiment_id: str) -> float:
        """Generate consistent hash for user assignment."""
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_value[:8], 16) / (16**8)
    
    @staticmethod
    def assign_user_to_variant(
        user_id: str,
        experiment_id: str,
        variants: List[ExperimentVariant]
    ) -> str:
        """Assign user to experiment variant based on consistent hashing."""
        
        hash_value = TrafficRouter.hash_user_id(user_id, experiment_id)
        
        cumulative_percentage = 0
        for variant in variants:
            cumulative_percentage += variant.traffic_percentage
            if hash_value * 100 <= cumulative_percentage:
                return variant.id
        
        # Fallback to last variant
        return variants[-1].id
    
    @staticmethod
    def check_segmentation_criteria(
        user_attributes: Dict[str, Any],
        segmentation: Optional[SegmentationCriteria]
    ) -> bool:
        """Check if user meets segmentation criteria."""
        
        if not segmentation:
            return True
        
        if segmentation.type == SegmentationType.RANDOM:
            # Random sampling based on percentage
            if segmentation.percentage:
                return random.random() * 100 < segmentation.percentage
            return True
        
        elif segmentation.type == SegmentationType.ATTRIBUTE:
            # Attribute-based segmentation
            for key, expected_value in segmentation.criteria.items():
                if key not in user_attributes or user_attributes[key] != expected_value:
                    return False
            return True
        
        # Add more segmentation types as needed
        return True

# ============================================================================
# EXPERIMENT MANAGEMENT
# ============================================================================

class ExperimentManager:
    """Experiment management utilities."""
    
    @staticmethod
    def create_experiment(experiment_data: ExperimentCreate, user_email: str) -> str:
        """Create a new experiment."""
        
        experiment_id = str(uuid4())
        
        # Calculate statistical power if not provided
        statistical_power = None
        if experiment_data.target_sample_size and experiment_data.minimum_effect_size:
            # Simplified power calculation
            statistical_power = 0.8  # Default assumption
        
        experiment = {
            "id": experiment_id,
            "name": experiment_data.name,
            "description": experiment_data.description,
            "hypothesis": experiment_data.hypothesis,
            "status": ExperimentStatus.DRAFT,
            "variants": [variant.dict() for variant in experiment_data.variants],
            "metrics": [metric.dict() for metric in experiment_data.metrics],
            "segmentation": experiment_data.segmentation.dict() if experiment_data.segmentation else None,
            "stopping_rules": [rule.dict() for rule in experiment_data.stopping_rules] if experiment_data.stopping_rules else [],
            "target_sample_size": experiment_data.target_sample_size,
            "current_sample_size": 0,
            "estimated_duration_days": experiment_data.estimated_duration_days,
            "confidence_level": experiment_data.confidence_level,
            "minimum_effect_size": experiment_data.minimum_effect_size,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "ended_at": None,
            "created_by": user_email,
            "statistical_power": statistical_power,
            "significance_reached": False
        }
        
        EXPERIMENTS[experiment_id] = experiment
        EXPERIMENT_METRICS[experiment_id] = {"data": [], "assignments": []}
        
        return experiment_id
    
    @staticmethod
    def start_experiment(experiment_id: str) -> bool:
        """Start an experiment."""
        
        if experiment_id not in EXPERIMENTS:
            return False
        
        experiment = EXPERIMENTS[experiment_id]
        if experiment["status"] != ExperimentStatus.DRAFT:
            return False
        
        experiment["status"] = ExperimentStatus.RUNNING
        experiment["started_at"] = datetime.utcnow()
        
        return True
    
    @staticmethod
    def stop_experiment(experiment_id: str) -> bool:
        """Stop an experiment."""
        
        if experiment_id not in EXPERIMENTS:
            return False
        
        experiment = EXPERIMENTS[experiment_id]
        if experiment["status"] not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            return False
        
        experiment["status"] = ExperimentStatus.STOPPED
        experiment["ended_at"] = datetime.utcnow()
        
        return True

# Initialize analyzers and managers
analyzer = StatisticalAnalyzer()
router_util = TrafficRouter()
experiment_manager = ExperimentManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/experiments", response_model=Dict[str, Any])
async def create_experiment(
    experiment: ExperimentCreate,
    current_user: str = Depends(get_current_user_email)
):
    """Create a new A/B test experiment."""
    
    try:
        experiment_id = experiment_manager.create_experiment(experiment, current_user)
        
        return {
            "experiment_id": experiment_id,
            "message": "Experiment created successfully",
            "status": "draft"
        }
        
    except Exception as e:
        logger.error(f"Experiment creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    current_user: str = Depends(get_current_user_email)
):
    """List experiments for the current user."""
    
    user_experiments = []
    
    for exp_id, exp_data in EXPERIMENTS.items():
        if exp_data["created_by"] == current_user:
            if status is None or exp_data["status"] == status:
                
                experiment = ExperimentResponse(
                    id=exp_id,
                    name=exp_data["name"],
                    description=exp_data["description"],
                    hypothesis=exp_data["hypothesis"],
                    status=exp_data["status"],
                    variants=[ExperimentVariant(**v) for v in exp_data["variants"]],
                    metrics=[MetricDefinition(**m) for m in exp_data["metrics"]],
                    segmentation=SegmentationCriteria(**exp_data["segmentation"]) if exp_data["segmentation"] else None,
                    stopping_rules=[StoppingRule(**r) for r in exp_data["stopping_rules"]],
                    target_sample_size=exp_data["target_sample_size"],
                    current_sample_size=exp_data["current_sample_size"],
                    estimated_duration_days=exp_data["estimated_duration_days"],
                    confidence_level=exp_data["confidence_level"],
                    minimum_effect_size=exp_data["minimum_effect_size"],
                    created_at=exp_data["created_at"].isoformat(),
                    started_at=exp_data["started_at"].isoformat() if exp_data["started_at"] else None,
                    ended_at=exp_data["ended_at"].isoformat() if exp_data["ended_at"] else None,
                    created_by=exp_data["created_by"],
                    statistical_power=exp_data["statistical_power"],
                    significance_reached=exp_data["significance_reached"]
                )
                
                user_experiments.append(experiment)
    
    return user_experiments


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Start an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    if experiment["created_by"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if not experiment_manager.start_experiment(experiment_id):
        raise HTTPException(status_code=400, detail="Cannot start experiment in current state")
    
    return {"message": "Experiment started successfully"}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Stop an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    if experiment["created_by"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if not experiment_manager.stop_experiment(experiment_id):
        raise HTTPException(status_code=400, detail="Cannot stop experiment in current state")
    
    return {"message": "Experiment stopped successfully"}


@router.post("/sample-size", response_model=Dict[str, Any])
async def calculate_sample_size(
    request: SampleSizeRequest,
    current_user: str = Depends(get_current_user_email)
):
    """Calculate required sample size for A/B test."""
    
    try:
        sample_size = analyzer.calculate_sample_size(
            baseline_rate=request.baseline_conversion_rate,
            effect_size=request.minimum_effect_size,
            confidence_level=request.confidence_level,
            power=request.statistical_power,
            two_tailed=request.two_tailed
        )
        
        return {
            "sample_size_per_variant": sample_size,
            "total_sample_size": sample_size * 2,  # For two variants
            "parameters": {
                "baseline_conversion_rate": request.baseline_conversion_rate,
                "minimum_effect_size": request.minimum_effect_size,
                "confidence_level": request.confidence_level,
                "statistical_power": request.statistical_power,
                "two_tailed": request.two_tailed
            }
        }
        
    except Exception as e:
        logger.error(f"Sample size calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate sample size: {str(e)}")


@router.get("/health")
async def experiments_health():
    """Health check for experiments service."""
    
    return {
        "status": "healthy",
        "service": "A/B Testing Framework",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Experiment Setup Interface",
            "Traffic Routing & User Segmentation",
            "Statistical Analysis",
            "Sample Size Calculation"
        ],
        "active_experiments": len([e for e in EXPERIMENTS.values() if e["status"] == ExperimentStatus.RUNNING]),
        "total_experiments": len(EXPERIMENTS)
    }


# ============================================================================
# TRAFFIC ROUTING & USER ASSIGNMENT ENDPOINTS
# ============================================================================

@router.post("/experiments/{experiment_id}/assign-user", response_model=Dict[str, Any])
async def assign_user_to_experiment(
    experiment_id: str,
    user_id: str = Query(..., description="User ID to assign"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    user_attributes: Optional[Dict[str, Any]] = None,
    current_user: str = Depends(get_current_user_email)
):
    """Assign a user to an experiment variant."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    
    # Check if experiment is running
    if experiment["status"] != ExperimentStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Experiment is not running")
    
    # Check segmentation criteria
    segmentation = SegmentationCriteria(**experiment["segmentation"]) if experiment["segmentation"] else None
    if not router_util.check_segmentation_criteria(user_attributes or {}, segmentation):
        return {
            "assigned": False,
            "reason": "User does not meet segmentation criteria",
            "experiment_id": experiment_id,
            "user_id": user_id
        }
    
    # Check if user is already assigned
    assignment_key = f"{user_id}:{experiment_id}"
    if assignment_key in USER_ASSIGNMENTS:
        existing_assignment = USER_ASSIGNMENTS[assignment_key]
        return {
            "assigned": True,
            "variant_id": existing_assignment["variant_id"],
            "experiment_id": experiment_id,
            "user_id": user_id,
            "assigned_at": existing_assignment["assigned_at"].isoformat(),
            "existing_assignment": True
        }
    
    # Assign user to variant
    variants = [ExperimentVariant(**v) for v in experiment["variants"]]
    variant_id = router_util.assign_user_to_variant(user_id, experiment_id, variants)
    
    # Store assignment
    assignment = {
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "assigned_at": datetime.utcnow(),
        "session_id": session_id,
        "user_attributes": user_attributes
    }
    
    USER_ASSIGNMENTS[assignment_key] = assignment
    
    # Update experiment sample size
    experiment["current_sample_size"] += 1
    
    # Log assignment for tracking
    if experiment_id not in EXPERIMENT_METRICS:
        EXPERIMENT_METRICS[experiment_id] = {"data": [], "assignments": []}
    
    EXPERIMENT_METRICS[experiment_id]["assignments"].append(assignment)
    
    return {
        "assigned": True,
        "variant_id": variant_id,
        "experiment_id": experiment_id,
        "user_id": user_id,
        "assigned_at": assignment["assigned_at"].isoformat(),
        "existing_assignment": False
    }


@router.get("/experiments/{experiment_id}/assignment/{user_id}", response_model=Dict[str, Any])
async def get_user_assignment(
    experiment_id: str,
    user_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Get user's assignment for an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    assignment_key = f"{user_id}:{experiment_id}"
    if assignment_key not in USER_ASSIGNMENTS:
        return {
            "assigned": False,
            "experiment_id": experiment_id,
            "user_id": user_id
        }
    
    assignment = USER_ASSIGNMENTS[assignment_key]
    return {
        "assigned": True,
        "variant_id": assignment["variant_id"],
        "experiment_id": experiment_id,
        "user_id": user_id,
        "assigned_at": assignment["assigned_at"].isoformat(),
        "session_id": assignment.get("session_id"),
        "user_attributes": assignment.get("user_attributes")
    }


@router.post("/experiments/{experiment_id}/metrics", response_model=Dict[str, Any])
async def record_experiment_metric(
    experiment_id: str,
    user_id: str = Query(..., description="User ID"),
    metric_id: str = Query(..., description="Metric ID"),
    value: float = Query(..., description="Metric value"),
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: str = Depends(get_current_user_email)
):
    """Record a metric value for an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    
    # Verify user is assigned to experiment
    assignment_key = f"{user_id}:{experiment_id}"
    if assignment_key not in USER_ASSIGNMENTS:
        raise HTTPException(status_code=400, detail="User not assigned to experiment")
    
    assignment = USER_ASSIGNMENTS[assignment_key]
    
    # Record metric
    metric_record = {
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant_id": assignment["variant_id"],
        "metric_id": metric_id,
        "value": value,
        "timestamp": timestamp or datetime.utcnow(),
        "metadata": metadata or {}
    }
    
    if experiment_id not in EXPERIMENT_METRICS:
        EXPERIMENT_METRICS[experiment_id] = {"data": [], "assignments": []}
    
    EXPERIMENT_METRICS[experiment_id]["data"].append(metric_record)
    
    return {
        "recorded": True,
        "metric_record": {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant_id": assignment["variant_id"],
            "metric_id": metric_id,
            "value": value,
            "timestamp": metric_record["timestamp"].isoformat()
        }
    }


@router.get("/experiments/{experiment_id}/results", response_model=Dict[str, Any])
async def get_experiment_results(
    experiment_id: str,
    metric_id: Optional[str] = Query(None, description="Specific metric to analyze"),
    current_user: str = Depends(get_current_user_email)
):
    """Get statistical analysis results for an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    if experiment["created_by"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if experiment_id not in EXPERIMENT_METRICS:
        return {
            "experiment_id": experiment_id,
            "error": "No data available for analysis"
        }
    
    metrics_data = EXPERIMENT_METRICS[experiment_id]["data"]
    
    if not metrics_data:
        return {
            "experiment_id": experiment_id,
            "error": "No metrics recorded yet"
        }
    
    # Group data by variant and metric
    variant_data = {}
    for record in metrics_data:
        variant_id = record["variant_id"]
        record_metric_id = record["metric_id"]
        
        if metric_id and record_metric_id != metric_id:
            continue
        
        if variant_id not in variant_data:
            variant_data[variant_id] = {}
        
        if record_metric_id not in variant_data[variant_id]:
            variant_data[variant_id][record_metric_id] = []
        
        variant_data[variant_id][record_metric_id].append(record["value"])
    
    # Perform statistical analysis
    results = {
        "experiment_id": experiment_id,
        "experiment_name": experiment["name"],
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "total_participants": len(EXPERIMENT_METRICS[experiment_id]["assignments"]),
        "variants": [],
        "statistical_tests": {}
    }
    
    # Get control and treatment variants
    variants = [ExperimentVariant(**v) for v in experiment["variants"]]
    control_variant = next((v for v in variants if v.is_control), variants[0])
    treatment_variants = [v for v in variants if not v.is_control]
    
    # Analyze each metric
    for metric_name in set(record["metric_id"] for record in metrics_data):
        if metric_id and metric_name != metric_id:
            continue
        
        control_data = variant_data.get(control_variant.id, {}).get(metric_name, [])
        
        for treatment_variant in treatment_variants:
            treatment_data = variant_data.get(treatment_variant.id, {}).get(metric_name, [])
            
            if len(control_data) >= 2 and len(treatment_data) >= 2:
                # Perform t-test
                test_result = analyzer.t_test_two_sample(
                    control_data, 
                    treatment_data,
                    experiment["confidence_level"]
                )
                
                test_key = f"{metric_name}_{control_variant.id}_vs_{treatment_variant.id}"
                results["statistical_tests"][test_key] = {
                    "metric": metric_name,
                    "control_variant": control_variant.id,
                    "treatment_variant": treatment_variant.id,
                    "test_type": "t_test",
                    **test_result
                }
    
    # Add variant summaries
    for variant in variants:
        variant_summary = {
            "variant_id": variant.id,
            "variant_name": variant.name,
            "is_control": variant.is_control,
            "participant_count": len([a for a in EXPERIMENT_METRICS[experiment_id]["assignments"] 
                                   if a["variant_id"] == variant.id]),
            "metrics": {}
        }
        
        for metric_name in set(record["metric_id"] for record in metrics_data):
            if metric_id and metric_name != metric_id:
                continue
            
            metric_values = variant_data.get(variant.id, {}).get(metric_name, [])
            if metric_values:
                variant_summary["metrics"][metric_name] = {
                    "count": len(metric_values),
                    "mean": statistics.mean(metric_values),
                    "std": statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                    "min": min(metric_values),
                    "max": max(metric_values)
                }
        
        results["variants"].append(variant_summary)
    
    return results


@router.get("/experiments/{experiment_id}/participants", response_model=Dict[str, Any])
async def get_experiment_participants(
    experiment_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: str = Depends(get_current_user_email)
):
    """Get list of participants in an experiment."""
    
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = EXPERIMENTS[experiment_id]
    if experiment["created_by"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if experiment_id not in EXPERIMENT_METRICS:
        return {
            "experiment_id": experiment_id,
            "participants": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    
    assignments = EXPERIMENT_METRICS[experiment_id]["assignments"]
    total = len(assignments)
    
    # Paginate results
    paginated_assignments = assignments[offset:offset + limit]
    
    participants = []
    for assignment in paginated_assignments:
        participants.append({
            "user_id": assignment["user_id"],
            "variant_id": assignment["variant_id"],
            "assigned_at": assignment["assigned_at"].isoformat(),
            "session_id": assignment.get("session_id"),
            "user_attributes": assignment.get("user_attributes")
        })
    
    return {
        "experiment_id": experiment_id,
        "participants": participants,
        "total": total,
        "limit": limit,
        "offset": offset
    } 