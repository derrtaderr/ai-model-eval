"""
API endpoints for evaluation management.
Provides REST endpoints for human evaluations, model evaluations, and related operations.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.orm import selectinload

from database.connection import get_db
from database.models import Trace, Evaluation, User, TraceTag
from auth.security import get_current_user_email

# Import evaluation-related services
try:
    from services.evaluator_models import evaluator_manager, EvaluationRequest, EvaluationCriteria
    EVALUATOR_MODELS_AVAILABLE = True
except ImportError:
    evaluator_manager = None
    EVALUATOR_MODELS_AVAILABLE = False

try:
    from services.batch_evaluation import batch_processor
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    batch_processor = None
    BATCH_PROCESSING_AVAILABLE = False

# User Analytics service
try:
    from services.user_analytics import user_analytics, UserActionType, EngagementLevel
    USER_ANALYTICS_AVAILABLE = True
except ImportError:
    user_analytics = None
    USER_ANALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()


# Enhanced filter schemas for advanced filtering
class DateRangeFilter(BaseModel):
    """Schema for date range filtering."""
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")


class NumericRangeFilter(BaseModel):
    """Schema for numeric range filtering."""
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")


class TagFilter(BaseModel):
    """Schema for tag-based filtering."""
    tag_type: str = Field(..., description="Type of tag (tool, scenario, topic, etc.)")
    tag_values: List[str] = Field(..., description="List of tag values to filter by")
    match_all: bool = Field(False, description="If true, trace must have ALL tag values, else ANY")


class FilterOperator(str):
    """Filter combination operators."""
    AND = "AND"
    OR = "OR"


class FilterGroup(BaseModel):
    """Schema for grouped filters with logical operations."""
    model_config = ConfigDict(protected_namespaces=())
    
    operator: str = Field("AND", description="Logic operator for this group (AND/OR)")
    filters: List[Dict[str, Any]] = Field(..., description="Individual filters in this group")
    nested_groups: Optional[List['FilterGroup']] = Field(None, description="Nested filter groups")
    
    @validator('operator')
    def validate_operator(cls, v):
        if v not in ["AND", "OR"]:
            raise ValueError("operator must be either 'AND' or 'OR'")
        return v


class AdvancedFilterCombination(BaseModel):
    """Schema for complex filter combinations with nested logic."""
    model_config = ConfigDict(protected_namespaces=())
    
    root_operator: str = Field("AND", description="Root level logic operator")
    filter_groups: List[FilterGroup] = Field(..., description="Top-level filter groups")
    global_settings: Optional[Dict[str, Any]] = Field(None, description="Global filter settings (sort, limit, etc.)")
    
    @validator('root_operator')
    def validate_root_operator(cls, v):
        if v not in ["AND", "OR"]:
            raise ValueError("root_operator must be either 'AND' or 'OR'")
        return v


class AdvancedFilterRequest(BaseModel):
    """Schema for advanced multi-dimensional filtering."""
    model_config = ConfigDict(protected_namespaces=())
    
    # Basic filters
    model_names: Optional[List[str]] = Field(None, description="Filter by model names")
    session_ids: Optional[List[str]] = Field(None, description="Filter by session IDs")
    user_ids: Optional[List[str]] = Field(None, description="Filter by user IDs")
    evaluation_statuses: Optional[List[str]] = Field(None, description="Filter by evaluation status")
    trace_statuses: Optional[List[str]] = Field(None, description="Filter by trace status")
    
    # Date range filters
    trace_date_range: Optional[DateRangeFilter] = Field(None, description="Filter by trace timestamp")
    evaluation_date_range: Optional[DateRangeFilter] = Field(None, description="Filter by evaluation date")
    
    # Numeric range filters
    latency_range: Optional[NumericRangeFilter] = Field(None, description="Filter by latency in ms")
    cost_range: Optional[NumericRangeFilter] = Field(None, description="Filter by cost in USD")
    score_range: Optional[NumericRangeFilter] = Field(None, description="Filter by evaluation score")
    
    # Tag-based filters
    tag_filters: Optional[List[TagFilter]] = Field(None, description="Filter by tags")
    
    # Text search
    search_query: Optional[str] = Field(None, description="Search in user input and model output")
    search_in_fields: Optional[List[str]] = Field(
        ["user_input", "model_output"], 
        description="Fields to search in"
    )
    
    # Combination logic
    filter_operator: str = Field("AND", description="How to combine filters (AND/OR)")
    
    # Sorting and pagination
    sort_by: str = Field("timestamp", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    limit: int = Field(50, ge=1, le=500, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    
    @validator('filter_operator')
    def validate_operator(cls, v):
        if v not in ["AND", "OR"]:
            raise ValueError("filter_operator must be either 'AND' or 'OR'")
        return v
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError("sort_order must be either 'asc' or 'desc'")
        return v


class FilterCondition(BaseModel):
    """Schema for individual filter conditions."""
    model_config = ConfigDict(protected_namespaces=())
    
    field: str = Field(..., description="Database field to filter on")
    operator: str = Field(..., description="Comparison operator (eq, ne, in, not_in, gt, gte, lt, lte, like, ilike)")
    value: Union[str, int, float, List[Any], None] = Field(..., description="Filter value(s)")
    table: Optional[str] = Field(None, description="Database table if joining")
    
    @validator('operator')
    def validate_operator(cls, v):
        valid_ops = ["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "like", "ilike", "is_null", "is_not_null"]
        if v not in valid_ops:
            raise ValueError(f"operator must be one of: {valid_ops}")
        return v


class QueryBuilder:
    """Advanced query builder for complex filter combinations."""
    
    def __init__(self):
        self.field_mappings = {
            # Trace fields
            "model_name": ("traces", "model_name"),
            "session_id": ("traces", "session_id"),
            "user_id": ("traces", "user_id"),
            "status": ("traces", "status"),
            "timestamp": ("traces", "timestamp"),
            "latency_ms": ("traces", "latency_ms"),
            "cost_usd": ("traces", "cost_usd"),
            "user_input": ("traces", "user_input"),
            "model_output": ("traces", "model_output"),
            "system_prompt": ("traces", "system_prompt"),
            
            # Evaluation fields
            "evaluation_status": ("evaluations", "label"),
            "evaluation_score": ("evaluations", "score"),
            "evaluated_at": ("evaluations", "evaluated_at"),
            "evaluator_type": ("evaluations", "evaluator_type"),
            
            # Tag fields
            "tag_type": ("trace_tags", "tag_type"),
            "tag_value": ("trace_tags", "tag_value"),
            "tag_confidence": ("trace_tags", "confidence_score")
        }
    
    def build_condition(self, condition: FilterCondition):
        """Build a SQLAlchemy condition from a FilterCondition."""
        table_name, field_name = self.field_mappings.get(condition.field, ("traces", condition.field))
        
        if table_name == "traces":
            column = getattr(Trace, field_name)
        elif table_name == "evaluations":
            column = getattr(Evaluation, field_name)
        elif table_name == "trace_tags":
            column = getattr(TraceTag, field_name)
        else:
            raise ValueError(f"Unknown table: {table_name}")
        
        # Build the condition based on operator
        if condition.operator == "eq":
            return column == condition.value
        elif condition.operator == "ne":
            return column != condition.value
        elif condition.operator == "in":
            return column.in_(condition.value)
        elif condition.operator == "not_in":
            return ~column.in_(condition.value)
        elif condition.operator == "gt":
            return column > condition.value
        elif condition.operator == "gte":
            return column >= condition.value
        elif condition.operator == "lt":
            return column < condition.value
        elif condition.operator == "lte":
            return column <= condition.value
        elif condition.operator == "like":
            return column.like(f"%{condition.value}%")
        elif condition.operator == "ilike":
            return column.ilike(f"%{condition.value}%")
        elif condition.operator == "is_null":
            return column.is_(None)
        elif condition.operator == "is_not_null":
            return column.is_not(None)
        else:
            raise ValueError(f"Unsupported operator: {condition.operator}")
    
    def build_group_conditions(self, filter_group: FilterGroup):
        """Build conditions for a filter group."""
        conditions = []
        
        # Process individual filters
        for filter_dict in filter_group.filters:
            try:
                condition = FilterCondition(**filter_dict)
                sql_condition = self.build_condition(condition)
                conditions.append(sql_condition)
            except Exception as e:
                print(f"Warning: Skipping invalid filter condition: {e}")
                continue
        
        # Process nested groups
        if filter_group.nested_groups:
            for nested_group in filter_group.nested_groups:
                nested_condition = self.build_group_conditions(nested_group)
                if nested_condition is not None:
                    conditions.append(nested_condition)
        
        # Combine conditions with the group's operator
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        elif filter_group.operator == "AND":
            return and_(*conditions)
        else:  # OR
            return or_(*conditions)
    
    def build_query_from_combination(self, combination: "AdvancedFilterCombination"):
        """Build a SQLAlchemy query from an AdvancedFilterCombination."""
        all_conditions = []
        
        # Process each filter group
        for group in combination.filter_groups:
            group_condition = self.build_group_conditions(group)
            if group_condition is not None:
                all_conditions.append(group_condition)
        
        # Combine all groups with root operator
        if not all_conditions:
            return None
        elif len(all_conditions) == 1:
            return all_conditions[0]
        elif combination.root_operator == "AND":
            return and_(*all_conditions)
        else:  # OR
            return or_(*all_conditions)


# Update the existing AdvancedFilterRequest to include filter combinations
class EnhancedAdvancedFilterRequest(AdvancedFilterRequest):
    """Enhanced version of AdvancedFilterRequest with complex combinations."""
    model_config = ConfigDict(protected_namespaces=())
    
    # Add support for advanced combinations
    filter_combinations: Optional["AdvancedFilterCombination"] = Field(
        None, 
        description="Advanced filter combinations with nested logic"
    )
    use_combinations: bool = Field(
        False, 
        description="Whether to use filter_combinations instead of basic filters"
    )


class FilterPreset(BaseModel):
    """Schema for saved filter presets."""
    id: Optional[str] = Field(None, description="Preset ID (for updates)")
    name: str = Field(..., description="Name of the preset")
    description: Optional[str] = Field(None, description="Description of the preset")
    filters: AdvancedFilterRequest = Field(..., description="Filter configuration")
    is_public: bool = Field(False, description="Whether this preset is public")
    created_by: Optional[str] = Field(None, description="Creator email")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class TaxonomyItem(BaseModel):
    """Schema for taxonomy items."""
    tag_type: str
    tag_value: str
    count: int
    confidence_score: Optional[float] = None


class TaxonomyResponse(BaseModel):
    """Schema for taxonomy response."""
    tools: List[TaxonomyItem]
    scenarios: List[TaxonomyItem]
    topics: List[TaxonomyItem]
    custom_tags: Dict[str, List[TaxonomyItem]]
    total_traces: int
    last_updated: datetime


# Original schemas remain the same
class EvaluationCreate(BaseModel):
    """Schema for creating a new human evaluation."""
    trace_id: str = Field(..., description="ID of the trace being evaluated")
    label: str = Field(..., description="Evaluation label: accepted, rejected, or in_review")
    score: Optional[float] = Field(None, ge=0, le=1, description="Numeric score between 0 and 1")
    critique: Optional[str] = Field(None, description="Detailed feedback and notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional evaluation metadata")


class EvaluationUpdate(BaseModel):
    """Schema for updating an existing evaluation."""
    label: Optional[str] = Field(None, description="Updated evaluation label")
    score: Optional[float] = Field(None, ge=0, le=1, description="Updated numeric score")
    critique: Optional[str] = Field(None, description="Updated feedback and notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class EvaluationResponse(BaseModel):
    """Schema for evaluation response."""
    id: str
    trace_id: str
    evaluator_type: str
    evaluator_id: Optional[str]
    evaluator_email: Optional[str]
    score: Optional[float]
    label: Optional[str]
    critique: Optional[str]
    metadata: Optional[Dict[str, Any]]
    evaluated_at: str


class TraceWithEvaluations(BaseModel):
    """Schema for trace with its evaluations."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    timestamp: str
    user_input: str
    model_output: str
    model_name: str
    system_prompt: Optional[str]
    session_id: Optional[str]
    trace_metadata: Optional[Dict[str, Any]]
    latency_ms: Optional[int]
    token_count: Optional[Dict[str, int]]
    cost_usd: Optional[float]
    status: str
    evaluations: List[EvaluationResponse]
    human_evaluation_status: str  # "pending", "accepted", "rejected", "in_review"
    tags: Optional[List[Dict[str, Any]]] = Field(None, description="Trace tags")


class EvaluationStats(BaseModel):
    """Schema for evaluation statistics."""
    total_traces: int
    evaluated_traces: int
    pending_traces: int
    accepted_traces: int
    rejected_traces: int
    in_review_traces: int
    evaluation_rate: float
    acceptance_rate: float
    agreement_data: List[Dict[str, Any]]


class FilteredTraceResponse(BaseModel):
    """Enhanced response for filtered traces."""
    traces: List[TraceWithEvaluations]
    total_count: int
    filtered_count: int
    pagination: Dict[str, Any]
    filter_summary: Dict[str, Any]
    applied_filters: AdvancedFilterRequest


class FilterPresetCreate(BaseModel):
    """Schema for creating a filter preset."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Name of the filter preset")
    description: Optional[str] = Field(None, description="Description of the preset")
    filter_config: Dict[str, Any] = Field(..., description="Complete filter configuration")
    is_public: bool = Field(False, description="Whether preset can be shared with other users")
    is_default: bool = Field(False, description="Whether this should be the user's default preset")


class FilterPresetUpdate(BaseModel):
    """Schema for updating a filter preset."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: Optional[str] = Field(None, description="Updated name")
    description: Optional[str] = Field(None, description="Updated description")
    filter_config: Optional[Dict[str, Any]] = Field(None, description="Updated filter configuration")
    is_public: Optional[bool] = Field(None, description="Updated public status")
    is_default: Optional[bool] = Field(None, description="Updated default status")


class FilterPresetResponse(BaseModel):
    """Schema for filter preset response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    name: str
    description: Optional[str]
    filter_config: Dict[str, Any]
    is_public: bool
    is_default: bool
    usage_count: int
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime]
    user_email: str


class FilterPresetsListResponse(BaseModel):
    """Schema for filter presets list response."""
    model_config = ConfigDict(protected_namespaces=())
    
    presets: List[FilterPresetResponse]
    total_count: int
    user_presets_count: int
    public_presets_count: int


class FilterShareRequest(BaseModel):
    """Schema for sharing filter configurations via URL."""
    model_config = ConfigDict(protected_namespaces=())
    
    filter_config: Dict[str, Any] = Field(..., description="Filter configuration to share")
    name: Optional[str] = Field(None, description="Optional name for the shared filter")
    description: Optional[str] = Field(None, description="Optional description")
    expires_in_hours: int = Field(24, ge=1, le=168, description="Hours until the share link expires")


class FilterShareResponse(BaseModel):
    """Schema for filter sharing response."""
    model_config = ConfigDict(protected_namespaces=())
    
    share_token: str = Field(..., description="Encoded token for URL sharing")
    share_url: str = Field(..., description="Complete shareable URL")
    expires_at: datetime = Field(..., description="When the share link expires")
    filter_summary: Dict[str, Any] = Field(..., description="Summary of shared filters")


class SharedFilterInfo(BaseModel):
    """Schema for shared filter information."""
    model_config = ConfigDict(protected_namespaces=())
    
    filter_config: Dict[str, Any] = Field(..., description="Decoded filter configuration")
    name: Optional[str] = Field(None, description="Name of the shared filter")
    description: Optional[str] = Field(None, description="Description of the shared filter")
    created_at: datetime = Field(..., description="When the share was created")
    expires_at: datetime = Field(..., description="When the share expires")
    is_expired: bool = Field(..., description="Whether the share has expired")


def encode_filter_config(filter_config: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
    """
    Encode filter configuration into a URL-safe token.
    
    Uses compression and base64 encoding for compact URLs.
    """
    try:
        # Prepare data for encoding
        data = {
            "filter_config": filter_config,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        # Convert to JSON and compress
        json_data = json.dumps(data, separators=(',', ':'))
        compressed_data = zlib.compress(json_data.encode('utf-8'))
        
        # Base64 encode for URL safety
        encoded = base64.urlsafe_b64encode(compressed_data).decode('ascii')
        
        return encoded
        
    except Exception as e:
        raise ValueError(f"Failed to encode filter configuration: {str(e)}")


def decode_filter_config(encoded_token: str) -> Dict[str, Any]:
    """
    Decode filter configuration from a URL token.
    
    Handles decompression and JSON parsing with error handling.
    """
    try:
        # Decode from base64
        compressed_data = base64.urlsafe_b64decode(encoded_token.encode('ascii'))
        
        # Decompress
        json_data = zlib.decompress(compressed_data).decode('utf-8')
        
        # Parse JSON
        data = json.loads(json_data)
        
        return data
        
    except Exception as e:
        raise ValueError(f"Failed to decode filter token: {str(e)}")


def generate_share_url(base_url: str, share_token: str) -> str:
    """Generate a complete shareable URL with the encoded filter token."""
    encoded_token = quote(share_token, safe='')
    return f"{base_url}?shared_filter={encoded_token}"


def extract_filter_summary(filter_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a human-readable summary of applied filters."""
    summary = {
        "total_filters": 0,
        "filter_types": [],
        "active_filters": {}
    }
    
    # Count different types of filters
    if filter_config.get("model_names"):
        summary["total_filters"] += 1
        summary["filter_types"].append("model_names")
        summary["active_filters"]["models"] = len(filter_config["model_names"])
    
    if filter_config.get("trace_date_range"):
        summary["total_filters"] += 1
        summary["filter_types"].append("date_range")
        summary["active_filters"]["date_range"] = True
    
    if filter_config.get("tag_filters"):
        summary["total_filters"] += 1
        summary["filter_types"].append("tags")
        summary["active_filters"]["tags"] = len(filter_config["tag_filters"])
    
    if filter_config.get("search_query"):
        summary["total_filters"] += 1
        summary["filter_types"].append("text_search")
        summary["active_filters"]["search"] = True
    
    if filter_config.get("evaluation_statuses"):
        summary["total_filters"] += 1
        summary["filter_types"].append("evaluation_status")
        summary["active_filters"]["evaluation_statuses"] = len(filter_config["evaluation_statuses"])
    
    return summary


@router.post("/evaluations/traces/search", response_model=FilteredTraceResponse)
async def search_traces_advanced(
    filter_request: AdvancedFilterRequest,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Advanced search endpoint with multi-dimensional filtering and caching.
    
    Supports filtering by model, status, date ranges, tags, text search, and more.
    Implements intelligent caching for performance optimization.
    """
    try:
        # Check cache first for performance optimization
        cache_key_data = {
            "user": current_user_email,
            "filters": filter_request.dict(),
            "timestamp_limit": int(datetime.utcnow().timestamp() // 300)  # 5-minute cache buckets
        }
        
        cached_result = await get_cached_filter_results(cache_key_data)
        if cached_result is not None:
            # Return cached result with cache indicator
            return FilteredTraceResponse(
                traces=cached_result.get("traces", []),
                total_count=cached_result.get("total_count", 0),
                filtered_count=cached_result.get("filtered_count", 0),
                pagination=cached_result.get("pagination", {}),
                filter_summary=cached_result.get("filter_summary", {"cached": True}),
                applied_filters=filter_request
            )
        
        # Start with base query
        query = select(Trace).options(
            selectinload(Trace.evaluations),
            selectinload(Trace.trace_tags)
        )
        
        # Count query for total results
        count_query = select(func.count(Trace.id))
        
        # Build filter conditions
        filter_conditions = []
        
        # Basic filters
        if filter_request.model_names:
            filter_conditions.append(Trace.model_name.in_(filter_request.model_names))
            
        if filter_request.session_ids:
            filter_conditions.append(Trace.session_id.in_(filter_request.session_ids))
            
        if filter_request.user_ids:
            user_uuids = [UUID(uid) for uid in filter_request.user_ids]
            filter_conditions.append(Trace.user_id.in_(user_uuids))
            
        if filter_request.trace_statuses:
            filter_conditions.append(Trace.status.in_(filter_request.trace_statuses))
        
        # Date range filters
        if filter_request.trace_date_range:
            if filter_request.trace_date_range.start_date:
                filter_conditions.append(Trace.timestamp >= filter_request.trace_date_range.start_date)
            if filter_request.trace_date_range.end_date:
                filter_conditions.append(Trace.timestamp <= filter_request.trace_date_range.end_date)
        
        # Numeric range filters
        if filter_request.latency_range:
            if filter_request.latency_range.min_value is not None:
                filter_conditions.append(Trace.latency_ms >= filter_request.latency_range.min_value)
            if filter_request.latency_range.max_value is not None:
                filter_conditions.append(Trace.latency_ms <= filter_request.latency_range.max_value)
                
        if filter_request.cost_range:
            if filter_request.cost_range.min_value is not None:
                filter_conditions.append(Trace.cost_usd >= filter_request.cost_range.min_value)
            if filter_request.cost_range.max_value is not None:
                filter_conditions.append(Trace.cost_usd <= filter_request.cost_range.max_value)
        
        # Tag filters
        if filter_request.tag_filters:
            for tag_filter in filter_request.tag_filters:
                tag_subquery = select(TraceTag.trace_id).where(
                    and_(
                        TraceTag.tag_type == tag_filter.tag_type,
                        TraceTag.tag_value.in_(tag_filter.tag_values)
                    )
                )
                
                if tag_filter.match_all:
                    # Trace must have ALL specified tag values
                    for tag_value in tag_filter.tag_values:
                        tag_exists = select(TraceTag.trace_id).where(
                            and_(
                                TraceTag.tag_type == tag_filter.tag_type,
                                TraceTag.tag_value == tag_value
                            )
                        )
                        filter_conditions.append(Trace.id.in_(tag_exists))
                else:
                    # Trace must have ANY of the specified tag values
                    filter_conditions.append(Trace.id.in_(tag_subquery))
        
        # Text search
        if filter_request.search_query:
            search_conditions = []
            for field in filter_request.search_in_fields:
                if field == "user_input":
                    search_conditions.append(Trace.user_input.ilike(f"%{filter_request.search_query}%"))
                elif field == "model_output":
                    search_conditions.append(Trace.model_output.ilike(f"%{filter_request.search_query}%"))
                elif field == "system_prompt":
                    search_conditions.append(Trace.system_prompt.ilike(f"%{filter_request.search_query}%"))
            
            if search_conditions:
                filter_conditions.append(or_(*search_conditions))
        
        # Apply filters
        if filter_conditions:
            if filter_request.filter_operator == "AND":
                combined_filter = and_(*filter_conditions)
            else:
                combined_filter = or_(*filter_conditions)
            
            query = query.where(combined_filter)
            count_query = count_query.where(combined_filter)
        
        # Get total count
        count_result = await db.execute(count_query)
        total_filtered_count = count_result.scalar() or 0
        
        # Apply sorting
        sort_column = getattr(Trace, filter_request.sort_by, Trace.timestamp)
        if filter_request.sort_order == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        query = query.offset(filter_request.offset).limit(filter_request.limit)
        
        # Execute query
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Process traces and handle evaluation status filtering
        processed_traces = []
        for trace in traces:
            # Determine human evaluation status
            human_evaluations = [e for e in trace.evaluations if e.evaluator_type == "human"]
            
            if not human_evaluations:
                human_eval_status = "pending"
            else:
                latest_eval = max(human_evaluations, key=lambda e: e.evaluated_at)
                human_eval_status = latest_eval.label or "pending"
            
            # Apply evaluation status filter if specified
            if (filter_request.evaluation_statuses and 
                human_eval_status not in filter_request.evaluation_statuses):
                continue
            
            # Format trace data
            formatted_evaluations = []
            for eval in trace.evaluations:
                # Apply score range filter to evaluations
                if (filter_request.score_range and eval.score is not None):
                    if (filter_request.score_range.min_value is not None and 
                        eval.score < filter_request.score_range.min_value):
                        continue
                    if (filter_request.score_range.max_value is not None and 
                        eval.score > filter_request.score_range.max_value):
                        continue
                
                formatted_evaluations.append(EvaluationResponse(
                    id=str(eval.id),
                    trace_id=str(eval.trace_id),
                    evaluator_type=eval.evaluator_type,
                    evaluator_id=str(eval.evaluator_id) if eval.evaluator_id else None,
                    evaluator_email=None,
                    score=eval.score,
                    label=eval.label,
                    critique=eval.critique,
                    metadata=eval.eval_metadata,
                    evaluated_at=eval.evaluated_at.isoformat()
                ))
            
            # Format tags
            formatted_tags = []
            for tag in trace.trace_tags:
                formatted_tags.append({
                    "type": tag.tag_type,
                    "value": tag.tag_value,
                    "confidence": tag.confidence_score
                })
            
            processed_traces.append(TraceWithEvaluations(
                id=str(trace.id),
                timestamp=trace.timestamp.isoformat(),
                user_input=trace.user_input,
                model_output=trace.model_output,
                model_name=trace.model_name,
                system_prompt=trace.system_prompt,
                session_id=trace.session_id,
                trace_metadata=trace.trace_metadata,
                latency_ms=trace.latency_ms,
                token_count=trace.token_count,
                cost_usd=trace.cost_usd,
                status=trace.status,
                evaluations=formatted_evaluations,
                human_evaluation_status=human_eval_status,
                tags=formatted_tags
            ))
        
        # Calculate filter summary
        filter_summary = {
            "filters_applied": len([f for f in [
                filter_request.model_names,
                filter_request.session_ids,
                filter_request.user_ids,
                filter_request.evaluation_statuses,
                filter_request.trace_statuses,
                filter_request.trace_date_range,
                filter_request.evaluation_date_range,
                filter_request.latency_range,
                filter_request.cost_range,
                filter_request.score_range,
                filter_request.tag_filters,
                filter_request.search_query
            ] if f is not None]),
            "operator": filter_request.filter_operator
        }
        
        # Pagination info
        pagination = {
            "offset": filter_request.offset,
            "limit": filter_request.limit,
            "has_next": len(processed_traces) == filter_request.limit,
            "has_previous": filter_request.offset > 0
        }
        
        # Save to cache
        await cache_filter_results(cache_key_data, {
            "traces": processed_traces,
            "total_count": total_filtered_count,
            "filtered_count": len(processed_traces),
            "pagination": pagination,
            "filter_summary": filter_summary
        })
        
        return FilteredTraceResponse(
            traces=processed_traces,
            total_count=total_filtered_count,
            filtered_count=len(processed_traces),
            pagination=pagination,
            filter_summary=filter_summary,
            applied_filters=filter_request
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search traces: {str(e)}"
        )


@router.post("/filters/convert-to-advanced", response_model=AdvancedFilterCombination)
async def convert_basic_to_advanced_filters(
    basic_filters: AdvancedFilterRequest,
    current_user: str = Depends(get_current_user_email)
):
    """
    Convert basic filter request to advanced filter combination format.
    
    Useful for migrating from simple filters to complex combinations.
    """
    try:
        filter_conditions = []
        
        # Convert basic filters to filter conditions
        if basic_filters.model_names:
            filter_conditions.append({
                "field": "model_name",
                "operator": "in",
                "value": basic_filters.model_names
            })
        
        if basic_filters.session_ids:
            filter_conditions.append({
                "field": "session_id",
                "operator": "in",
                "value": basic_filters.session_ids
            })
        
        if basic_filters.user_ids:
            filter_conditions.append({
                "field": "user_id",
                "operator": "in",
                "value": basic_filters.user_ids
            })
        
        if basic_filters.trace_statuses:
            filter_conditions.append({
                "field": "status",
                "operator": "in",
                "value": basic_filters.trace_statuses
            })
        
        if basic_filters.search_query:
            # Create a group for text search across multiple fields
            search_conditions = []
            for field in basic_filters.search_in_fields:
                search_conditions.append({
                    "field": field,
                    "operator": "ilike",
                    "value": basic_filters.search_query
                })
            
            # Add as a separate group with OR logic
            if search_conditions:
                filter_conditions.extend(search_conditions)
        
        # Date range filters
        if basic_filters.trace_date_range:
            if basic_filters.trace_date_range.start_date:
                filter_conditions.append({
                    "field": "timestamp",
                    "operator": "gte",
                    "value": basic_filters.trace_date_range.start_date.isoformat()
                })
            if basic_filters.trace_date_range.end_date:
                filter_conditions.append({
                    "field": "timestamp",
                    "operator": "lte",
                    "value": basic_filters.trace_date_range.end_date.isoformat()
                })
        
        # Numeric range filters
        if basic_filters.latency_range:
            if basic_filters.latency_range.min_value is not None:
                filter_conditions.append({
                    "field": "latency_ms",
                    "operator": "gte",
                    "value": basic_filters.latency_range.min_value
                })
            if basic_filters.latency_range.max_value is not None:
                filter_conditions.append({
                    "field": "latency_ms",
                    "operator": "lte",
                    "value": basic_filters.latency_range.max_value
                })
        
        if basic_filters.cost_range:
            if basic_filters.cost_range.min_value is not None:
                filter_conditions.append({
                    "field": "cost_usd",
                    "operator": "gte",
                    "value": basic_filters.cost_range.min_value
                })
            if basic_filters.cost_range.max_value is not None:
                filter_conditions.append({
                    "field": "cost_usd",
                    "operator": "lte",
                    "value": basic_filters.cost_range.max_value
                })
        
        # Create a single filter group with all conditions
        main_group = FilterGroup(
            operator=basic_filters.filter_operator,
            filters=filter_conditions,
            nested_groups=None
        )
        
        # Create global settings
        global_settings = {
            "sort_by": basic_filters.sort_by,
            "sort_order": basic_filters.sort_order,
            "limit": basic_filters.limit,
            "offset": basic_filters.offset
        }
        
        # Create the advanced combination
        advanced_combination = AdvancedFilterCombination(
            root_operator="AND",
            filter_groups=[main_group],
            global_settings=global_settings
        )
        
        return advanced_combination
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert filters: {str(e)}"
        )


@router.post("/filters/validate-combination", response_model=Dict[str, Any])
async def validate_filter_combination(
    combination: AdvancedFilterCombination,
    current_user: str = Depends(get_current_user_email)
):
    """
    Validate a filter combination without executing it.
    
    Returns validation results and complexity analysis.
    """
    try:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "complexity": _calculate_filter_complexity(combination),
            "estimated_performance": "good"
        }
        
        # Validate filter groups
        for i, group in enumerate(combination.filter_groups):
            group_validation = _validate_filter_group(group, f"group_{i}")
            validation_results["errors"].extend(group_validation["errors"])
            validation_results["warnings"].extend(group_validation["warnings"])
        
        # Check complexity
        if validation_results["complexity"] > 10:
            validation_results["warnings"].append("High complexity filter may impact performance")
            validation_results["estimated_performance"] = "slow"
        elif validation_results["complexity"] > 5:
            validation_results["estimated_performance"] = "moderate"
        
        # Set overall validity
        validation_results["is_valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate filter combination: {str(e)}"
        )


def _calculate_filter_complexity(combination: AdvancedFilterCombination) -> int:
    """Calculate complexity score for a filter combination."""
    complexity = 0
    
    for group in combination.filter_groups:
        complexity += _calculate_group_complexity(group)
    
    return complexity


def _calculate_group_complexity(group: FilterGroup) -> int:
    """Calculate complexity score for a filter group."""
    complexity = len(group.filters)
    
    if group.nested_groups:
        for nested_group in group.nested_groups:
            complexity += _calculate_group_complexity(nested_group) + 1  # +1 for nesting
    
    return complexity


def _validate_filter_group(group: FilterGroup, group_path: str) -> Dict[str, List[str]]:
    """Validate a filter group structure."""
    validation = {"errors": [], "warnings": []}
    
    # Check if group has any filters or nested groups
    if not group.filters and not group.nested_groups:
        validation["errors"].append(f"{group_path}: Filter group must have either filters or nested groups")
    
    # Validate individual filters
    for i, filter_dict in enumerate(group.filters):
        try:
            condition = FilterCondition(**filter_dict)
            # Additional validation could be added here
        except Exception as e:
            validation["errors"].append(f"{group_path}.filter_{i}: Invalid filter condition: {str(e)}")
    
    # Validate nested groups recursively
    if group.nested_groups:
        for i, nested_group in enumerate(group.nested_groups):
            nested_validation = _validate_filter_group(nested_group, f"{group_path}.nested_{i}")
            validation["errors"].extend(nested_validation["errors"])
            validation["warnings"].extend(nested_validation["warnings"])
    
    return validation


# Performance Optimization Endpoints

@router.get("/cache/stats")
async def get_cache_statistics(
    current_user: str = Depends(get_current_user_email)
):
    """
    Get cache performance statistics and configuration.
    """
    try:
        cache_manager = await get_cache_manager()
        stats = await cache_manager.get_cache_stats()
        
        return {
            "cache_statistics": stats,
            "performance_info": {
                "cache_hit_optimization": "5-minute buckets for filter results",
                "taxonomy_cache_ttl": "24 hours",
                "query_cache_ttl": "15 minutes",
                "filter_cache_ttl": "30 minutes"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.post("/cache/invalidate")
async def invalidate_caches(
    cache_type: Optional[str] = Query(None, description="Specific cache type to invalidate (filter, taxonomy, all)"),
    current_user: str = Depends(get_current_user_email)
):
    """
    Invalidate caches for performance optimization.
    Useful after bulk data updates or configuration changes.
    """
    try:
        cache_manager = await get_cache_manager()
        
        if cache_type == "filter":
            await cache_manager.invalidate_filter_cache()
            message = "Filter cache invalidated successfully"
        elif cache_type == "taxonomy":
            await cache_manager.invalidate_taxonomy_cache()
            message = "Taxonomy cache invalidated successfully"
        elif cache_type == "all" or cache_type is None:
            await cache_manager.clear()
            message = "All caches invalidated successfully"
        else:
            raise HTTPException(status_code=400, detail="Invalid cache type. Use 'filter', 'taxonomy', or 'all'")
        
        return {
            "message": message,
            "cache_type": cache_type or "all",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache: {str(e)}"
        )


@router.get("/performance/query-analysis")
async def analyze_query_performance(
    sample_size: int = Query(100, ge=10, le=1000, description="Number of recent queries to analyze"),
    current_user: str = Depends(get_current_user_email),
    session: AsyncSession = Depends(get_db)
):
    """
    Analyze query performance and provide optimization recommendations.
    """
    try:
        # Analyze recent trace queries (simplified analysis)
        recent_traces_query = (
            select(func.count(Trace.id).label('total_traces'))
            .select_from(Trace)
        )
        result = await session.execute(recent_traces_query)
        total_traces = result.scalar() or 0
        
        # Analyze index usage (simulated for now)
        index_analysis = {
            "timestamp_index_usage": "High - Used in 95% of queries",
            "model_name_index_usage": "Medium - Used in 60% of queries",
            "session_id_index_usage": "Low - Used in 30% of queries",
            "user_id_index_usage": "Low - Used in 25% of queries"
        }
        
        # Performance recommendations
        recommendations = []
        
        if total_traces > 10000:
            recommendations.append({
                "type": "database",
                "priority": "high",
                "recommendation": "Consider implementing database partitioning for trace data",
                "reason": f"Large dataset detected ({total_traces} traces)"
            })
        
        if total_traces > 1000:
            recommendations.append({
                "type": "caching",
                "priority": "medium",
                "recommendation": "Enable Redis caching for distributed environments",
                "reason": "Medium dataset size benefits from distributed caching"
            })
        
        recommendations.append({
            "type": "query",
            "priority": "low",
            "recommendation": "Use specific field filters instead of full-text search when possible",
            "reason": "Full-text search has higher performance cost"
        })
        
        return {
            "performance_analysis": {
                "total_traces": total_traces,
                "dataset_size_category": "large" if total_traces > 10000 else "medium" if total_traces > 1000 else "small",
                "estimated_query_time": "<100ms" if total_traces < 1000 else "<500ms" if total_traces < 10000 else "<2s"
            },
            "index_analysis": index_analysis,
            "recommendations": recommendations,
            "optimization_tips": [
                "Use pagination with reasonable page sizes (≤100 items)",
                "Apply specific filters before generic text search",
                "Cache frequently used filter combinations",
                "Use date range filters to limit dataset scope"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze query performance: {str(e)}"
        )


@router.post("/performance/optimize-database")
async def optimize_database_performance(
    operation: str = Query(..., description="Optimization operation: 'analyze_tables', 'update_statistics', 'reindex'"),
    current_user: str = Depends(get_current_user_email),
    session: AsyncSession = Depends(get_db)
):
    """
    Perform database optimization operations.
    Note: Some operations may require database admin privileges.
    """
    try:
        optimization_results = {}
        
        if operation == "analyze_tables":
            # Simulate table analysis (actual implementation would vary by database)
            optimization_results = {
                "operation": "analyze_tables",
                "status": "completed",
                "tables_analyzed": ["traces", "evaluations", "trace_tags", "filter_presets"],
                "recommendations": [
                    "traces table: Consider archiving data older than 6 months",
                    "evaluations table: Index performance is optimal",
                    "trace_tags table: Consider composite index on (trace_id, tag_type)",
                    "filter_presets table: Low usage, no optimization needed"
                ]
            }
        
        elif operation == "update_statistics":
            # Simulate statistics update
            optimization_results = {
                "operation": "update_statistics",
                "status": "completed",
                "statistics_updated": ["traces", "evaluations", "trace_tags"],
                "query_planner_improvements": "Updated statistics should improve query planning for large datasets"
            }
        
        elif operation == "reindex":
            # Simulate reindexing
            optimization_results = {
                "operation": "reindex",
                "status": "completed",
                "indexes_rebuilt": [
                    "idx_traces_timestamp",
                    "idx_traces_model_name",
                    "idx_traces_session_timestamp",
                    "idx_evaluations_trace_type"
                ],
                "performance_improvement": "5-15% query performance improvement expected"
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid operation. Use 'analyze_tables', 'update_statistics', or 'reindex'"
            )
        
        return {
            "optimization_results": optimization_results,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "This is a simulation. Actual database optimization requires admin privileges."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize database: {str(e)}"
        )


# Model-based evaluation schemas
class ModelEvaluationRequestSchema(BaseModel):
    """Schema for requesting model-based evaluation."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_id: str = Field(..., description="ID of the trace to evaluate")
    user_input: str = Field(..., description="Original user input")
    model_output: str = Field(..., description="AI model's response to evaluate")
    system_prompt: Optional[str] = Field(None, description="System prompt used")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    criteria: List[str] = Field(default=["coherence", "relevance"], description="Evaluation criteria")
    custom_prompt: Optional[str] = Field(None, description="Custom evaluation prompt")
    reference_answer: Optional[str] = Field(None, description="Reference answer for comparison")
    evaluator_model: Optional[str] = Field(None, description="Specific evaluator model to use")
    use_calibration: bool = Field(default=True, description="Whether to apply score calibration")


class ModelEvaluationResponse(BaseModel):
    """Schema for model-based evaluation response."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_id: str
    evaluator_model: str
    criteria: str
    score: float
    reasoning: str
    confidence: float
    evaluation_time_ms: int
    cost_usd: Optional[float]
    metadata: Dict[str, Any]


class BatchEvaluationRequest(BaseModel):
    """Schema for batch evaluation request."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_ids: List[str] = Field(..., description="List of trace IDs to evaluate")
    criteria: List[str] = Field(..., description="Evaluation criteria to assess")
    evaluator_model: Optional[str] = Field(None, description="Specific evaluator model to use")
    custom_prompt: Optional[str] = Field(None, description="Custom evaluation prompt")
    parallel_workers: int = Field(5, ge=1, le=20, description="Number of parallel evaluation workers")


class BatchEvaluationResponse(BaseModel):
    """Schema for batch evaluation response."""
    model_config = ConfigDict(protected_namespaces=())
    
    total_traces: int
    successful_evaluations: int
    failed_evaluations: int
    results: List[ModelEvaluationResponse]
    errors: List[Dict[str, Any]]
    total_time_ms: int
    total_cost_usd: float


class AvailableEvaluatorsResponse(BaseModel):
    """Schema for available evaluators response."""
    model_config = ConfigDict(protected_namespaces=())
    
    available_evaluators: List[str]
    cost_estimates: Dict[str, Dict[str, float]]
    evaluation_criteria: List[str]


# Model-based evaluation endpoints

@router.get("/model-evaluation/evaluators", response_model=AvailableEvaluatorsResponse)
async def get_available_evaluators(
    current_user: str = Depends(get_current_user_email)
):
    """
    Get list of available model evaluators and their capabilities.
    """
    try:
        available_evaluators = await evaluator_manager.get_available_evaluators()
        cost_estimates = await evaluator_manager.get_cost_estimates()
        
        # Get available evaluation criteria
        criteria_list = [criteria.value for criteria in EvaluationCriteria]
        
        return AvailableEvaluatorsResponse(
            available_evaluators=available_evaluators,
            cost_estimates=cost_estimates,
            evaluation_criteria=criteria_list
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available evaluators: {str(e)}"
        )


@router.post("/model-evaluation/evaluate", response_model=List[ModelEvaluationResponse])
async def evaluate_trace_with_model(
    request: ModelEvaluationRequestSchema,
    current_user: str = Depends(get_current_user_email),
    session: AsyncSession = Depends(get_db)
):
    """
    Evaluate a single trace using model-based evaluation.
    Supports multiple criteria and evaluator selection.
    """
    try:
        # Get the trace from database
        trace_query = select(Trace).where(Trace.id == request.trace_id)
        result = await session.execute(trace_query)
        trace = result.scalar_one_or_none()
        
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")
        
        # Convert criteria strings to EvaluationCriteria enums
        criteria_enums = []
        for criteria_str in request.criteria:
            try:
                criteria_enums.append(EvaluationCriteria(criteria_str))
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid criteria: {criteria_str}. Available: {[c.value for c in EvaluationCriteria]}"
                )
        
        # Create evaluation request
        eval_request = ModelEvaluationRequest(
            trace_id=request.trace_id,
            user_input=request.user_input,
            model_output=request.model_output,
            system_prompt=request.system_prompt,
            context=request.context,
            criteria=criteria_enums,
            custom_prompt=request.custom_prompt,
            reference_answer=request.reference_answer
        )
        
        # Perform evaluation for each criteria
        evaluation_results = await evaluator_manager.evaluate_multiple_criteria(
            eval_request,
            criteria_enums,
            request.evaluator_model
        )
        
        # Convert results to response format
        response_results = []
        for result in evaluation_results:
            response_results.append(ModelEvaluationResponse(
                trace_id=result.trace_id,
                evaluator_model=result.evaluator_model,
                criteria=result.criteria.value,
                score=result.score,
                reasoning=result.reasoning,
                confidence=result.confidence,
                evaluation_time_ms=result.evaluation_time_ms or 0,
                cost_usd=result.cost_usd,
                metadata=result.metadata
            ))
        
        # Store evaluation results in database
        for result in evaluation_results:
            evaluation = Evaluation(
                trace_id=trace.id,
                evaluator_type="model",
                score=result.score,
                label=f"model_score_{result.score:.2f}",
                critique=result.reasoning,
                eval_metadata={
                    "evaluator_model": result.evaluator_model,
                    "criteria": result.criteria.value,
                    "confidence": result.confidence,
                    "evaluation_time_ms": result.evaluation_time_ms,
                    "cost_usd": result.cost_usd,
                    **result.metadata
                }
            )
            session.add(evaluation)
        
        await session.commit()
        
        return response_results
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate trace: {str(e)}"
        )


@router.post("/model-evaluation/batch", response_model=BatchEvaluationResponse)
async def batch_evaluate_traces(
    request: BatchEvaluationRequest,
    current_user: str = Depends(get_current_user_email),
    session: AsyncSession = Depends(get_db)
):
    """
    Evaluate multiple traces in batch using model-based evaluation.
    Supports parallel processing for improved performance.
    """
    try:
        start_time = time.time()
        
        # Validate trace IDs exist
        traces_query = select(Trace).where(Trace.id.in_(request.trace_ids))
        result = await session.execute(traces_query)
        traces = result.scalars().all()
        
        if len(traces) != len(request.trace_ids):
            found_ids = {str(trace.id) for trace in traces}
            missing_ids = set(request.trace_ids) - found_ids
            raise HTTPException(
                status_code=404, 
                detail=f"Traces not found: {list(missing_ids)}"
            )
        
        # Convert criteria strings to enums
        criteria_enums = []
        for criteria_str in request.criteria:
            try:
                criteria_enums.append(EvaluationCriteria(criteria_str))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid criteria: {criteria_str}"
                )
        
        # Create evaluation requests
        eval_requests = []
        for trace in traces:
            eval_request = ModelEvaluationRequest(
                trace_id=str(trace.id),
                user_input=trace.user_input,
                model_output=trace.model_output,
                system_prompt=trace.system_prompt,
                context=trace.context,
                criteria=criteria_enums,
                custom_prompt=request.custom_prompt
            )
            eval_requests.append(eval_request)
        
        # Perform batch evaluation with parallel processing
        all_results = []
        errors = []
        
        async def evaluate_trace_batch(eval_request):
            try:
                results = await evaluator_manager.evaluate_multiple_criteria(
                    eval_request,
                    criteria_enums,
                    request.evaluator_model
                )
                return results
            except Exception as e:
                errors.append({
                    "trace_id": eval_request.trace_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                return []
        
        # Process in batches to avoid overwhelming the API
        batch_size = request.parallel_workers
        for i in range(0, len(eval_requests), batch_size):
            batch = eval_requests[i:i + batch_size]
            tasks = [evaluate_trace_batch(req) for req in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for results in batch_results:
                if isinstance(results, list):
                    all_results.extend(results)
        
        # Convert results to response format
        response_results = []
        total_cost = 0.0
        
        for result in all_results:
            response_results.append(ModelEvaluationResponse(
                trace_id=result.trace_id,
                evaluator_model=result.evaluator_model,
                criteria=result.criteria.value,
                score=result.score,
                reasoning=result.reasoning,
                confidence=result.confidence,
                evaluation_time_ms=result.evaluation_time_ms or 0,
                cost_usd=result.cost_usd,
                metadata=result.metadata
            ))
            
            if result.cost_usd:
                total_cost += result.cost_usd
        
        # Store evaluation results in database
        for result in all_results:
            evaluation = Evaluation(
                trace_id=result.trace_id,
                evaluator_type="model",
                score=result.score,
                label=f"model_score_{result.score:.2f}",
                critique=result.reasoning,
                eval_metadata={
                    "evaluator_model": result.evaluator_model,
                    "criteria": result.criteria.value,
                    "confidence": result.confidence,
                    "evaluation_time_ms": result.evaluation_time_ms,
                    "cost_usd": result.cost_usd,
                    **result.metadata
                }
            )
            session.add(evaluation)
        
        await session.commit()
        
        total_time = int((time.time() - start_time) * 1000)
        
        return BatchEvaluationResponse(
            total_traces=len(request.trace_ids),
            successful_evaluations=len(all_results),
            failed_evaluations=len(errors),
            results=response_results,
            errors=errors,
            total_time_ms=total_time,
            total_cost_usd=total_cost
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform batch evaluation: {str(e)}"
        )


# Evaluation template management endpoints

@router.get("/model-evaluation/templates")
async def list_evaluation_templates(
    category: Optional[str] = Query(None, description="Filter by template category"),
    criteria: Optional[str] = Query(None, description="Filter by evaluation criteria"),
    current_user: str = Depends(get_current_user_email)
):
    """
    List available evaluation templates.
    Supports filtering by category and criteria.
    """
    try:
        from services.evaluation_templates import template_library, TemplateCategory
        
        templates = template_library.list_all_templates()
        
        # Apply filters
        if criteria:
            try:
                criteria_enum = EvaluationCriteria(criteria)
                templates = [t for t in templates if t.criteria == criteria_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid criteria: {criteria}")
        
        if category:
            try:
                category_enum = TemplateCategory(category)
                category_templates = template_library.get_templates_by_category(category_enum)
                template_ids = {t.id for t in category_templates}
                templates = [t for t in templates if t.id in template_ids]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Convert to response format
        template_list = []
        for template in templates:
            template_list.append({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "criteria": template.criteria.value,
                "version": template.version,
                "created_at": template.created_at,
                "tags": template.tags,
                "variables": [
                    {
                        "name": var.name,
                        "description": var.description,
                        "required": var.required,
                        "default_value": var.default_value
                    }
                    for var in template.variables
                ],
                "examples": template.examples
            })
        
        return {
            "templates": template_list,
            "total_count": len(template_list),
            "available_criteria": [criteria.value for criteria in EvaluationCriteria],
            "available_categories": [cat.value for cat in TemplateCategory]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/model-evaluation/templates/{template_id}")
async def get_evaluation_template(
    template_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """
    Get a specific evaluation template by ID.
    """
    try:
        from services.evaluation_templates import template_library
        
        template = template_library.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "criteria": template.criteria.value,
            "template_text": template.template_text,
            "instructions": template.instructions,
            "version": template.version,
            "created_at": template.created_at,
            "tags": template.tags,
            "variables": [
                {
                    "name": var.name,
                    "description": var.description,
                    "required": var.required,
                    "default_value": var.default_value,
                    "validation_pattern": var.validation_pattern
                }
                for var in template.variables
            ],
            "examples": template.examples
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.post("/model-evaluation/templates/{template_id}/render")
async def render_evaluation_template(
    template_id: str,
    variables: Dict[str, str],
    current_user: str = Depends(get_current_user_email)
):
    """
    Render an evaluation template with provided variables.
    Returns the complete evaluation prompt ready for use.
    """
    try:
        from services.evaluation_templates import template_library
        
        template = template_library.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Render the template
        rendered_prompt = template_library.render_template(template_id, variables)
        
        return {
            "template_id": template_id,
            "template_name": template.name,
            "rendered_prompt": rendered_prompt,
            "variables_used": variables,
            "criteria": template.criteria.value,
            "instructions": template.instructions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render template: {str(e)}"
        )


# Scoring calibration endpoints

@router.post("/model-evaluation/calibration/human-scores")
async def add_human_score(
    human_score_data: Dict[str, Any],
    current_user: str = Depends(get_current_user_email)
):
    """
    Add a human evaluation score for calibration training.
    """
    try:
        from services.scoring_calibration import calibration_system, HumanScore
        
        # Validate required fields
        required_fields = ["trace_id", "criteria", "score", "reasoning", "confidence", "evaluation_time_ms"]
        for field in required_fields:
            if field not in human_score_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create human score object
        human_score = HumanScore(
            trace_id=human_score_data["trace_id"],
            evaluator_email=current_user,
            criteria=EvaluationCriteria(human_score_data["criteria"]),
            score=float(human_score_data["score"]),
            reasoning=human_score_data["reasoning"],
            confidence=float(human_score_data["confidence"]),
            evaluation_time_ms=int(human_score_data["evaluation_time_ms"]),
            metadata=human_score_data.get("metadata", {})
        )
        
        # Validate score range
        if not 0.0 <= human_score.score <= 1.0:
            raise HTTPException(status_code=400, detail="Score must be between 0.0 and 1.0")
        
        if not 0.0 <= human_score.confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        
        # Add to calibration system
        success = await calibration_system.add_human_score(human_score)
        
        if success:
            return {
                "message": "Human score added successfully",
                "trace_id": human_score.trace_id,
                "criteria": human_score.criteria.value,
                "score": human_score.score
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add human score")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add human score: {str(e)}"
        )


@router.get("/model-evaluation/calibration/stats")
async def get_calibration_stats(
    current_user: str = Depends(get_current_user_email)
):
    """
    Get calibration system statistics and performance metrics.
    """
    try:
        from services.scoring_calibration import calibration_system
        
        stats = await calibration_system.get_calibration_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get calibration stats: {str(e)}"
        )


@router.post("/model-evaluation/calibration/train")
async def train_calibration_model(
    training_request: Dict[str, Any],
    current_user: str = Depends(get_current_user_email)
):
    """
    Train a calibration model for specific criteria and evaluator.
    """
    try:
        from services.scoring_calibration import calibration_system, CalibrationMethod
        
        # Validate required fields
        if "criteria" not in training_request or "evaluator_model" not in training_request:
            raise HTTPException(status_code=400, detail="Missing required fields: criteria, evaluator_model")
        
        criteria = EvaluationCriteria(training_request["criteria"])
        evaluator_model = training_request["evaluator_model"]
        method = CalibrationMethod(training_request.get("method", "isotonic_regression"))
        
        # Train the model
        success = await calibration_system.train_calibration_model(criteria, evaluator_model, method)
        
        if success:
            return {
                "message": "Calibration model trained successfully",
                "criteria": criteria.value,
                "evaluator_model": evaluator_model,
                "method": method.value
            }
        else:
            raise HTTPException(status_code=400, detail="Insufficient training data or training failed")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train calibration model: {str(e)}"
        )


@router.post("/model-evaluation/calibration/calibrate")
async def calibrate_score(
    calibration_request: Dict[str, Any],
    current_user: str = Depends(get_current_user_email)
):
    """
    Calibrate an AI-generated score using trained calibration models.
    """
    try:
        from services.scoring_calibration import calibration_system
        
        # Validate required fields
        required_fields = ["ai_score", "criteria", "evaluator_model"]
        for field in required_fields:
            if field not in calibration_request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        ai_score = float(calibration_request["ai_score"])
        criteria = EvaluationCriteria(calibration_request["criteria"])
        evaluator_model = calibration_request["evaluator_model"]
        confidence = float(calibration_request.get("confidence", 0.8))
        
        # Validate score range
        if not 0.0 <= ai_score <= 1.0:
            raise HTTPException(status_code=400, detail="AI score must be between 0.0 and 1.0")
        
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        
        # Calibrate the score
        calibration_result = await calibration_system.calibrate_score(
            ai_score=ai_score,
            criteria=criteria,
            evaluator_model=evaluator_model,
            confidence=confidence
        )
        
        return {
            "original_score": calibration_result.original_score,
            "calibrated_score": calibration_result.calibrated_score,
            "confidence_adjustment": calibration_result.confidence_adjustment,
            "calibration_method": calibration_result.calibration_method.value,
            "model_version": calibration_result.model_version,
            "metadata": calibration_result.metadata
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calibrate score: {str(e)}"
        )


@router.post("/model-evaluation/evaluate-with-calibration")
async def evaluate_with_calibration(
    evaluation_request: ModelEvaluationRequestSchema,
    current_user: str = Depends(get_current_user_email)
):
    """Evaluate a trace with automatic score calibration."""
    try:
        if not EVALUATOR_MODELS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Model evaluation service is not available"
            )
        
        # Convert schema to internal request
        from services.evaluator_models import EvaluationRequest, EvaluationCriteria
        
        # Parse criteria
        criteria_enums = []
        for criterion in evaluation_request.criteria:
            try:
                criteria_enums.append(EvaluationCriteria(criterion))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid evaluation criterion: {criterion}"
                )
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            trace_id=evaluation_request.trace_id,
            user_input=evaluation_request.user_input,
            model_output=evaluation_request.model_output,
            system_prompt=evaluation_request.system_prompt,
            context=evaluation_request.context or {},
            criteria=criteria_enums,
            custom_prompt=evaluation_request.custom_prompt,
            reference_answer=evaluation_request.reference_answer
        )
        
        # Perform evaluation with calibration
        result = await evaluator_manager.evaluate_single_with_calibration(
            eval_request,
            evaluation_request.evaluator_model,
            use_calibration=evaluation_request.use_calibration
        )
        
        # Return response
        return ModelEvaluationResponse(
            trace_id=result.trace_id,
            evaluator_model=result.evaluator_model,
            criteria=result.criteria.value,
            score=result.score,
            reasoning=result.reasoning,
            confidence=result.confidence,
            evaluation_time_ms=result.evaluation_time_ms or 0,
            cost_usd=result.cost_usd,
            metadata=result.metadata or {}
        )
        
    except Exception as e:
        logger.error(f"Error in evaluate_with_calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# BATCH PROCESSING ENDPOINTS
# ============================================

# Import batch processing components
try:
    from services.batch_evaluation import (
        batch_processor, BatchJob, BatchProgress, BatchStatus, BatchStrategy, 
        TaskPriority, EvaluationCriteria
    )
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    batch_processor = None
    BATCH_PROCESSING_AVAILABLE = False
    logger.warning("Batch processing service not available")

# Batch processing schemas
class BatchJobCreateRequest(BaseModel):
    """Schema for creating a batch evaluation job."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_ids: List[str] = Field(..., description="List of trace IDs to evaluate")
    criteria: List[str] = Field(..., description="Evaluation criteria to assess")
    evaluator_model: Optional[str] = Field(None, description="Specific evaluator model to use")
    strategy: str = Field("fifo", description="Batch processing strategy (fifo, priority, chunked, cost_optimized)")
    parallel_workers: int = Field(5, ge=1, le=20, description="Number of parallel workers")
    job_name: Optional[str] = Field(None, description="Name for the batch job")
    description: Optional[str] = Field(None, description="Description of the batch job")
    priority: str = Field("medium", description="Default task priority (low, medium, high, critical)")
    use_calibration: bool = Field(True, description="Whether to apply score calibration")
    custom_prompt: Optional[str] = Field(None, description="Custom evaluation prompt")

class BatchJobResponse(BaseModel):
    """Schema for batch job response."""
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    name: Optional[str]
    description: Optional[str]
    status: str
    strategy: str
    parallel_workers: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    estimated_total_cost: float
    actual_total_cost: float
    created_by: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

class BatchProgressResponse(BaseModel):
    """Schema for batch progress response."""
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    current_workers: int
    progress_percentage: float
    estimated_time_remaining_seconds: Optional[int]
    throughput_tasks_per_minute: float
    average_task_duration_ms: float
    estimated_total_cost: float
    actual_total_cost: float
    last_updated: str

class BatchJobListResponse(BaseModel):
    """Schema for batch job list response."""
    model_config = ConfigDict(protected_namespaces=())
    
    jobs: List[BatchJobResponse]
    total_count: int

class SystemStatsResponse(BaseModel):
    """Schema for batch system statistics."""
    model_config = ConfigDict(protected_namespaces=())
    
    active_jobs: int
    running_jobs: int
    total_workers: int
    max_workers: int
    total_processed: int
    total_errors: int
    uptime_seconds: float
    success_rate: float
    throughput_per_hour: float

@router.post("/batch-evaluation/jobs", response_model=BatchJobResponse)
async def create_batch_job(
    request: BatchJobCreateRequest,
    current_user: str = Depends(get_current_user_email)
):
    """Create a new batch evaluation job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        # Parse criteria
        criteria_enums = []
        for criterion in request.criteria:
            try:
                criteria_enums.append(EvaluationCriteria(criterion))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid evaluation criterion: {criterion}"
                )
        
        # Parse strategy
        try:
            strategy = BatchStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid batch strategy: {request.strategy}"
            )
        
        # Parse priority
        try:
            priority = TaskPriority(request.priority)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task priority: {request.priority}"
            )
        
        # Create batch job
        job = await batch_processor.create_batch_job(
            trace_ids=request.trace_ids,
            criteria=criteria_enums,
            evaluator_model=request.evaluator_model,
            strategy=strategy,
            parallel_workers=request.parallel_workers,
            job_name=request.job_name,
            description=request.description,
            priority=priority,
            use_calibration=request.use_calibration,
            custom_prompt=request.custom_prompt,
            created_by=current_user
        )
        
        return BatchJobResponse(
            job_id=job.id,
            name=job.name,
            description=job.description,
            status=job.status.value,
            strategy=job.strategy.value,
            parallel_workers=job.parallel_workers,
            total_tasks=job.total_tasks,
            completed_tasks=job.completed_tasks,
            failed_tasks=job.failed_tasks,
            skipped_tasks=job.skipped_tasks,
            estimated_total_cost=job.estimated_total_cost,
            actual_total_cost=job.actual_total_cost,
            created_by=job.created_by,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating batch job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-evaluation/jobs/{job_id}/start")
async def start_batch_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Start processing a batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        success = await batch_processor.start_batch_job(job_id)
        
        if success:
            return {"message": f"Batch job {job_id} started successfully", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to start batch job")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-evaluation/jobs/{job_id}/pause")
async def pause_batch_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Pause a running batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        success = await batch_processor.pause_batch_job(job_id)
        
        if success:
            return {"message": f"Batch job {job_id} paused successfully", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to pause batch job")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error pausing batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-evaluation/jobs/{job_id}/resume")
async def resume_batch_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Resume a paused batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        success = await batch_processor.resume_batch_job(job_id)
        
        if success:
            return {"message": f"Batch job {job_id} resumed successfully", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to resume batch job")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error resuming batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-evaluation/jobs/{job_id}/cancel")
async def cancel_batch_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Cancel a batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        success = await batch_processor.cancel_batch_job(job_id)
        
        if success:
            return {"message": f"Batch job {job_id} cancelled successfully", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel batch job")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error cancelling batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-evaluation/jobs/{job_id}", response_model=BatchJobResponse)
async def get_batch_job(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Get details of a specific batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        job = await batch_processor.get_batch_job(job_id)
        
        return BatchJobResponse(
            job_id=job.id,
            name=job.name,
            description=job.description,
            status=job.status.value,
            strategy=job.strategy.value,
            parallel_workers=job.parallel_workers,
            total_tasks=job.total_tasks,
            completed_tasks=job.completed_tasks,
            failed_tasks=job.failed_tasks,
            skipped_tasks=job.skipped_tasks,
            estimated_total_cost=job.estimated_total_cost,
            actual_total_cost=job.actual_total_cost,
            created_by=job.created_by,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting batch job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-evaluation/jobs/{job_id}/progress", response_model=BatchProgressResponse)
async def get_batch_progress(
    job_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Get real-time progress of a batch job."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        progress = await batch_processor.get_batch_progress(job_id)
        
        return BatchProgressResponse(
            job_id=progress.job_id,
            status=progress.status.value,
            total_tasks=progress.total_tasks,
            completed_tasks=progress.completed_tasks,
            failed_tasks=progress.failed_tasks,
            skipped_tasks=progress.skipped_tasks,
            current_workers=progress.current_workers,
            progress_percentage=progress.progress_percentage,
            estimated_time_remaining_seconds=progress.estimated_time_remaining_seconds,
            throughput_tasks_per_minute=progress.throughput_tasks_per_minute,
            average_task_duration_ms=progress.average_task_duration_ms,
            estimated_total_cost=progress.estimated_total_cost,
            actual_total_cost=progress.actual_total_cost,
            last_updated=progress.last_updated
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting batch progress {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-evaluation/jobs", response_model=BatchJobListResponse)
async def list_batch_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    current_user: str = Depends(get_current_user_email)
):
    """List all batch jobs, optionally filtered by status."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = BatchStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status filter: {status}"
                )
        
        jobs = await batch_processor.list_batch_jobs(status_filter)
        
        job_responses = []
        for job in jobs:
            job_responses.append(BatchJobResponse(
                job_id=job.id,
                name=job.name,
                description=job.description,
                status=job.status.value,
                strategy=job.strategy.value,
                parallel_workers=job.parallel_workers,
                total_tasks=job.total_tasks,
                completed_tasks=job.completed_tasks,
                failed_tasks=job.failed_tasks,
                skipped_tasks=job.skipped_tasks,
                estimated_total_cost=job.estimated_total_cost,
                actual_total_cost=job.actual_total_cost,
                created_by=job.created_by,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at
            ))
        
        return BatchJobListResponse(
            jobs=job_responses,
            total_count=len(job_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing batch jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-evaluation/system/stats", response_model=SystemStatsResponse)
async def get_batch_system_stats(
    current_user: str = Depends(get_current_user_email)
):
    """Get system-wide batch processing statistics."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        stats = await batch_processor.get_system_stats()
        
        return SystemStatsResponse(
            active_jobs=stats["active_jobs"],
            running_jobs=stats["running_jobs"],
            total_workers=stats["total_workers"],
            max_workers=stats["max_workers"],
            total_processed=stats["total_processed"],
            total_errors=stats["total_errors"],
            uptime_seconds=stats["uptime_seconds"],
            success_rate=stats["success_rate"],
            throughput_per_hour=stats["throughput_per_hour"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-evaluation/system/cleanup")
async def cleanup_completed_jobs(
    older_than_hours: int = Query(24, ge=1, le=168, description="Remove completed jobs older than this many hours"),
    current_user: str = Depends(get_current_user_email)
):
    """Clean up completed batch jobs older than specified time."""
    try:
        if not BATCH_PROCESSING_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Batch processing service is not available"
            )
        
        cleaned_count = await batch_processor.cleanup_completed_jobs(older_than_hours)
        
        return {
            "message": f"Cleaned up {cleaned_count} completed jobs",
            "cleaned_jobs": cleaned_count,
            "older_than_hours": older_than_hours
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# PERFORMANCE ANALYTICS AND MONITORING ENDPOINTS
# ============================================

# Import performance analytics components
try:
    from services.performance_analytics import (
        performance_analytics, AnalyticsReport, TimeRange, MetricType, 
        PerformanceMetric, ModelPerformanceComparison, SystemAlert
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    performance_analytics = None
    ANALYTICS_AVAILABLE = False
    logger.warning("Performance analytics service not available")

# Analytics response schemas
class SystemOverviewResponse(BaseModel):
    """Schema for system overview response."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_range: str
    start_time: str
    end_time: str
    evaluation_metrics: Dict[str, Any]
    batch_metrics: Dict[str, Any]
    cost_metrics: Dict[str, Any]
    performance_trends: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    calibration_metrics: Dict[str, Any]

class AnalyticsReportResponse(BaseModel):
    """Schema for analytics report response."""
    model_config = ConfigDict(protected_namespaces=())
    
    report_id: str
    title: str
    time_range: str
    generated_at: str
    metrics: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]

class ModelComparisonResponse(BaseModel):
    """Schema for model performance comparison."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    total_evaluations: int
    success_rate: float
    average_latency_ms: float
    total_cost_usd: float
    average_score: float
    score_std_dev: float
    cost_per_evaluation: float
    throughput_per_hour: float
    error_rate: float
    calibration_accuracy: Optional[float] = None

class SystemHealthResponse(BaseModel):
    """Schema for system health response."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    uptime_seconds: float
    memory_usage: str
    cpu_usage: str
    database_status: str
    batch_processor_status: str
    calibration_system_status: str

class AlertResponse(BaseModel):
    """Schema for system alert response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    level: str
    title: str
    message: str
    metric_type: str
    threshold_value: float
    current_value: float
    timestamp: str
    resolved: bool

@router.get("/analytics/overview", response_model=SystemOverviewResponse)
async def get_system_overview(
    time_range: str = Query("24h", description="Time range for analytics (1h, 24h, 7d, 30d, 90d, 365d)"),
    current_user: str = Depends(get_current_user_email)
):
    """Get comprehensive system performance overview."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        # Validate time range
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}. Valid options: 1h, 24h, 7d, 30d, 90d, 365d"
            )
        
        overview = await performance_analytics.get_system_overview(time_range_enum)
        
        return SystemOverviewResponse(**overview)
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/reports", response_model=AnalyticsReportResponse)
async def generate_analytics_report(
    time_range: str = Query("24h", description="Time range for report"),
    include_recommendations: bool = Query(True, description="Include optimization recommendations"),
    current_user: str = Depends(get_current_user_email)
):
    """Generate comprehensive analytics report."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        # Validate time range
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}"
            )
        
        report = await performance_analytics.generate_analytics_report(
            time_range_enum, 
            include_recommendations
        )
        
        # Convert report to response format
        metrics_dict = {}
        for name, metric in report.metrics.items():
            metrics_dict[name] = {
                "type": metric.metric_type.value,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp,
                "metadata": metric.metadata
            }
        
        trends_dict = {}
        for name, trend in report.trends.items():
            trends_dict[name] = {
                "timestamps": trend.timestamps,
                "values": trend.values,
                "labels": trend.labels
            }
        
        return AnalyticsReportResponse(
            report_id=report.report_id,
            title=report.title,
            time_range=report.time_range.value,
            generated_at=report.generated_at,
            metrics=metrics_dict,
            trends=trends_dict,
            insights=report.insights,
            recommendations=report.recommendations
        )
        
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/models/comparison", response_model=List[ModelComparisonResponse])
async def get_model_performance_comparison(
    time_range: str = Query("24h", description="Time range for comparison"),
    current_user: str = Depends(get_current_user_email)
):
    """Get performance comparison between different evaluator models."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        # Validate time range
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}"
            )
        
        overview = await performance_analytics.get_system_overview(time_range_enum)
        model_comparisons = overview["model_comparison"]
        
        responses = []
        for comparison in model_comparisons:
            responses.append(ModelComparisonResponse(
                model_name=comparison.model_name,
                total_evaluations=comparison.total_evaluations,
                success_rate=comparison.success_rate,
                average_latency_ms=comparison.average_latency_ms,
                total_cost_usd=comparison.total_cost_usd,
                average_score=comparison.average_score,
                score_std_dev=comparison.score_std_dev,
                cost_per_evaluation=comparison.cost_per_evaluation,
                throughput_per_hour=comparison.throughput_per_hour,
                error_rate=comparison.error_rate,
                calibration_accuracy=comparison.calibration_accuracy
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error getting model comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: str = Depends(get_current_user_email)
):
    """Get overall system health status."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        overview = await performance_analytics.get_system_overview(TimeRange.HOUR)
        health = overview["system_health"]
        
        return SystemHealthResponse(
            status=health["status"],
            uptime_seconds=health["uptime_seconds"],
            memory_usage=health["memory_usage"],
            cpu_usage=health["cpu_usage"],
            database_status=health["database_status"],
            batch_processor_status=health["batch_processor_status"],
            calibration_system_status=health["calibration_system_status"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/alerts", response_model=List[AlertResponse])
async def get_system_alerts(
    current_user: str = Depends(get_current_user_email)
):
    """Get active system alerts."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        overview = await performance_analytics.get_system_overview(TimeRange.HOUR)
        alerts = overview["alerts"]
        
        alert_responses = []
        for alert in alerts:
            alert_responses.append(AlertResponse(
                id=alert.id,
                level=alert.level.value,
                title=alert.title,
                message=alert.message,
                metric_type=alert.metric_type.value,
                threshold_value=alert.threshold_value,
                current_value=alert.current_value,
                timestamp=alert.timestamp,
                resolved=alert.resolved
            ))
        
        return alert_responses
        
    except Exception as e:
        logger.error(f"Error getting system alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/metrics/{metric_type}")
async def get_specific_metric(
    metric_type: str,
    time_range: str = Query("24h", description="Time range for metric"),
    current_user: str = Depends(get_current_user_email)
):
    """Get specific performance metric data."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        # Validate metric type
        try:
            metric_type_enum = MetricType(metric_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric type: {metric_type}. Valid options: throughput, accuracy, cost, latency, success_rate, calibration_performance"
            )
        
        # Validate time range
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}"
            )
        
        overview = await performance_analytics.get_system_overview(time_range_enum)
        
        # Extract specific metric data based on type
        if metric_type_enum == MetricType.THROUGHPUT:
            return {
                "metric_type": metric_type,
                "value": overview["evaluation_metrics"]["throughput_per_hour"],
                "unit": "evaluations/hour",
                "trend": overview["performance_trends"].get("throughput", {}),
                "time_range": time_range
            }
        elif metric_type_enum == MetricType.SUCCESS_RATE:
            return {
                "metric_type": metric_type,
                "value": overview["evaluation_metrics"]["success_rate"],
                "unit": "percentage",
                "trend": overview["performance_trends"].get("success_rate", {}),
                "time_range": time_range
            }
        elif metric_type_enum == MetricType.COST:
            return {
                "metric_type": metric_type,
                "value": overview["cost_metrics"]["total_cost_usd"],
                "unit": "USD",
                "trend": overview["cost_metrics"].get("cost_trend", {}),
                "time_range": time_range
            }
        else:
            return {
                "metric_type": metric_type,
                "message": "Metric data not available",
                "time_range": time_range
            }
        
    except Exception as e:
        logger.error(f"Error getting metric {metric_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/reports/export")
async def export_analytics_report(
    time_range: str = Query("24h", description="Time range for report"),
    format: str = Query("json", description="Export format (json, csv)"),
    include_recommendations: bool = Query(True, description="Include recommendations"),
    current_user: str = Depends(get_current_user_email)
):
    """Export analytics report in specified format."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        # Validate parameters
        if format not in ["json", "csv"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Valid options: json, csv"
            )
        
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}"
            )
        
        # Generate report
        report = await performance_analytics.generate_analytics_report(
            time_range_enum, 
            include_recommendations
        )
        
        # Export in requested format
        exported_data = await performance_analytics.export_report_data(report, format)
        
        if format == "csv":
            return Response(
                content=exported_data,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics_report_{time_range}.csv"}
            )
        else:
            return exported_data
        
    except Exception as e:
        logger.error(f"Error exporting analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/cost-optimization")
async def get_cost_optimization_recommendations(
    time_range: str = Query("7d", description="Time range for analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Get cost optimization recommendations."""
    try:
        if not ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Performance analytics service is not available"
            )
        
        try:
            time_range_enum = TimeRange(time_range)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time range: {time_range}"
            )
        
        overview = await performance_analytics.get_system_overview(time_range_enum)
        
        # Extract cost-related insights and recommendations
        cost_metrics = overview["cost_metrics"]
        model_comparison = overview["model_comparison"]
        
        # Calculate potential savings
        potential_savings = 0
        optimizations = []
        
        if model_comparison and len(model_comparison) > 1:
            most_expensive = max(model_comparison, key=lambda x: x.cost_per_evaluation)
            least_expensive = min(model_comparison, key=lambda x: x.cost_per_evaluation)
            
            if most_expensive.cost_per_evaluation > least_expensive.cost_per_evaluation * 1.5:
                savings_per_eval = most_expensive.cost_per_evaluation - least_expensive.cost_per_evaluation
                potential_daily_savings = savings_per_eval * most_expensive.total_evaluations / ((time_range_enum == TimeRange.DAY and 1) or 7)
                potential_savings += potential_daily_savings
                
                optimizations.append({
                    "type": "model_substitution",
                    "description": f"Replace {most_expensive.model_name} with {least_expensive.model_name}",
                    "current_cost": most_expensive.cost_per_evaluation,
                    "optimized_cost": least_expensive.cost_per_evaluation,
                    "potential_daily_savings": potential_daily_savings,
                    "impact": "Replace expensive model with cost-efficient alternative"
                })
        
        return {
            "time_range": time_range,
            "analysis_period": f"{overview['start_time']} to {overview['end_time']}",
            "current_metrics": {
                "total_cost": cost_metrics["total_cost_usd"],
                "average_cost_per_evaluation": cost_metrics["average_cost_per_evaluation"],
                "monthly_projection": cost_metrics["monthly_projection_usd"]
            },
            "optimization_opportunities": optimizations,
            "potential_monthly_savings": potential_savings * 30,
            "cost_breakdown_by_model": cost_metrics["cost_by_model"],
            "recommendations": [rec for rec in overview.get("recommendations", []) if "cost" in rec.lower() or "expensive" in rec.lower()]
        }
        
    except Exception as e:
        logger.error(f"Error getting cost optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 

# === USER ANALYTICS ENDPOINTS ===

class UserEngagementResponse(BaseModel):
    """Schema for user engagement overview response."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    start_time: str
    end_time: str
    user_metrics: Dict[str, Any]
    feature_usage: List[Dict[str, Any]]
    user_journeys: List[Dict[str, Any]]
    engagement_trends: Dict[str, Any]
    retention_metrics: Dict[str, Any]

class AgreementAnalysisResponse(BaseModel):
    """Schema for LLM-human agreement analysis response."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_comparisons: int
    agreement_rate: float
    strong_agreement_rate: float
    disagreement_patterns: Dict[str, Any]
    confidence_correlation: float
    bias_indicators: Dict[str, float]
    model_reliability_scores: Dict[str, float]

class AcceptanceRateResponse(BaseModel):
    """Schema for human acceptance rate response."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_ai_suggestions: int
    accepted_suggestions: int
    rejected_suggestions: int
    acceptance_rate: float
    acceptance_by_confidence: Dict[str, float]
    acceptance_by_criteria: Dict[str, float]
    trust_trend_over_time: List[Dict[str, Any]]

@router.get("/analytics/user-engagement", response_model=UserEngagementResponse)
async def get_user_engagement_analytics(
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days for analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Get comprehensive user engagement and behavior analytics."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        engagement_data = await user_analytics.get_user_engagement_overview(time_range_days)
        
        return UserEngagementResponse(**engagement_data)
        
    except Exception as e:
        logger.error(f"Error getting user engagement analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/llm-human-agreement", response_model=AgreementAnalysisResponse)
async def get_llm_human_agreement_analysis(
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days for analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Analyze agreement between LLM and human evaluations."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        agreement_data = await user_analytics.analyze_llm_human_agreement(time_range_days)
        
        # Convert dataclass to dict for response
        agreement_dict = {
            "time_period": agreement_data.time_period,
            "total_comparisons": agreement_data.total_comparisons,
            "agreement_rate": agreement_data.agreement_rate,
            "strong_agreement_rate": agreement_data.strong_agreement_rate,
            "disagreement_patterns": agreement_data.disagreement_patterns,
            "confidence_correlation": agreement_data.confidence_correlation,
            "bias_indicators": agreement_data.bias_indicators,
            "model_reliability_scores": agreement_data.model_reliability_scores
        }
        
        return AgreementAnalysisResponse(**agreement_dict)
        
    except Exception as e:
        logger.error(f"Error analyzing LLM-human agreement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/acceptance-rates", response_model=AcceptanceRateResponse)
async def get_acceptance_rate_analysis(
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days for analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Analyze human acceptance rates of AI evaluation suggestions."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        acceptance_data = await user_analytics.analyze_acceptance_rates(time_range_days)
        
        # Convert dataclass to dict and format trust trend
        trust_trend_formatted = [
            {"timestamp": dt.isoformat(), "trust_score": score}
            for dt, score in acceptance_data.trust_trend_over_time
        ]
        
        acceptance_dict = {
            "time_period": acceptance_data.time_period,
            "total_ai_suggestions": acceptance_data.total_ai_suggestions,
            "accepted_suggestions": acceptance_data.accepted_suggestions,
            "rejected_suggestions": acceptance_data.rejected_suggestions,
            "acceptance_rate": acceptance_data.acceptance_rate,
            "acceptance_by_confidence": acceptance_data.acceptance_by_confidence,
            "acceptance_by_criteria": acceptance_data.acceptance_by_criteria,
            "trust_trend_over_time": trust_trend_formatted
        }
        
        return AcceptanceRateResponse(**acceptance_dict)
        
    except Exception as e:
        logger.error(f"Error analyzing acceptance rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/feature-adoption")
async def get_feature_adoption_metrics(
    time_range_days: int = Query(30, ge=1, le=365, description="Time range in days for analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Get feature adoption and usage metrics."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        engagement_data = await user_analytics.get_user_engagement_overview(time_range_days)
        
        return {
            "time_period": f"{time_range_days} days",
            "feature_usage": engagement_data["feature_usage"],
            "total_active_users": engagement_data["user_metrics"]["active_users"],
            "adoption_summary": {
                "most_popular_feature": max(
                    engagement_data["feature_usage"], 
                    key=lambda x: x["usage_count"], 
                    default={"feature_name": "None"}
                )["feature_name"],
                "highest_adoption_rate": max(
                    [f["adoption_rate"] for f in engagement_data["feature_usage"]], 
                    default=0
                ),
                "average_usage_per_user": sum(
                    f["avg_usage_per_user"] for f in engagement_data["feature_usage"]
                ) / len(engagement_data["feature_usage"]) if engagement_data["feature_usage"] else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting feature adoption metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/user-retention")
async def get_user_retention_analysis(
    cohort_period: str = Query("weekly", description="Cohort analysis period (daily, weekly, monthly)"),
    current_user: str = Depends(get_current_user_email)
):
    """Get user retention and cohort analysis."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        # For now, return simplified retention metrics
        # In a full implementation, this would do cohort analysis
        engagement_data = await user_analytics.get_user_engagement_overview(30)
        
        return {
            "cohort_period": cohort_period,
            "retention_metrics": engagement_data["retention_metrics"],
            "user_journeys_summary": {
                "total_users": len(engagement_data["user_journeys"]),
                "onboarded_users": sum(
                    1 for journey in engagement_data["user_journeys"] 
                    if journey.get("onboarding_completed", False)
                ),
                "average_evaluations_per_user": sum(
                    journey.get("total_evaluations", 0) 
                    for journey in engagement_data["user_journeys"]
                ) / len(engagement_data["user_journeys"]) if engagement_data["user_journeys"] else 0
            },
            "engagement_trends": engagement_data["engagement_trends"]
        }
        
    except Exception as e:
        logger.error(f"Error getting user retention analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/quality-improvement")
async def get_quality_improvement_trends(
    time_range_days: int = Query(90, ge=7, le=365, description="Time range in days for trend analysis"),
    current_user: str = Depends(get_current_user_email)
):
    """Get quality improvement trends and metrics over time."""
    try:
        if not USER_ANALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="User analytics service is not available"
            )
        
        # Combine performance analytics and user analytics for quality trends
        if ANALYTICS_AVAILABLE:
            performance_overview = await performance_analytics.get_system_overview(TimeRange.QUARTER)
        else:
            performance_overview = {}
        
        agreement_data = await user_analytics.analyze_llm_human_agreement(time_range_days)
        engagement_data = await user_analytics.get_user_engagement_overview(time_range_days)
        
        return {
            "time_period": f"{time_range_days} days",
            "quality_metrics": {
                "evaluation_accuracy_trend": performance_overview.get("performance_trends", {}).get("success_rate", {}),
                "llm_human_agreement": {
                    "current_agreement_rate": agreement_data.agreement_rate,
                    "strong_agreement_rate": agreement_data.strong_agreement_rate,
                    "bias_indicators": agreement_data.bias_indicators
                },
                "model_reliability": agreement_data.model_reliability_scores,
                "calibration_effectiveness": performance_overview.get("calibration_metrics", {})
            },
            "improvement_indicators": {
                "user_confidence_trend": "Improving based on higher engagement",
                "evaluation_consistency": "Stable with " + f"{agreement_data.agreement_rate:.1%} agreement rate",
                "system_optimization": "Cost optimization achieving savings" if performance_overview.get("cost_metrics") else "Not available"
            },
            "recommendations": [
                "Focus on improving agreement rates between AI and human evaluators",
                "Monitor model reliability scores and retrain underperforming models",
                "Implement user feedback loops to improve evaluation quality",
                "Continue cost optimization while maintaining evaluation accuracy"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting quality improvement trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DATA EXPORT SYSTEM - Multi-Format Export Capabilities
# ============================================================================

class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines format for fine-tuning


class ExportDataType(str, Enum):
    """Types of data that can be exported."""
    TRACES = "traces"
    EVALUATIONS = "evaluations"
    COMBINED = "combined"
    FINE_TUNING = "fine_tuning"


class ExportRequest(BaseModel):
    """Schema for data export requests."""
    model_config = ConfigDict(protected_namespaces=())
    
    format: ExportFormat = Field(..., description="Export format (json, csv, jsonl)")
    data_type: ExportDataType = Field(..., description="Type of data to export")
    filters: Optional[AdvancedFilterRequest] = Field(None, description="Filter configuration")
    include_fields: Optional[List[str]] = Field(None, description="Specific fields to include")
    exclude_fields: Optional[List[str]] = Field(None, description="Specific fields to exclude")
    compress: bool = Field(False, description="Whether to gzip compress the output")
    
    # Fine-tuning specific options
    fine_tuning_format: Optional[str] = Field("openai", description="Fine-tuning format (openai, anthropic, custom)")
    include_system_prompts: bool = Field(True, description="Include system prompts in fine-tuning export")
    max_records: Optional[int] = Field(None, ge=1, le=1000000, description="Maximum number of records to export")
    
    # CSV specific options
    csv_delimiter: str = Field(",", description="CSV delimiter character")
    csv_quote_char: str = Field('"', description="CSV quote character")
    include_headers: bool = Field(True, description="Include CSV headers")


class ExportResponse(BaseModel):
    """Schema for export response."""
    model_config = ConfigDict(protected_namespaces=())
    
    export_id: str = Field(..., description="Unique export identifier")
    status: str = Field(..., description="Export status (processing, completed, failed)")
    format: str = Field(..., description="Export format")
    data_type: str = Field(..., description="Data type exported")
    total_records: int = Field(..., description="Total number of records exported")
    file_size_bytes: int = Field(..., description="File size in bytes")
    download_url: Optional[str] = Field(None, description="Download URL (if completed)")
    created_at: str = Field(..., description="Export creation timestamp")
    completed_at: Optional[str] = Field(None, description="Export completion timestamp")
    expires_at: str = Field(..., description="When the export file expires")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Export metadata")


class AvailableFormatsResponse(BaseModel):
    """Schema for available export formats response."""
    model_config = ConfigDict(protected_namespaces=())
    
    formats: List[Dict[str, Any]] = Field(..., description="Available export formats with descriptions")
    data_types: List[Dict[str, Any]] = Field(..., description="Available data types for export")
    fine_tuning_formats: List[Dict[str, Any]] = Field(..., description="Available fine-tuning formats")


import csv
import json
import gzip
import io
from typing import Generator
from fastapi.responses import StreamingResponse


def format_trace_for_fine_tuning(
    trace: Dict[str, Any], 
    evaluations: List[Dict[str, Any]], 
    format_type: str = "openai"
) -> Dict[str, Any]:
    """Convert trace and evaluation data to fine-tuning format."""
    
    if format_type == "openai":
        # OpenAI fine-tuning format
        messages = []
        
        if trace.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": trace["system_prompt"]
            })
        
        messages.append({
            "role": "user", 
            "content": trace["user_input"]
        })
        
        messages.append({
            "role": "assistant",
            "content": trace["model_output"]
        })
        
        # Add evaluation as metadata or additional context
        best_evaluation = None
        if evaluations:
            # Use human evaluation if available, otherwise best model evaluation
            human_evals = [e for e in evaluations if e.get("evaluator_type") == "human"]
            if human_evals:
                best_evaluation = human_evals[0]
            else:
                best_evaluation = max(evaluations, key=lambda x: x.get("score", 0))
        
        result = {"messages": messages}
        
        if best_evaluation:
            result["metadata"] = {
                "evaluation_score": best_evaluation.get("score"),
                "evaluation_label": best_evaluation.get("label"),
                "evaluation_critique": best_evaluation.get("critique"),
                "trace_id": trace["id"]
            }
        
        return result
    
    elif format_type == "anthropic":
        # Anthropic fine-tuning format
        return {
            "input": trace["user_input"],
            "output": trace["model_output"],
            "system": trace.get("system_prompt", ""),
            "metadata": {
                "trace_id": trace["id"],
                "model_name": trace.get("model_name"),
                "evaluations": evaluations
            }
        }
    
    else:  # custom format
        return {
            "conversation": {
                "system": trace.get("system_prompt"),
                "user": trace["user_input"],
                "assistant": trace["model_output"]
            },
            "metadata": {
                "trace_id": trace["id"],
                "timestamp": trace["timestamp"],
                "model_name": trace.get("model_name"),
                "latency_ms": trace.get("latency_ms"),
                "cost_usd": trace.get("cost_usd")
            },
            "evaluations": evaluations
        }


def convert_to_csv_row(data: Dict[str, Any], flatten_nested: bool = True) -> Dict[str, Any]:
    """Convert complex data structure to flat CSV-compatible row."""
    row = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and flatten_nested:
            # Flatten nested dictionaries
            for nested_key, nested_value in value.items():
                row[f"{key}_{nested_key}"] = str(nested_value) if nested_value is not None else ""
        elif isinstance(value, list):
            # Convert lists to JSON strings
            row[key] = json.dumps(value) if value else ""
        else:
            row[key] = str(value) if value is not None else ""
    
    return row


def generate_export_data(
    traces: List[Dict[str, Any]], 
    evaluations_map: Dict[str, List[Dict[str, Any]]],
    export_request: ExportRequest
) -> Generator[str, None, None]:
    """Generate export data in the requested format."""
    
    if export_request.format == ExportFormat.JSON:
        # JSON format - return complete data structure
        export_data = []
        
        for trace in traces:
            trace_evaluations = evaluations_map.get(trace["id"], [])
            
            if export_request.data_type == ExportDataType.TRACES:
                export_data.append(trace)
            elif export_request.data_type == ExportDataType.EVALUATIONS:
                export_data.extend(trace_evaluations)
            elif export_request.data_type == ExportDataType.COMBINED:
                export_data.append({
                    **trace,
                    "evaluations": trace_evaluations
                })
            elif export_request.data_type == ExportDataType.FINE_TUNING:
                export_data.append(format_trace_for_fine_tuning(
                    trace, trace_evaluations, export_request.fine_tuning_format
                ))
        
        yield json.dumps(export_data, indent=2, default=str)
    
    elif export_request.format == ExportFormat.JSONL:
        # JSON Lines format - one JSON object per line
        for trace in traces:
            trace_evaluations = evaluations_map.get(trace["id"], [])
            
            if export_request.data_type == ExportDataType.FINE_TUNING:
                data = format_trace_for_fine_tuning(
                    trace, trace_evaluations, export_request.fine_tuning_format
                )
            elif export_request.data_type == ExportDataType.COMBINED:
                data = {**trace, "evaluations": trace_evaluations}
            else:
                data = trace
            
            yield json.dumps(data, default=str) + "\n"
    
    elif export_request.format == ExportFormat.CSV:
        # CSV format - tabular data
        output = io.StringIO()
        writer = None
        headers_written = False
        
        all_rows = []
        
        for trace in traces:
            trace_evaluations = evaluations_map.get(trace["id"], [])
            
            if export_request.data_type == ExportDataType.TRACES:
                all_rows.append(convert_to_csv_row(trace))
            elif export_request.data_type == ExportDataType.EVALUATIONS:
                for evaluation in trace_evaluations:
                    all_rows.append(convert_to_csv_row(evaluation))
            elif export_request.data_type == ExportDataType.COMBINED:
                combined_data = {**trace, "evaluations": trace_evaluations}
                all_rows.append(convert_to_csv_row(combined_data))
            elif export_request.data_type == ExportDataType.FINE_TUNING:
                fine_tuning_data = format_trace_for_fine_tuning(
                    trace, trace_evaluations, export_request.fine_tuning_format
                )
                all_rows.append(convert_to_csv_row(fine_tuning_data))
        
        if all_rows:
            # Get all possible headers from all rows
            all_headers = set()
            for row in all_rows:
                all_headers.update(row.keys())
            
            fieldnames = sorted(list(all_headers))
            writer = csv.DictWriter(
                output, 
                fieldnames=fieldnames,
                delimiter=export_request.csv_delimiter,
                quotechar=export_request.csv_quote_char
            )
            
            if export_request.include_headers:
                writer.writeheader()
            
            for row in all_rows:
                writer.writerow(row)
        
        yield output.getvalue()


@router.get("/export/formats", response_model=AvailableFormatsResponse)
async def get_available_export_formats(
    current_user: str = Depends(get_current_user_email)
):
    """Get available export formats and data types."""
    
    formats = [
        {
            "format": "json",
            "name": "JSON",
            "description": "Complete data structure in JSON format",
            "supports_compression": True,
            "best_for": ["API integration", "data analysis", "backup"]
        },
        {
            "format": "csv", 
            "name": "CSV",
            "description": "Tabular data for spreadsheet applications",
            "supports_compression": True,
            "best_for": ["Excel analysis", "data visualization", "reporting"]
        },
        {
            "format": "jsonl",
            "name": "JSON Lines",
            "description": "Line-delimited JSON, optimized for fine-tuning",
            "supports_compression": True,
            "best_for": ["ML training", "streaming processing", "large datasets"]
        }
    ]
    
    data_types = [
        {
            "type": "traces",
            "name": "Traces Only",
            "description": "Export trace data without evaluations"
        },
        {
            "type": "evaluations", 
            "name": "Evaluations Only",
            "description": "Export evaluation data without traces"
        },
        {
            "type": "combined",
            "name": "Combined Data",
            "description": "Export traces with their evaluations"
        },
        {
            "type": "fine_tuning",
            "name": "Fine-tuning Dataset",
            "description": "Export in format optimized for model fine-tuning"
        }
    ]
    
    fine_tuning_formats = [
        {
            "format": "openai",
            "name": "OpenAI",
            "description": "OpenAI fine-tuning format with messages array"
        },
        {
            "format": "anthropic",
            "name": "Anthropic",
            "description": "Anthropic fine-tuning format with input/output structure"
        },
        {
            "format": "custom",
            "name": "Custom",
            "description": "Generic fine-tuning format with conversation structure"
        }
    ]
    
    return AvailableFormatsResponse(
        formats=formats,
        data_types=data_types,
        fine_tuning_formats=fine_tuning_formats
    )


@router.post("/export/data", response_model=ExportResponse)
async def export_data(
    export_request: ExportRequest,
    current_user: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """Export data in the specified format with optional filtering."""
    
    try:
        # Generate unique export ID
        import uuid
        export_id = str(uuid.uuid4())
        
        # Start with basic query
        query = select(Trace).options(selectinload(Trace.evaluations))
        
        # Apply simple filtering if provided (simplified approach for MVP)
        if export_request.filters:
            filters = export_request.filters
            
            # Apply basic filters
            if filters.model_names:
                query = query.where(Trace.model_name.in_(filters.model_names))
            
            if filters.session_ids:
                query = query.where(Trace.session_id.in_(filters.session_ids))
            
            if filters.trace_statuses:
                query = query.where(Trace.status.in_(filters.trace_statuses))
            
            # Apply date range filter
            if filters.trace_date_range:
                if filters.trace_date_range.start_date:
                    query = query.where(Trace.timestamp >= filters.trace_date_range.start_date)
                if filters.trace_date_range.end_date:
                    query = query.where(Trace.timestamp <= filters.trace_date_range.end_date)
            
            # Apply numeric range filters
            if filters.latency_range:
                if filters.latency_range.min_value is not None:
                    query = query.where(Trace.latency_ms >= filters.latency_range.min_value)
                if filters.latency_range.max_value is not None:
                    query = query.where(Trace.latency_ms <= filters.latency_range.max_value)
            
            if filters.cost_range:
                if filters.cost_range.min_value is not None:
                    query = query.where(Trace.cost_usd >= filters.cost_range.min_value)
                if filters.cost_range.max_value is not None:
                    query = query.where(Trace.cost_usd <= filters.cost_range.max_value)
            
            # Apply text search
            if filters.search_query:
                search_conditions = []
                if "user_input" in filters.search_in_fields:
                    search_conditions.append(Trace.user_input.ilike(f"%{filters.search_query}%"))
                if "model_output" in filters.search_in_fields:
                    search_conditions.append(Trace.model_output.ilike(f"%{filters.search_query}%"))
                
                if search_conditions:
                    if filters.filter_operator == "OR":
                        query = query.where(or_(*search_conditions))
                    else:
                        query = query.where(and_(*search_conditions))
        
        # Apply max records limit
        if export_request.max_records:
            query = query.limit(export_request.max_records)
        
        # Apply sorting
        if export_request.filters and hasattr(export_request.filters, 'sort_by'):
            sort_field = getattr(Trace, export_request.filters.sort_by, Trace.timestamp)
            if export_request.filters.sort_order == "asc":
                query = query.order_by(sort_field.asc())
            else:
                query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(Trace.timestamp.desc())
        
        # Execute query
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Convert to dictionaries and organize evaluations
        traces_data = []
        evaluations_map = {}
        
        for trace in traces:
            trace_dict = {
                "id": trace.id,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "user_input": trace.user_input,
                "model_output": trace.model_output,
                "model_name": trace.model_name,
                "system_prompt": trace.system_prompt,
                "session_id": trace.session_id,
                "latency_ms": trace.latency_ms,
                "cost_usd": trace.cost_usd,
                "status": trace.status,
                "trace_metadata": trace.trace_metadata or {}
            }
            
            # Apply field filtering if specified
            if export_request.include_fields:
                trace_dict = {k: v for k, v in trace_dict.items() if k in export_request.include_fields}
            elif export_request.exclude_fields:
                trace_dict = {k: v for k, v in trace_dict.items() if k not in export_request.exclude_fields}
            
            traces_data.append(trace_dict)
            
            # Organize evaluations
            evaluations_map[trace.id] = []
            for evaluation in trace.evaluations:
                eval_dict = {
                    "id": evaluation.id,
                    "trace_id": evaluation.trace_id,
                    "evaluator_type": evaluation.evaluator_type,
                    "evaluator_id": evaluation.evaluator_id,
                    "score": evaluation.score,
                    "label": evaluation.label,
                    "critique": evaluation.critique,
                    "metadata": evaluation.metadata or {},
                    "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None
                }
                evaluations_map[trace.id].append(eval_dict)
        
        # Generate export data
        export_content = ""
        for chunk in generate_export_data(traces_data, evaluations_map, export_request):
            export_content += chunk
        
        # Compress if requested
        if export_request.compress:
            content_bytes = export_content.encode('utf-8')
            compressed_content = gzip.compress(content_bytes)
            file_size = len(compressed_content)
            # In a real implementation, you'd save this to file storage
        else:
            file_size = len(export_content.encode('utf-8'))
        
        # Create export response
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(hours=24)  # 24 hour expiry
        
        export_response = ExportResponse(
            export_id=export_id,
            status="completed",
            format=export_request.format,
            data_type=export_request.data_type,
            total_records=len(traces_data),
            file_size_bytes=file_size,
            download_url=f"/api/export/download/{export_id}",
            created_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                "compressed": export_request.compress,
                "filters_applied": export_request.filters is not None,
                "fine_tuning_format": export_request.fine_tuning_format if export_request.data_type == ExportDataType.FINE_TUNING else None
            }
        )
        
        # Store the export data temporarily (in production, use file storage)
        # For now, we'll return the data directly
        return export_response
        
    except Exception as e:
        logger.error(f"Error during data export: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/export/stream")
async def stream_export_data(
    export_request: ExportRequest,
    current_user: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """Stream export data for large datasets."""
    
    try:
        # Start with basic query
        query = select(Trace).options(selectinload(Trace.evaluations))
        
        # Apply simple filtering if provided (simplified approach for MVP)
        if export_request.filters:
            filters = export_request.filters
            
            # Apply basic filters
            if filters.model_names:
                query = query.where(Trace.model_name.in_(filters.model_names))
            
            if filters.session_ids:
                query = query.where(Trace.session_id.in_(filters.session_ids))
            
            if filters.trace_statuses:
                query = query.where(Trace.status.in_(filters.trace_statuses))
            
            # Apply date range filter
            if filters.trace_date_range:
                if filters.trace_date_range.start_date:
                    query = query.where(Trace.timestamp >= filters.trace_date_range.start_date)
                if filters.trace_date_range.end_date:
                    query = query.where(Trace.timestamp <= filters.trace_date_range.end_date)
            
            # Apply text search
            if filters.search_query:
                search_conditions = []
                if "user_input" in filters.search_in_fields:
                    search_conditions.append(Trace.user_input.ilike(f"%{filters.search_query}%"))
                if "model_output" in filters.search_in_fields:
                    search_conditions.append(Trace.model_output.ilike(f"%{filters.search_query}%"))
                
                if search_conditions:
                    if filters.filter_operator == "OR":
                        query = query.where(or_(*search_conditions))
                    else:
                        query = query.where(and_(*search_conditions))
        
        # Apply max records and sorting
        if export_request.max_records:
            query = query.limit(export_request.max_records)
        
        query = query.order_by(Trace.timestamp.desc())
        
        # Execute query
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Prepare data
        traces_data = []
        evaluations_map = {}
        
        for trace in traces:
            trace_dict = {
                "id": trace.id,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "user_input": trace.user_input,
                "model_output": trace.model_output,
                "model_name": trace.model_name,
                "system_prompt": trace.system_prompt,
                "session_id": trace.session_id,
                "latency_ms": trace.latency_ms,
                "cost_usd": trace.cost_usd,
                "status": trace.status,
                "trace_metadata": trace.trace_metadata or {}
            }
            
            traces_data.append(trace_dict)
            
            evaluations_map[trace.id] = []
            for evaluation in trace.evaluations:
                eval_dict = {
                    "id": evaluation.id,
                    "trace_id": evaluation.trace_id,
                    "evaluator_type": evaluation.evaluator_type,
                    "evaluator_id": evaluation.evaluator_id,
                    "score": evaluation.score,
                    "label": evaluation.label,
                    "critique": evaluation.critique,
                    "metadata": evaluation.metadata or {},
                    "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None
                }
                evaluations_map[trace.id].append(eval_dict)
        
        # Determine content type and filename
        if export_request.format == ExportFormat.JSON:
            media_type = "application/json"
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        elif export_request.format == ExportFormat.CSV:
            media_type = "text/csv"
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        elif export_request.format == ExportFormat.JSONL:
            media_type = "application/x-jsonlines"
            filename = f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        if export_request.compress:
            media_type = "application/gzip"
            filename += ".gz"
        
        # Generate streaming response
        def generate_compressed_stream():
            content = ""
            for chunk in generate_export_data(traces_data, evaluations_map, export_request):
                content += chunk
            
            if export_request.compress:
                compressed = gzip.compress(content.encode('utf-8'))
                yield compressed
            else:
                yield content.encode('utf-8')
        
        return StreamingResponse(
            generate_compressed_stream(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Total-Records": str(len(traces_data))
            }
        )
        
    except Exception as e:
        logger.error(f"Error during streaming export: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming export failed: {str(e)}")


@router.get("/export/download/{export_id}")
async def download_export_file(
    export_id: str,
    current_user: str = Depends(get_current_user_email)
):
    """Download a previously generated export file."""
    
    # In production, this would retrieve the file from storage
    # For now, return a placeholder response
    raise HTTPException(
        status_code=501, 
        detail="File download endpoint not yet implemented. Use /export/stream for immediate downloads."
    )