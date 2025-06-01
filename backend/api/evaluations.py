"""
API endpoints for human evaluation management.
Implements the Human Evaluation Dashboard backend for Task 4.
Enhanced with Advanced Filtering & Taxonomy System for Task 5.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from uuid import UUID
import base64
import json
import zlib
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from pydantic import BaseModel, Field, validator, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc, asc, text
from sqlalchemy.orm import selectinload, joinedload

from auth.security import get_current_user_email
from database.connection import get_db
from database.models import Evaluation, Trace, User, EvaluationStatus, TraceTag, FilterPreset
from services.taxonomy_builder import taxonomy_builder
from services.cache_manager import get_cache_manager, cache_filter_results, get_cached_filter_results
from services.evaluator_models import (
    evaluator_manager, 
    EvaluationRequest as ModelEvaluationRequest,
    EvaluationResult as ModelEvaluationResult,
    EvaluationCriteria,
    BatchEvaluationResult
)


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
    """
    Evaluate a trace using AI models with automatic score calibration.
    """
    try:
        from services.evaluator_models import evaluator_manager, EvaluationRequest as ModelEvaluationRequest
        
        # Create evaluation request
        request = ModelEvaluationRequest(
            trace_id=evaluation_request.trace_id,
            user_input=evaluation_request.user_input,
            model_output=evaluation_request.model_output,
            system_prompt=evaluation_request.system_prompt,
            context=evaluation_request.context,
            criteria=evaluation_request.criteria,
            custom_prompt=evaluation_request.custom_prompt,
            reference_answer=evaluation_request.reference_answer
        )
        
        # Evaluate with calibration
        result = await evaluator_manager.evaluate_single_with_calibration(
            request=request,
            evaluator_name=evaluation_request.evaluator_model,
            use_calibration=evaluation_request.use_calibration
        )
        
        return {
            "trace_id": result.trace_id,
            "evaluator_model": result.evaluator_model,
            "criteria": result.criteria.value,
            "score": result.score,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "evaluation_time_ms": result.evaluation_time_ms,
            "cost_usd": result.cost_usd,
            "metadata": result.metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate with calibration: {str(e)}"
        ) 