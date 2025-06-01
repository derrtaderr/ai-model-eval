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
    Advanced search endpoint with multi-dimensional filtering.
    
    Supports filtering by model, status, date ranges, tags, text search, and more.
    """
    try:
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


@router.get("/evaluations/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy(
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current taxonomy of tags for building dynamic filters.
    
    Returns organized tag data for tools, scenarios, topics, and custom tags.
    """
    try:
        # Get tag counts by type and value
        tag_stats_query = select(
            TraceTag.tag_type,
            TraceTag.tag_value,
            func.count(TraceTag.id).label('count'),
            func.avg(TraceTag.confidence_score).label('avg_confidence')
        ).group_by(TraceTag.tag_type, TraceTag.tag_value).order_by(
            TraceTag.tag_type, func.count(TraceTag.id).desc()
        )
        
        result = await db.execute(tag_stats_query)
        tag_data = result.all()
        
        # Get total traces count
        total_traces_query = select(func.count(Trace.id))
        total_result = await db.execute(total_traces_query)
        total_traces = total_result.scalar() or 0
        
        # Organize tags by type
        tools = []
        scenarios = []
        topics = []
        custom_tags = {}
        
        for row in tag_data:
            tag_item = TaxonomyItem(
                tag_type=row.tag_type,
                tag_value=row.tag_value,
                count=row.count,
                confidence_score=float(row.avg_confidence) if row.avg_confidence else None
            )
            
            if row.tag_type == "tool":
                tools.append(tag_item)
            elif row.tag_type == "scenario":
                scenarios.append(tag_item)
            elif row.tag_type == "topic":
                topics.append(tag_item)
            else:
                if row.tag_type not in custom_tags:
                    custom_tags[row.tag_type] = []
                custom_tags[row.tag_type].append(tag_item)
        
        return TaxonomyResponse(
            tools=tools,
            scenarios=scenarios,
            topics=topics,
            custom_tags=custom_tags,
            total_traces=total_traces,
            last_updated=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve taxonomy: {str(e)}"
        )


@router.get("/evaluations/filter-options", response_model=Dict[str, List[str]])
async def get_filter_options(
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get available filter options for dropdown menus.
    
    Returns lists of unique values for various filter fields.
    """
    try:
        filter_options = {}
        
        # Get unique model names
        model_query = select(Trace.model_name).distinct().order_by(Trace.model_name)
        model_result = await db.execute(model_query)
        filter_options["model_names"] = [row[0] for row in model_result.all()]
        
        # Get unique trace statuses
        status_query = select(Trace.status).distinct().order_by(Trace.status)
        status_result = await db.execute(status_query)
        filter_options["trace_statuses"] = [row[0] for row in status_result.all()]
        
        # Get unique evaluation labels
        eval_query = select(Evaluation.label).distinct().where(
            Evaluation.label.is_not(None)
        ).order_by(Evaluation.label)
        eval_result = await db.execute(eval_query)
        filter_options["evaluation_statuses"] = [row[0] for row in eval_result.all()]
        
        # Get unique tag types and values
        tag_query = select(TraceTag.tag_type, TraceTag.tag_value).distinct().order_by(
            TraceTag.tag_type, TraceTag.tag_value
        )
        tag_result = await db.execute(tag_query)
        
        tag_options = {}
        for tag_type, tag_value in tag_result.all():
            if tag_type not in tag_options:
                tag_options[tag_type] = []
            tag_options[tag_type].append(tag_value)
        
        filter_options["tags"] = tag_options
        
        return filter_options
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve filter options: {str(e)}"
        )


@router.get("/evaluations/traces", response_model=List[TraceWithEvaluations])
async def get_traces_for_evaluation(
    limit: int = Query(50, ge=1, le=200, description="Number of traces to return"),
    offset: int = Query(0, ge=0, description="Number of traces to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by evaluation status"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get traces with their evaluation status for the evaluation dashboard.
    
    Returns traces along with their human evaluations, optimized for the evaluation interface.
    This is the legacy simple filtering endpoint - use /search for advanced filtering.
    """
    try:
        # Build base query with evaluations and tags loaded
        query = select(Trace).options(
            selectinload(Trace.evaluations),
            selectinload(Trace.trace_tags)
        )
        
        # Apply filters
        if model_name:
            query = query.where(Trace.model_name == model_name)
        
        # Order by timestamp (newest first)
        query = query.order_by(Trace.timestamp.desc())
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = await db.execute(query)
        traces = result.scalars().all()
        
        # Process traces and determine evaluation status
        traces_with_evaluations = []
        for trace in traces:
            # Determine human evaluation status
            human_evaluations = [e for e in trace.evaluations if e.evaluator_type == "human"]
            
            if not human_evaluations:
                human_eval_status = "pending"
            else:
                # Use the most recent human evaluation
                latest_eval = max(human_evaluations, key=lambda e: e.evaluated_at)
                human_eval_status = latest_eval.label or "pending"
            
            # Filter by evaluation status if requested
            if status_filter and human_eval_status != status_filter.lower():
                continue
            
            # Format evaluations
            formatted_evaluations = []
            for eval in trace.evaluations:
                formatted_evaluations.append(EvaluationResponse(
                    id=str(eval.id),
                    trace_id=str(eval.trace_id),
                    evaluator_type=eval.evaluator_type,
                    evaluator_id=str(eval.evaluator_id) if eval.evaluator_id else None,
                    evaluator_email=None,  # TODO: Load from user relationship
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
            
            traces_with_evaluations.append(TraceWithEvaluations(
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
        
        return traces_with_evaluations
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve traces for evaluation: {str(e)}"
        )


@router.post("/evaluations", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation_data: EvaluationCreate,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a human evaluation for a trace.
    
    This endpoint allows human evaluators to accept, reject, or mark traces for review.
    """
    try:
        # Validate trace exists
        trace_uuid = UUID(evaluation_data.trace_id)
        trace_query = select(Trace).where(Trace.id == trace_uuid)
        result = await db.execute(trace_query)
        trace = result.scalar_one_or_none()
        
        if not trace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trace not found"
            )
        
        # Validate evaluation label
        valid_labels = ["accepted", "rejected", "in_review"]
        if evaluation_data.label not in valid_labels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid label. Must be one of: {valid_labels}"
            )
        
        # For now, we'll use a placeholder user_id since user management isn't fully implemented
        # In production, you'd look up the user by email
        evaluator_id = None  # TODO: Implement user lookup by email
        
        # Check if evaluation already exists for this trace by this evaluator
        existing_eval_query = select(Evaluation).where(
            and_(
                Evaluation.trace_id == trace_uuid,
                Evaluation.evaluator_type == "human",
                Evaluation.evaluator_id == evaluator_id
            )
        )
        result = await db.execute(existing_eval_query)
        existing_eval = result.scalar_one_or_none()
        
        if existing_eval:
            # Update existing evaluation
            existing_eval.score = evaluation_data.score
            existing_eval.label = evaluation_data.label
            existing_eval.critique = evaluation_data.critique
            existing_eval.eval_metadata = evaluation_data.metadata
            existing_eval.evaluated_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(existing_eval)
            
            return {
                "evaluation_id": str(existing_eval.id),
                "message": "Evaluation updated successfully"
            }
        else:
            # Create new evaluation
            new_evaluation = Evaluation(
                trace_id=trace_uuid,
                evaluator_type="human",
                evaluator_id=evaluator_id,
                score=evaluation_data.score,
                label=evaluation_data.label,
                critique=evaluation_data.critique,
                eval_metadata=evaluation_data.metadata,
                evaluated_at=datetime.utcnow()
            )
            
            db.add(new_evaluation)
            await db.commit()
            await db.refresh(new_evaluation)
            
            return {
                "evaluation_id": str(new_evaluation.id),
                "message": "Evaluation created successfully"
            }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create evaluation: {str(e)}"
        )


@router.get("/evaluations/stats", response_model=EvaluationStats)
async def get_evaluation_statistics(
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation statistics for the analytics dashboard.
    
    Returns metrics like evaluation rates, acceptance rates, and agreement data.
    """
    try:
        # Get total traces count
        total_traces_query = select(func.count(Trace.id))
        result = await db.execute(total_traces_query)
        total_traces = result.scalar() or 0
        
        # Get evaluation counts by status
        evaluation_stats_query = select(
            Evaluation.label,
            func.count(Evaluation.id).label('count')
        ).where(
            Evaluation.evaluator_type == "human"
        ).group_by(Evaluation.label)
        
        result = await db.execute(evaluation_stats_query)
        eval_counts = {row.label: row.count for row in result}
        
        # Calculate metrics
        evaluated_traces = sum(eval_counts.values())
        pending_traces = total_traces - evaluated_traces
        accepted_traces = eval_counts.get("accepted", 0)
        rejected_traces = eval_counts.get("rejected", 0)
        in_review_traces = eval_counts.get("in_review", 0)
        
        evaluation_rate = (evaluated_traces / total_traces * 100) if total_traces > 0 else 0
        acceptance_rate = (accepted_traces / evaluated_traces * 100) if evaluated_traces > 0 else 0
        
        # Generate mock agreement data for now (in production, this would compare model vs human evaluations)
        agreement_data = []
        
        return EvaluationStats(
            total_traces=total_traces,
            evaluated_traces=evaluated_traces,
            pending_traces=pending_traces,
            accepted_traces=accepted_traces,
            rejected_traces=rejected_traces,
            in_review_traces=in_review_traces,
            evaluation_rate=round(evaluation_rate, 2),
            acceptance_rate=round(acceptance_rate, 2),
            agreement_data=agreement_data
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evaluation statistics: {str(e)}"
        )


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific evaluation by its ID.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        query = select(Evaluation).where(Evaluation.id == eval_uuid)
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        return EvaluationResponse(
            id=str(evaluation.id),
            trace_id=str(evaluation.trace_id),
            evaluator_type=evaluation.evaluator_type,
            evaluator_id=str(evaluation.evaluator_id) if evaluation.evaluator_id else None,
            evaluator_email=None,  # TODO: Load from user relationship
            score=evaluation.score,
            label=evaluation.label,
            critique=evaluation.critique,
            metadata=evaluation.eval_metadata,
            evaluated_at=evaluation.evaluated_at.isoformat()
        )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evaluation: {str(e)}"
        )


@router.put("/evaluations/{evaluation_id}", response_model=Dict[str, str])
async def update_evaluation(
    evaluation_id: str,
    evaluation_update: EvaluationUpdate,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing evaluation.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        
        # Validate label if provided
        if evaluation_update.label:
            valid_labels = ["accepted", "rejected", "in_review"]
            if evaluation_update.label not in valid_labels:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid label. Must be one of: {valid_labels}"
                )
        
        # Build update dictionary
        update_data = {}
        if evaluation_update.label is not None:
            update_data["label"] = evaluation_update.label
        if evaluation_update.score is not None:
            update_data["score"] = evaluation_update.score
        if evaluation_update.critique is not None:
            update_data["critique"] = evaluation_update.critique
        if evaluation_update.metadata is not None:
            update_data["eval_metadata"] = evaluation_update.metadata
        
        if update_data:
            update_data["evaluated_at"] = datetime.utcnow()
            
            query = update(Evaluation).where(Evaluation.id == eval_uuid).values(**update_data)
            result = await db.execute(query)
            
            if result.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Evaluation not found"
                )
            
            await db.commit()
        
        return {
            "evaluation_id": evaluation_id,
            "message": "Evaluation updated successfully"
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update evaluation: {str(e)}"
        )


@router.delete("/evaluations/{evaluation_id}", response_model=Dict[str, str])
async def delete_evaluation(
    evaluation_id: str,
    current_user_email: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an evaluation.
    """
    try:
        eval_uuid = UUID(evaluation_id)
        
        query = select(Evaluation).where(Evaluation.id == eval_uuid)
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        await db.delete(evaluation)
        await db.commit()
        
        return {
            "evaluation_id": evaluation_id,
            "message": "Evaluation deleted successfully"
        }
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid evaluation ID format"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete evaluation: {str(e)}"
        )


@router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_dynamic_taxonomy(
    force_rebuild: bool = Query(False, description="Force rebuild taxonomy cache"),
    limit: int = Query(1000, description="Max traces to analyze"),
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Get dynamically built taxonomy from trace data.
    Uses LLM analysis to categorize scenarios and extract patterns.
    """
    try:
        # Build taxonomy using the LLM service
        taxonomy_data = await taxonomy_builder.build_taxonomy_from_traces(
            session=session,
            limit=limit,
            force_rebuild=force_rebuild
        )
        
        # Convert to response format
        tools = taxonomy_data.get("tools", [])
        scenarios = taxonomy_data.get("scenarios", [])
        topics = taxonomy_data.get("topics", [])
        performance = taxonomy_data.get("performance", [])
        metadata_categories = taxonomy_data.get("metadata_categories", {})
        
        # Flatten metadata categories into custom_tags
        custom_tags = {}
        for category, items in metadata_categories.items():
            custom_tags[category] = items
        
        return TaxonomyResponse(
            tools=[TaxonomyItem(
                tag_type=item["tag_type"],
                tag_value=item["tag_value"],
                count=item["count"],
                confidence_score=item.get("confidence_score")
            ) for item in tools],
            scenarios=[TaxonomyItem(
                tag_type=item["tag_type"],
                tag_value=item["tag_value"],
                count=item["count"],
                confidence_score=item.get("confidence_score")
            ) for item in scenarios],
            topics=[TaxonomyItem(
                tag_type=item["tag_type"],
                tag_value=item["tag_value"],
                count=item["count"],
                confidence_score=item.get("confidence_score")
            ) for item in topics],
            custom_tags={
                category: [TaxonomyItem(
                    tag_type=item["tag_type"],
                    tag_value=item["tag_value"],
                    count=item["count"],
                    confidence_score=item.get("confidence_score")
                ) for item in items] for category, items in custom_tags.items()
            },
            total_traces=taxonomy_data.get("total_traces", 0),
            last_updated=datetime.fromisoformat(taxonomy_data.get("analysis_timestamp", datetime.utcnow().isoformat()))
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building taxonomy: {str(e)}"
        )


@router.post("/traces/{trace_id}/apply-taxonomy")
async def apply_taxonomy_to_trace(
    trace_id: str,
    force_reanalysis: bool = Query(False, description="Force reanalysis of trace"),
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Apply taxonomy tags to a specific trace using LLM analysis.
    """
    try:
        new_tags = await taxonomy_builder.apply_taxonomy_tags(
            session=session,
            trace_id=trace_id,
            force_reanalysis=force_reanalysis
        )
        
        return {
            "trace_id": trace_id,
            "applied_tags": new_tags,
            "message": f"Applied {len(new_tags)} new tags to trace"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying taxonomy: {str(e)}"
        )


@router.post("/rebuild-taxonomy")
async def rebuild_taxonomy(
    limit: int = Query(1000, description="Max traces to analyze"),
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Force rebuild the taxonomy cache with fresh analysis.
    """
    try:
        taxonomy_data = await taxonomy_builder.build_taxonomy_from_traces(
            session=session,
            limit=limit,
            force_rebuild=True
        )
        
        return {
            "message": "Taxonomy rebuilt successfully",
            "total_traces_analyzed": taxonomy_data.get("total_traces", 0),
            "categories_found": {
                "tools": len(taxonomy_data.get("tools", [])),
                "scenarios": len(taxonomy_data.get("scenarios", [])),
                "topics": len(taxonomy_data.get("topics", [])),
                "performance": len(taxonomy_data.get("performance", [])),
                "metadata_categories": len(taxonomy_data.get("metadata_categories", {}))
            },
            "cache_expires_at": taxonomy_data.get("cache_expires_at")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rebuilding taxonomy: {str(e)}"
        )


@router.post("/filter-presets", response_model=FilterPresetResponse, status_code=status.HTTP_201_CREATED)
async def create_filter_preset(
    preset_data: FilterPresetCreate,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Create a new filter preset for the current user.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # If this is set as default, unset other defaults for this user
        if preset_data.is_default:
            await session.execute(
                update(FilterPreset)
                .where(and_(FilterPreset.user_id == user.id, FilterPreset.is_default == True))
                .values(is_default=False)
            )
        
        # Create new preset
        new_preset = FilterPreset(
            name=preset_data.name,
            description=preset_data.description,
            user_id=user.id,
            filter_config=preset_data.filter_config,
            is_public=preset_data.is_public,
            is_default=preset_data.is_default
        )
        
        session.add(new_preset)
        await session.commit()
        await session.refresh(new_preset)
        
        return FilterPresetResponse(
            id=str(new_preset.id),
            name=new_preset.name,
            description=new_preset.description,
            filter_config=new_preset.filter_config,
            is_public=new_preset.is_public,
            is_default=new_preset.is_default,
            usage_count=new_preset.usage_count,
            created_at=new_preset.created_at,
            updated_at=new_preset.updated_at,
            last_used_at=new_preset.last_used_at,
            user_email=user.email
        )
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating filter preset: {str(e)}")


@router.get("/filter-presets", response_model=FilterPresetsListResponse)
async def get_filter_presets(
    include_public: bool = Query(True, description="Include public presets from other users"),
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Get all filter presets accessible to the current user.
    Includes user's own presets and optionally public presets.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Build query for presets
        if include_public:
            presets_query = (
                select(FilterPreset, User.email.label("user_email"))
                .join(User, FilterPreset.user_id == User.id)
                .where(or_(
                    FilterPreset.user_id == user.id,
                    FilterPreset.is_public == True
                ))
                .order_by(
                    FilterPreset.user_id == user.id,  # User's presets first
                    FilterPreset.is_default.desc(),
                    FilterPreset.usage_count.desc(),
                    FilterPreset.updated_at.desc()
                )
            )
        else:
            presets_query = (
                select(FilterPreset, User.email.label("user_email"))
                .join(User, FilterPreset.user_id == User.id)
                .where(FilterPreset.user_id == user.id)
                .order_by(
                    FilterPreset.is_default.desc(),
                    FilterPreset.usage_count.desc(),
                    FilterPreset.updated_at.desc()
                )
            )
        
        result = await session.execute(presets_query)
        preset_rows = result.all()
        
        # Convert to response format
        presets = []
        user_presets_count = 0
        public_presets_count = 0
        
        for preset, user_email in preset_rows:
            presets.append(FilterPresetResponse(
                id=str(preset.id),
                name=preset.name,
                description=preset.description,
                filter_config=preset.filter_config,
                is_public=preset.is_public,
                is_default=preset.is_default,
                usage_count=preset.usage_count,
                created_at=preset.created_at,
                updated_at=preset.updated_at,
                last_used_at=preset.last_used_at,
                user_email=user_email
            ))
            
            if preset.user_id == user.id:
                user_presets_count += 1
            else:
                public_presets_count += 1
        
        return FilterPresetsListResponse(
            presets=presets,
            total_count=len(presets),
            user_presets_count=user_presets_count,
            public_presets_count=public_presets_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving filter presets: {str(e)}")


@router.get("/filter-presets/{preset_id}", response_model=FilterPresetResponse)
async def get_filter_preset(
    preset_id: str,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Get a specific filter preset by ID.
    User can access their own presets or public presets.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get preset with access control
        preset_query = (
            select(FilterPreset, User.email.label("user_email"))
            .join(User, FilterPreset.user_id == User.id)
            .where(
                and_(
                    FilterPreset.id == preset_id,
                    or_(
                        FilterPreset.user_id == user.id,
                        FilterPreset.is_public == True
                    )
                )
            )
        )
        
        result = await session.execute(preset_query)
        preset_row = result.first()
        
        if not preset_row:
            raise HTTPException(status_code=404, detail="Filter preset not found or access denied")
        
        preset, user_email = preset_row
        
        return FilterPresetResponse(
            id=str(preset.id),
            name=preset.name,
            description=preset.description,
            filter_config=preset.filter_config,
            is_public=preset.is_public,
            is_default=preset.is_default,
            usage_count=preset.usage_count,
            created_at=preset.created_at,
            updated_at=preset.updated_at,
            last_used_at=preset.last_used_at,
            user_email=user_email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving filter preset: {str(e)}")


@router.put("/filter-presets/{preset_id}", response_model=FilterPresetResponse)
async def update_filter_preset(
    preset_id: str,
    preset_data: FilterPresetUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Update a filter preset. Only the owner can update their presets.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get preset (only user's own presets can be updated)
        preset_query = (
            select(FilterPreset)
            .where(and_(
                FilterPreset.id == preset_id,
                FilterPreset.user_id == user.id
            ))
        )
        
        result = await session.execute(preset_query)
        preset = result.scalar_one_or_none()
        
        if not preset:
            raise HTTPException(status_code=404, detail="Filter preset not found or access denied")
        
        # If setting as default, unset other defaults for this user
        if preset_data.is_default:
            await session.execute(
                update(FilterPreset)
                .where(and_(
                    FilterPreset.user_id == user.id, 
                    FilterPreset.is_default == True,
                    FilterPreset.id != preset_id
                ))
                .values(is_default=False)
            )
        
        # Update preset fields
        update_data = {}
        if preset_data.name is not None:
            update_data["name"] = preset_data.name
        if preset_data.description is not None:
            update_data["description"] = preset_data.description
        if preset_data.filter_config is not None:
            update_data["filter_config"] = preset_data.filter_config
        if preset_data.is_public is not None:
            update_data["is_public"] = preset_data.is_public
        if preset_data.is_default is not None:
            update_data["is_default"] = preset_data.is_default
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            await session.execute(
                update(FilterPreset)
                .where(FilterPreset.id == preset_id)
                .values(**update_data)
            )
        
        await session.commit()
        await session.refresh(preset)
        
        return FilterPresetResponse(
            id=str(preset.id),
            name=preset.name,
            description=preset.description,
            filter_config=preset.filter_config,
            is_public=preset.is_public,
            is_default=preset.is_default,
            usage_count=preset.usage_count,
            created_at=preset.created_at,
            updated_at=preset.updated_at,
            last_used_at=preset.last_used_at,
            user_email=user.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating filter preset: {str(e)}")


@router.delete("/filter-presets/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_filter_preset(
    preset_id: str,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Delete a filter preset. Only the owner can delete their presets.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get preset (only user's own presets can be deleted)
        preset_query = (
            select(FilterPreset)
            .where(and_(
                FilterPreset.id == preset_id,
                FilterPreset.user_id == user.id
            ))
        )
        
        result = await session.execute(preset_query)
        preset = result.scalar_one_or_none()
        
        if not preset:
            raise HTTPException(status_code=404, detail="Filter preset not found or access denied")
        
        await session.delete(preset)
        await session.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting filter preset: {str(e)}")


@router.post("/filter-presets/{preset_id}/apply", response_model=FilteredTraceResponse)
async def apply_filter_preset(
    preset_id: str,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Apply a filter preset and return filtered results.
    Also updates the preset's usage statistics.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get preset with access control
        preset_query = (
            select(FilterPreset)
            .where(
                and_(
                    FilterPreset.id == preset_id,
                    or_(
                        FilterPreset.user_id == user.id,
                        FilterPreset.is_public == True
                    )
                )
            )
        )
        
        result = await session.execute(preset_query)
        preset = result.scalar_one_or_none()
        
        if not preset:
            raise HTTPException(status_code=404, detail="Filter preset not found or access denied")
        
        # Update usage statistics
        await session.execute(
            update(FilterPreset)
            .where(FilterPreset.id == preset_id)
            .values(
                usage_count=FilterPreset.usage_count + 1,
                last_used_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        # Apply the filter configuration
        filter_config = preset.filter_config
        
        # Convert preset config to AdvancedFilterRequest
        advanced_filter = AdvancedFilterRequest(**filter_config)
        
        # Use existing search functionality
        return await search_traces_advanced(advanced_filter, current_user, session)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Error applying filter preset: {str(e)}")


@router.get("/filter-presets/user/default", response_model=Optional[FilterPresetResponse])
async def get_user_default_preset(
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Get the user's default filter preset if one exists.
    """
    try:
        # Get user
        user_query = select(User).where(User.email == current_user)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get default preset
        preset_query = (
            select(FilterPreset)
            .where(and_(
                FilterPreset.user_id == user.id,
                FilterPreset.is_default == True
            ))
        )
        
        result = await session.execute(preset_query)
        preset = result.scalar_one_or_none()
        
        if not preset:
            return None
        
        return FilterPresetResponse(
            id=str(preset.id),
            name=preset.name,
            description=preset.description,
            filter_config=preset.filter_config,
            is_public=preset.is_public,
            is_default=preset.is_default,
            usage_count=preset.usage_count,
            created_at=preset.created_at,
            updated_at=preset.updated_at,
            last_used_at=preset.last_used_at,
            user_email=user.email
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving default preset: {str(e)}")


@router.post("/filters/share", response_model=FilterShareResponse)
async def create_filter_share_url(
    share_request: FilterShareRequest,
    request: Request,
    current_user: str = Depends(get_current_user_email)
):
    """
    Create a shareable URL for a filter configuration.
    
    Encodes the filter settings into a compact, URL-safe token that can be
    shared across users and sessions.
    """
    try:
        # Prepare metadata for the share
        metadata = {
            "name": share_request.name,
            "description": share_request.description,
            "created_by": current_user,
            "expires_in_hours": share_request.expires_in_hours
        }
        
        # Encode the filter configuration
        share_token = encode_filter_config(
            filter_config=share_request.filter_config,
            metadata=metadata
        )
        
        # Generate the complete shareable URL
        base_url = f"{request.url.scheme}://{request.url.netloc}/evaluations"
        share_url = generate_share_url(base_url, share_token)
        
        # Calculate expiration
        expires_at = datetime.utcnow() + timedelta(hours=share_request.expires_in_hours)
        
        # Generate filter summary
        filter_summary = extract_filter_summary(share_request.filter_config)
        
        return FilterShareResponse(
            share_token=share_token,
            share_url=share_url,
            expires_at=expires_at,
            filter_summary=filter_summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create share URL: {str(e)}"
        )


@router.get("/filters/shared/{share_token}", response_model=SharedFilterInfo)
async def get_shared_filter_info(
    share_token: str,
    current_user: str = Depends(get_current_user_email)
):
    """
    Get information about a shared filter without applying it.
    
    Decodes the share token and returns metadata about the shared filter
    including expiration status.
    """
    try:
        # Decode the share token
        decoded_data = decode_filter_config(share_token)
        
        # Extract metadata
        metadata = decoded_data.get("metadata", {})
        created_at = datetime.fromisoformat(decoded_data.get("created_at"))
        
        # Calculate expiration
        expires_in_hours = metadata.get("expires_in_hours", 24)
        expires_at = created_at + timedelta(hours=expires_in_hours)
        is_expired = datetime.utcnow() > expires_at
        
        return SharedFilterInfo(
            filter_config=decoded_data.get("filter_config", {}),
            name=metadata.get("name"),
            description=metadata.get("description"),
            created_at=created_at,
            expires_at=expires_at,
            is_expired=is_expired
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decode shared filter: {str(e)}"
        )


@router.post("/filters/shared/{share_token}/apply", response_model=FilteredTraceResponse)
async def apply_shared_filter(
    share_token: str,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Apply a shared filter configuration and return filtered results.
    
    Decodes the share token, validates expiration, and applies the filters
    to return trace data.
    """
    try:
        # Decode the share token
        decoded_data = decode_filter_config(share_token)
        
        # Check expiration
        metadata = decoded_data.get("metadata", {})
        created_at = datetime.fromisoformat(decoded_data.get("created_at"))
        expires_in_hours = metadata.get("expires_in_hours", 24)
        expires_at = created_at + timedelta(hours=expires_in_hours)
        
        if datetime.utcnow() > expires_at:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Shared filter has expired"
            )
        
        # Extract filter configuration
        filter_config = decoded_data.get("filter_config", {})
        
        # Convert to AdvancedFilterRequest
        advanced_filter = AdvancedFilterRequest(**filter_config)
        
        # Apply the filters using existing search functionality
        return await search_traces_advanced(advanced_filter, current_user, session)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply shared filter: {str(e)}"
        )


@router.get("/filters/decode")
async def decode_filter_url_param(
    shared_filter: str = Query(..., description="Encoded filter token from URL parameter"),
    current_user: str = Depends(get_current_user_email)
):
    """
    Decode a filter configuration from a URL parameter.
    
    This endpoint is typically called when a user visits a shared URL
    with a ?shared_filter=<token> parameter.
    """
    try:
        # URL decode the parameter
        decoded_token = unquote(shared_filter)
        
        # Get shared filter info
        shared_info = await get_shared_filter_info(decoded_token, current_user)
        
        return {
            "filter_config": shared_info.filter_config,
            "metadata": {
                "name": shared_info.name,
                "description": shared_info.description,
                "created_at": shared_info.created_at.isoformat(),
                "expires_at": shared_info.expires_at.isoformat(),
                "is_expired": shared_info.is_expired
            },
            "share_token": decoded_token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decode URL parameter: {str(e)}"
        )


@router.post("/filters/encode", response_model=Dict[str, str])
async def encode_filter_for_url(
    filter_config: Dict[str, Any],
    current_user: str = Depends(get_current_user_email)
):
    """
    Encode a filter configuration for URL embedding.
    
    This is a utility endpoint for frontend applications that need to
    generate shareable URLs programmatically.
    """
    try:
        # Create a simple share request
        share_request = FilterShareRequest(
            filter_config=filter_config,
            expires_in_hours=24
        )
        
        # Encode the configuration
        metadata = {"created_by": current_user}
        encoded_token = encode_filter_config(filter_config, metadata)
        
        return {
            "encoded_token": encoded_token,
            "url_parameter": quote(encoded_token, safe=''),
            "message": "Filter configuration encoded successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode filter configuration: {str(e)}"
        )


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


@router.post("/filters/advanced-combinations", response_model=FilteredTraceResponse)
async def execute_advanced_filter_combinations(
    combination_request: AdvancedFilterCombination,
    session: AsyncSession = Depends(get_db),
    current_user: str = Depends(get_current_user_email)
):
    """
    Execute advanced filter combinations with nested AND/OR logic.
    
    Supports complex filter structures with multiple groups and nested conditions.
    """
    try:
        # Initialize query builder
        query_builder = QueryBuilder()
        
        # Build the complex filter condition
        filter_condition = query_builder.build_query_from_combination(combination_request)
        
        # Start with base query
        query = select(Trace).options(
            selectinload(Trace.evaluations),
            selectinload(Trace.trace_tags)
        )
        
        # Count query for total results
        count_query = select(func.count(Trace.id))
        
        # Apply the complex filter condition
        if filter_condition is not None:
            query = query.where(filter_condition)
            count_query = count_query.where(filter_condition)
        
        # Get total count
        count_result = await session.execute(count_query)
        total_filtered_count = count_result.scalar() or 0
        
        # Apply global settings (sorting, pagination, etc.)
        global_settings = combination_request.global_settings or {}
        sort_by = global_settings.get("sort_by", "timestamp")
        sort_order = global_settings.get("sort_order", "desc")
        limit = global_settings.get("limit", 50)
        offset = global_settings.get("offset", 0)
        
        # Apply sorting
        sort_column = getattr(Trace, sort_by, Trace.timestamp)
        if sort_order == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        # Execute query
        result = await session.execute(query)
        traces = result.scalars().all()
        
        # Process traces (reuse existing logic)
        processed_traces = []
        for trace in traces:
            # Determine human evaluation status
            human_evaluations = [e for e in trace.evaluations if e.evaluator_type == "human"]
            
            if not human_evaluations:
                human_eval_status = "pending"
            else:
                latest_eval = max(human_evaluations, key=lambda e: e.evaluated_at)
                human_eval_status = latest_eval.label or "pending"
            
            # Format evaluations
            formatted_evaluations = []
            for eval in trace.evaluations:
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
        
        # Create filter summary
        filter_summary = {
            "total_filter_groups": len(combination_request.filter_groups),
            "root_operator": combination_request.root_operator,
            "complexity_score": _calculate_filter_complexity(combination_request)
        }
        
        # Create pagination info
        pagination = {
            "offset": offset,
            "limit": limit,
            "has_next": len(processed_traces) == limit,
            "has_previous": offset > 0
        }
        
        # Create a mock AdvancedFilterRequest for compatibility
        mock_advanced_filter = AdvancedFilterRequest()
        
        return FilteredTraceResponse(
            traces=processed_traces,
            total_count=total_filtered_count,
            filtered_count=len(processed_traces),
            pagination=pagination,
            filter_summary=filter_summary,
            applied_filters=mock_advanced_filter
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute advanced filter combinations: {str(e)}"
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
    """Validate a filter group and return errors/warnings."""
    result = {"errors": [], "warnings": []}
    
    # Validate individual filters
    for i, filter_dict in enumerate(group.filters):
        try:
            FilterCondition(**filter_dict)
        except Exception as e:
            result["errors"].append(f"{group_path}.filter_{i}: {str(e)}")
    
    # Validate nested groups
    if group.nested_groups:
        for i, nested_group in enumerate(group.nested_groups):
            nested_validation = _validate_filter_group(nested_group, f"{group_path}.nested_{i}")
            result["errors"].extend(nested_validation["errors"])
            result["warnings"].extend(nested_validation["warnings"])
    
    return result 