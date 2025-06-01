"""
LLM-powered taxonomy builder service.
Automatically generates taxonomies from trace data using AI analysis.
Part of Task 5 - Advanced Filtering & Taxonomy System.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from database.models import Trace, TraceTag
from database.connection import get_db
from config.settings import get_settings


class TaxonomyBuilder:
    """
    LLM-powered service for building dynamic taxonomies from trace data.
    Analyzes patterns in user inputs, model outputs, and metadata to automatically
    categorize traces and build filtering taxonomies.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_available = openai_available and self.settings.openai_api_key
        if self.openai_available:
            openai.api_key = self.settings.openai_api_key
        self.cache_duration_hours = 24
        self._taxonomy_cache = {}
        self._last_build_time = None
        
    async def build_taxonomy_from_traces(
        self, 
        session: AsyncSession, 
        limit: int = 1000,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build a comprehensive taxonomy from existing trace data.
        
        Args:
            session: Database session
            limit: Maximum number of traces to analyze
            force_rebuild: Force rebuild even if cache is valid
            
        Returns:
            Dictionary containing categorized taxonomy
        """
        
        # Check if we need to rebuild
        if not force_rebuild and self._is_cache_valid():
            return self._taxonomy_cache
            
        # Get recent traces for analysis
        traces_query = (
            select(Trace)
            .options(selectinload(Trace.trace_tags))
            .order_by(Trace.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(traces_query)
        traces = result.scalars().all()
        
        if not traces:
            return self._get_empty_taxonomy()
            
        # Build taxonomy using multiple approaches
        taxonomy = await self._build_comprehensive_taxonomy(traces)
        
        # Cache the results
        self._taxonomy_cache = taxonomy
        self._last_build_time = datetime.utcnow()
        
        return taxonomy
    
    async def _build_comprehensive_taxonomy(self, traces: List[Trace]) -> Dict[str, Any]:
        """Build taxonomy using multiple analysis approaches."""
        
        # Prepare trace data for analysis
        trace_data = self._prepare_trace_data(traces)
        
        # Run multiple analysis approaches in parallel
        tasks = [
            self._analyze_tools_and_functions(trace_data),
            self._analyze_scenarios_with_llm(trace_data) if self.openai_available else self._analyze_scenarios_basic(trace_data),
            self._analyze_topics_and_domains(trace_data),
            self._analyze_performance_patterns(trace_data),
            self._extract_metadata_categories(trace_data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        taxonomy = {
            "tools": results[0] if not isinstance(results[0], Exception) else [],
            "scenarios": results[1] if not isinstance(results[1], Exception) else [],
            "topics": results[2] if not isinstance(results[2], Exception) else [],
            "performance": results[3] if not isinstance(results[3], Exception) else [],
            "metadata_categories": results[4] if not isinstance(results[4], Exception) else {},
            "total_traces": len(traces),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "cache_expires_at": (datetime.utcnow() + timedelta(hours=self.cache_duration_hours)).isoformat()
        }
        
        return taxonomy
    
    def _prepare_trace_data(self, traces: List[Trace]) -> List[Dict[str, Any]]:
        """Prepare trace data for analysis."""
        prepared_data = []
        
        for trace in traces:
            trace_dict = {
                "id": trace.id,
                "user_input": trace.user_input,
                "model_output": trace.model_output,
                "model_name": trace.model_name,
                "system_prompt": trace.system_prompt,
                "metadata": trace.metadata or {},
                "latency_ms": trace.latency_ms,
                "cost_usd": trace.cost_usd,
                "tags": [{"type": tag.tag_type, "value": tag.tag_value} for tag in trace.trace_tags] if trace.trace_tags else []
            }
            prepared_data.append(trace_dict)
            
        return prepared_data
    
    async def _analyze_tools_and_functions(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze traces to identify tools and functions used."""
        tool_counter = Counter()
        function_counter = Counter()
        
        for trace in trace_data:
            # Extract from metadata
            metadata = trace.get("metadata", {})
            
            # Look for tool usage patterns
            if "tools_used" in metadata:
                for tool in metadata["tools_used"]:
                    tool_counter[tool] += 1
                    
            if "functions" in metadata:
                for func in metadata["functions"]:
                    if isinstance(func, dict) and "name" in func:
                        function_counter[func["name"]] += 1
                    elif isinstance(func, str):
                        function_counter[func] += 1
            
            # Analyze model names for provider/tool patterns
            model_name = trace.get("model_name", "")
            if model_name:
                # Extract provider (e.g., "gpt-4" -> "openai")
                provider = self._extract_provider_from_model(model_name)
                if provider:
                    tool_counter[f"provider:{provider}"] += 1
            
            # Look for function call patterns in user input and output
            user_input = trace.get("user_input", "")
            model_output = trace.get("model_output", "")
            
            # Extract function calls from text (basic pattern matching)
            function_patterns = re.findall(r'(\w+)\s*\([^)]*\)', user_input + " " + model_output)
            for func in function_patterns:
                if len(func) > 2 and func.lower() not in ['the', 'and', 'for', 'are', 'can']:
                    function_counter[f"detected:{func}"] += 1
        
        # Convert to taxonomy format
        tools_taxonomy = []
        for tool, count in tool_counter.most_common(50):
            tools_taxonomy.append({
                "tag_type": "tool",
                "tag_value": tool,
                "count": count,
                "confidence_score": min(0.9, count / len(trace_data))
            })
            
        for func, count in function_counter.most_common(50):
            tools_taxonomy.append({
                "tag_type": "function",
                "tag_value": func,
                "count": count,
                "confidence_score": min(0.9, count / len(trace_data))
            })
        
        return tools_taxonomy
    
    async def _analyze_scenarios_basic(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic scenario analysis without LLM (fallback when OpenAI is not available)."""
        scenario_patterns = {
            "question_answering": ["what", "how", "why", "when", "where", "?"],
            "code_generation": ["code", "function", "class", "def ", "import", "return"],
            "text_generation": ["write", "generate", "create", "compose"],
            "analysis": ["analyze", "compare", "evaluate", "review", "assess"],
            "translation": ["translate", "español", "français", "deutsch", "中文"],
            "summarization": ["summarize", "summary", "tldr", "brief"],
            "conversation": ["hello", "hi", "thanks", "please", "sorry"],
            "calculation": ["calculate", "compute", "+", "-", "*", "/", "="]
        }
        
        scenario_counter = Counter()
        
        for trace in trace_data:
            text_content = f"{trace.get('user_input', '')} {trace.get('model_output', '')}".lower()
            
            for scenario, keywords in scenario_patterns.items():
                if any(keyword in text_content for keyword in keywords):
                    scenario_counter[scenario] += 1
        
        # Convert to taxonomy format
        scenarios_taxonomy = []
        for scenario, count in scenario_counter.items():
            if count > 0:
                scenarios_taxonomy.append({
                    "tag_type": "scenario",
                    "tag_value": scenario,
                    "count": count,
                    "confidence_score": min(0.8, count / len(trace_data)),
                    "description": f"Basic pattern detection for {scenario}",
                    "patterns": scenario_patterns[scenario][:3]  # Show first 3 patterns
                })
        
        return scenarios_taxonomy
    
    async def _analyze_scenarios_with_llm(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to analyze and categorize scenarios from trace data."""
        if not self.openai_available:
            return await self._analyze_scenarios_basic(trace_data)
            
        # Sample traces for LLM analysis (to manage costs)
        sample_size = min(20, len(trace_data))  # Reduced for older API
        sample_traces = trace_data[:sample_size]
        
        # Prepare data for LLM analysis
        analysis_data = []
        for trace in sample_traces:
            analysis_data.append({
                "user_input": trace["user_input"][:300],  # Reduced for token efficiency
                "model_output": trace["model_output"][:300],
                "model_name": trace["model_name"]
            })
        
        prompt = self._build_scenario_analysis_prompt(analysis_data)
        
        try:
            # Use the older OpenAI API format
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing LLM conversation patterns and categorizing them into scenarios."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
            )
            
            # Parse LLM response
            scenarios = self._parse_scenario_response(response.choices[0].message.content)
            return scenarios
            
        except Exception as e:
            print(f"Error in LLM scenario analysis: {e}")
            return await self._analyze_scenarios_basic(trace_data)
    
    def _build_scenario_analysis_prompt(self, analysis_data: List[Dict[str, Any]]) -> str:
        """Build prompt for LLM scenario analysis."""
        
        prompt = """Analyze the following LLM conversation samples and identify common scenarios/use cases.

Please categorize these conversations into scenarios and provide a JSON response with the following format:
```json
{
  "scenarios": [
    {
      "tag_value": "scenario_name",
      "description": "Brief description",
      "patterns": ["pattern1", "pattern2"],
      "confidence_score": 0.8,
      "count": 5
    }
  ]
}
```

Conversation samples:
"""
        
        for i, trace in enumerate(analysis_data[:10]):  # Further reduced for older API
            prompt += f"\n--- Sample {i+1} ---\n"
            prompt += f"User: {trace['user_input'][:150]}\n"
            prompt += f"Assistant: {trace['model_output'][:150]}\n"
            prompt += f"Model: {trace['model_name']}\n"
        
        prompt += "\n\nPlease identify 3-5 most common scenarios from these samples."
        
        return prompt
    
    def _parse_scenario_response(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for scenario data."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                scenarios = data.get("scenarios", [])
                
                # Convert to taxonomy format
                taxonomy_scenarios = []
                for scenario in scenarios:
                    taxonomy_scenarios.append({
                        "tag_type": "scenario",
                        "tag_value": scenario.get("tag_value", "unknown"),
                        "count": scenario.get("count", 1),
                        "confidence_score": scenario.get("confidence_score", 0.5),
                        "description": scenario.get("description", ""),
                        "patterns": scenario.get("patterns", [])
                    })
                
                return taxonomy_scenarios
                
        except Exception as e:
            print(f"Error parsing scenario response: {e}")
            
        return []
    
    async def _analyze_topics_and_domains(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze traces to identify topics and domains."""
        topic_keywords = defaultdict(int)
        domain_indicators = defaultdict(int)
        
        # Common domain/topic keywords to look for
        domain_patterns = {
            "coding": ["code", "function", "programming", "debug", "error", "syntax"],
            "writing": ["write", "draft", "edit", "content", "article", "blog"],
            "analysis": ["analyze", "data", "statistics", "report", "insights"],
            "creative": ["creative", "story", "poem", "art", "design", "imagine"],
            "business": ["business", "strategy", "market", "finance", "revenue"],
            "education": ["learn", "teach", "explain", "tutorial", "lesson"],
            "research": ["research", "study", "investigate", "academic", "paper"],
            "support": ["help", "assist", "support", "question", "problem"]
        }
        
        for trace in trace_data:
            text_content = f"{trace.get('user_input', '')} {trace.get('model_output', '')}".lower()
            
            # Check domain patterns
            for domain, keywords in domain_patterns.items():
                if any(keyword in text_content for keyword in keywords):
                    domain_indicators[domain] += 1
            
            # Extract significant words (basic keyword extraction)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text_content)
            for word in words:
                if len(word) > 4 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'will']:
                    topic_keywords[word] += 1
        
        # Build taxonomy
        topics_taxonomy = []
        
        # Add domain categories
        for domain, count in domain_indicators.items():
            if count > 0:
                topics_taxonomy.append({
                    "tag_type": "domain",
                    "tag_value": domain,
                    "count": count,
                    "confidence_score": min(0.9, count / len(trace_data))
                })
        
        # Add top keywords as topics
        for keyword, count in topic_keywords.most_common(30):
            if count > 1:  # Only include keywords that appear multiple times
                topics_taxonomy.append({
                    "tag_type": "topic",
                    "tag_value": keyword,
                    "count": count,
                    "confidence_score": min(0.8, count / len(trace_data))
                })
        
        return topics_taxonomy
    
    async def _analyze_performance_patterns(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze performance patterns for categorization."""
        latencies = [trace.get("latency_ms", 0) for trace in trace_data if trace.get("latency_ms")]
        costs = [trace.get("cost_usd", 0) for trace in trace_data if trace.get("cost_usd")]
        
        performance_taxonomy = []
        
        if latencies:
            # Categorize by latency
            latency_categories = {
                "fast": (0, 1000),
                "medium": (1000, 5000),
                "slow": (5000, float('inf'))
            }
            
            for category, (min_lat, max_lat) in latency_categories.items():
                count = sum(1 for lat in latencies if min_lat <= lat < max_lat)
                if count > 0:
                    performance_taxonomy.append({
                        "tag_type": "latency",
                        "tag_value": category,
                        "count": count,
                        "confidence_score": 1.0
                    })
        
        if costs:
            # Categorize by cost
            cost_categories = {
                "low_cost": (0, 0.01),
                "medium_cost": (0.01, 0.1),
                "high_cost": (0.1, float('inf'))
            }
            
            for category, (min_cost, max_cost) in cost_categories.items():
                count = sum(1 for cost in costs if min_cost <= cost < max_cost)
                if count > 0:
                    performance_taxonomy.append({
                        "tag_type": "cost",
                        "tag_value": category,
                        "count": count,
                        "confidence_score": 1.0
                    })
        
        return performance_taxonomy
    
    async def _extract_metadata_categories(self, trace_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract categories from trace metadata."""
        metadata_categories = defaultdict(lambda: defaultdict(int))
        
        for trace in trace_data:
            metadata = trace.get("metadata", {})
            
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_categories[key][str(value)] += 1
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (str, int, float)):
                            metadata_categories[key][str(item)] += 1
        
        # Convert to taxonomy format
        categories = {}
        for category, values in metadata_categories.items():
            categories[category] = []
            for value, count in values.items():
                categories[category].append({
                    "tag_type": f"metadata_{category}",
                    "tag_value": value,
                    "count": count,
                    "confidence_score": min(0.9, count / len(trace_data))
                })
        
        return categories
    
    def _extract_provider_from_model(self, model_name: str) -> Optional[str]:
        """Extract provider from model name."""
        model_name_lower = model_name.lower()
        
        providers = {
            "openai": ["gpt", "text-davinci", "text-curie", "text-babbage", "text-ada"],
            "anthropic": ["claude"],
            "google": ["bard", "palm", "gemini"],
            "meta": ["llama"],
            "mistral": ["mistral"],
            "cohere": ["command"],
            "huggingface": ["huggingface"]
        }
        
        for provider, patterns in providers.items():
            if any(pattern in model_name_lower for pattern in patterns):
                return provider
                
        return None
    
    def _is_cache_valid(self) -> bool:
        """Check if cached taxonomy is still valid."""
        if not self._last_build_time or not self._taxonomy_cache:
            return False
            
        cache_age = datetime.utcnow() - self._last_build_time
        return cache_age.total_seconds() < (self.cache_duration_hours * 3600)
    
    def _get_empty_taxonomy(self) -> Dict[str, Any]:
        """Return empty taxonomy structure."""
        return {
            "tools": [],
            "scenarios": [],
            "topics": [],
            "performance": [],
            "metadata_categories": {},
            "total_traces": 0,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "cache_expires_at": (datetime.utcnow() + timedelta(hours=self.cache_duration_hours)).isoformat()
        }

    async def apply_taxonomy_tags(self, session: AsyncSession, trace_id: str, force_reanalysis: bool = False) -> List[Dict[str, Any]]:
        """Apply taxonomy tags to a specific trace."""
        # Get the trace
        trace_query = select(Trace).where(Trace.id == trace_id).options(selectinload(Trace.trace_tags))
        result = await session.execute(trace_query)
        trace = result.scalar_one_or_none()
        
        if not trace:
            return []
        
        # Check if we need to reanalyze
        existing_tags = [{"type": tag.tag_type, "value": tag.tag_value} for tag in trace.trace_tags]
        if existing_tags and not force_reanalysis:
            return existing_tags
        
        # Build taxonomy for this single trace
        trace_data = self._prepare_trace_data([trace])
        
        # Run analysis
        tools = await self._analyze_tools_and_functions(trace_data)
        topics = await self._analyze_topics_and_domains(trace_data)
        performance = await self._analyze_performance_patterns(trace_data)
        
        # Combine and filter high-confidence tags
        all_tags = tools + topics + performance
        high_confidence_tags = [tag for tag in all_tags if tag.get("confidence_score", 0) > 0.3]
        
        # Apply tags to database
        new_tags = []
        for tag_data in high_confidence_tags:
            # Check if tag already exists
            existing_tag = next(
                (tag for tag in trace.trace_tags 
                 if tag.tag_type == tag_data["tag_type"] and tag.tag_value == tag_data["tag_value"]),
                None
            )
            
            if not existing_tag:
                new_tag = TraceTag(
                    trace_id=trace.id,
                    tag_type=tag_data["tag_type"],
                    tag_value=tag_data["tag_value"],
                    confidence_score=tag_data.get("confidence_score", 0.5)
                )
                session.add(new_tag)
                new_tags.append({
                    "type": new_tag.tag_type,
                    "value": new_tag.tag_value,
                    "confidence": new_tag.confidence_score
                })
        
        await session.commit()
        return new_tags


# Global instance
taxonomy_builder = TaxonomyBuilder() 