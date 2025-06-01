"""
Evaluation Prompt Templates Service
Provides pre-built, research-backed evaluation prompt templates for model-based evaluation.
Part of Task 6.2 - Create Pre-built Evaluation Prompt Templates.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from services.evaluator_models import EvaluationCriteria


@dataclass
class TemplateVariable:
    """Represents a variable in an evaluation template."""
    name: str
    description: str
    required: bool = True
    default_value: Optional[str] = None
    validation_pattern: Optional[str] = None

@dataclass
class EvaluationTemplate:
    """Represents a complete evaluation template."""
    id: str
    name: str
    description: str
    criteria: EvaluationCriteria
    template_text: str
    variables: List[TemplateVariable] = field(default_factory=list)
    instructions: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)

class TemplateCategory(str, Enum):
    """Categories for organizing templates."""
    QUALITY_ASSESSMENT = "quality_assessment"
    FACTUAL_VERIFICATION = "factual_verification"
    STYLE_ANALYSIS = "style_analysis"
    SAFETY_EVALUATION = "safety_evaluation"
    TASK_SPECIFIC = "task_specific"

class EvaluationTemplateLibrary:
    """Library of pre-built evaluation templates."""
    
    def __init__(self):
        self.templates: Dict[str, EvaluationTemplate] = {}
        self.categories: Dict[TemplateCategory, List[str]] = {cat: [] for cat in TemplateCategory}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize the library with pre-built templates."""
        
        # Coherence evaluation template
        coherence_template = EvaluationTemplate(
            id="coherence_standard",
            name="Standard Coherence Evaluation",
            description="Evaluates logical flow, consistency, and clarity of AI responses",
            criteria=EvaluationCriteria.COHERENCE,
            template_text="""
You are evaluating the coherence of an AI assistant's response. Coherence refers to how well the response flows logically, maintains internal consistency, and presents ideas clearly.

## Evaluation Context
**User Query:** {user_input}
**AI Response:** {model_output}
{system_prompt_section}

## Evaluation Criteria for Coherence

Assess the response on the following dimensions:

1. **Logical Flow (30%)**: Do ideas progress logically from one to the next?
2. **Internal Consistency (25%)**: Are there any contradictions within the response?
3. **Clarity (25%)**: Is the response clear and easy to understand?
4. **Structure (20%)**: Is the response well-organized with appropriate transitions?

## Scoring Guidelines

- **0.9-1.0 (Excellent)**: Response is exceptionally coherent with perfect logical flow, no contradictions, crystal clear communication, and excellent structure
- **0.7-0.8 (Good)**: Response is coherent with good logical flow, minimal inconsistencies, clear communication, and good structure
- **0.5-0.6 (Fair)**: Response has some coherence issues, moderate logical flow problems, some unclear parts, but generally understandable
- **0.3-0.4 (Poor)**: Response has significant coherence problems, poor logical flow, multiple contradictions, or confusing structure
- **0.0-0.2 (Very Poor)**: Response is incoherent, illogical, contains major contradictions, or is very difficult to understand

## Response Format

Provide your evaluation in this exact JSON format:

```json
{
  "score": 0.85,
  "reasoning": "Detailed explanation of your evaluation covering logical flow, consistency, clarity, and structure. Reference specific parts of the response to support your assessment."
}
```

Focus specifically on coherence - do not penalize for factual errors, style issues, or other criteria unless they directly impact logical coherence.
            """,
            variables=[
                TemplateVariable("user_input", "The user's original query", required=True),
                TemplateVariable("model_output", "The AI assistant's response", required=True),
                TemplateVariable("system_prompt_section", "System prompt context (optional)", required=False, 
                               default_value="")
            ],
            instructions="This template evaluates logical flow and internal consistency of AI responses.",
            examples=[
                {
                    "user_input": "Explain how photosynthesis works",
                    "model_output": "Photosynthesis is the process by which plants convert sunlight into energy...",
                    "expected_score_range": "0.8-0.9"
                }
            ],
            tags=["coherence", "logic", "consistency", "clarity"]
        )
        
        # Relevance evaluation template
        relevance_template = EvaluationTemplate(
            id="relevance_standard",
            name="Standard Relevance Evaluation",
            description="Evaluates how well the AI response addresses the user's specific query",
            criteria=EvaluationCriteria.RELEVANCE,
            template_text="""
You are evaluating the relevance of an AI assistant's response to the user's query. Relevance measures how well the response addresses what the user actually asked for.

## Evaluation Context
**User Query:** {user_input}
**AI Response:** {model_output}
{system_prompt_section}
{reference_answer_section}

## Evaluation Criteria for Relevance

Assess the response on the following dimensions:

1. **Direct Address (40%)**: Does the response directly answer the user's question?
2. **Completeness (30%)**: Does the response cover all aspects of the query?
3. **Focus (20%)**: Does the response stay on-topic without unnecessary tangents?
4. **User Intent (10%)**: Does the response understand and address the user's underlying intent?

## Scoring Guidelines

- **0.9-1.0 (Excellent)**: Response directly and completely addresses the query, stays perfectly on-topic, and demonstrates clear understanding of user intent
- **0.7-0.8 (Good)**: Response addresses the main aspects of the query with good focus and understanding
- **0.5-0.6 (Fair)**: Response partially addresses the query but may miss some aspects or include some irrelevant information
- **0.3-0.4 (Poor)**: Response only tangentially addresses the query or misses key aspects
- **0.0-0.2 (Very Poor)**: Response does not address the user's query or is completely off-topic

## Response Format

Provide your evaluation in this exact JSON format:

```json
{
  "score": 0.75,
  "reasoning": "Detailed explanation of how well the response addresses the user's query. Identify what aspects were covered well and what might be missing or irrelevant."
}
```

Focus specifically on relevance - evaluate only how well the response matches what the user asked for.
            """,
            variables=[
                TemplateVariable("user_input", "The user's original query", required=True),
                TemplateVariable("model_output", "The AI assistant's response", required=True),
                TemplateVariable("system_prompt_section", "System prompt context (optional)", required=False, 
                               default_value=""),
                TemplateVariable("reference_answer_section", "Reference answer for comparison (optional)", required=False,
                               default_value="")
            ],
            instructions="This template evaluates how well the response addresses the user's specific query.",
            examples=[
                {
                    "user_input": "What's the weather like today?",
                    "model_output": "I don't have access to real-time weather data...",
                    "expected_score_range": "0.7-0.8"
                }
            ],
            tags=["relevance", "query_matching", "completeness"]
        )
        
        # Factual accuracy template
        factual_accuracy_template = EvaluationTemplate(
            id="factual_accuracy_standard",
            name="Standard Factual Accuracy Evaluation",
            description="Evaluates the correctness and accuracy of factual claims in AI responses",
            criteria=EvaluationCriteria.FACTUAL_ACCURACY,
            template_text="""
You are evaluating the factual accuracy of an AI assistant's response. Focus on verifiable facts, data, and claims that can be objectively assessed.

## Evaluation Context
**User Query:** {user_input}
**AI Response:** {model_output}
{system_prompt_section}
{reference_answer_section}

## Evaluation Criteria for Factual Accuracy

Assess the response on the following dimensions:

1. **Verifiable Facts (50%)**: Are the specific facts, data, and claims accurate?
2. **Currency (20%)**: Is the information up-to-date when relevant?
3. **Precision (20%)**: Are numerical values, dates, and specific details correct?
4. **Source Reliability (10%)**: Are any implicit sources or common knowledge claims reliable?

## Scoring Guidelines

- **0.9-1.0 (Excellent)**: All factual claims are accurate, current, and precise
- **0.7-0.8 (Good)**: Most factual claims are accurate with minor inaccuracies that don't affect the core message
- **0.5-0.6 (Fair)**: Some factual claims are accurate but there are notable errors or outdated information
- **0.3-0.4 (Poor)**: Many factual claims are inaccurate or misleading
- **0.0-0.2 (Very Poor)**: Most or all factual claims are incorrect

## Special Considerations

- If the response appropriately expresses uncertainty about facts, do not penalize
- Focus on factual claims, not opinions or subjective statements
- Consider the context and level of detail expected for the query
- If no factual claims are made, note this in your reasoning

## Response Format

Provide your evaluation in this exact JSON format:

```json
{
  "score": 0.80,
  "reasoning": "Detailed assessment of factual accuracy. Identify specific facts that are correct or incorrect. Note any uncertainty appropriately expressed by the AI."
}
```

Focus specifically on factual accuracy - do not evaluate other aspects like style or coherence.
            """,
            variables=[
                TemplateVariable("user_input", "The user's original query", required=True),
                TemplateVariable("model_output", "The AI assistant's response", required=True),
                TemplateVariable("system_prompt_section", "System prompt context (optional)", required=False, 
                               default_value=""),
                TemplateVariable("reference_answer_section", "Reference answer for fact-checking (optional)", required=False,
                               default_value="")
            ],
            instructions="This template evaluates the correctness of factual claims and data.",
            examples=[
                {
                    "user_input": "When was the Eiffel Tower built?",
                    "model_output": "The Eiffel Tower was built between 1887 and 1889...",
                    "expected_score_range": "0.9-1.0"
                }
            ],
            tags=["accuracy", "facts", "verification", "data"]
        )
        
        # Grammar and language quality template
        grammar_template = EvaluationTemplate(
            id="grammar_standard",
            name="Standard Grammar and Language Quality Evaluation",
            description="Evaluates grammar, spelling, syntax, and overall language mechanics",
            criteria=EvaluationCriteria.GRAMMAR,
            template_text="""
You are evaluating the grammar and language quality of an AI assistant's response. Focus on mechanics, syntax, and linguistic correctness.

## Evaluation Context
**User Query:** {user_input}
**AI Response:** {model_output}

## Evaluation Criteria for Grammar and Language Quality

Assess the response on the following dimensions:

1. **Grammar (40%)**: Correct use of grammatical rules and structures
2. **Spelling (20%)**: Accurate spelling of words
3. **Syntax (20%)**: Proper sentence structure and word order
4. **Punctuation (10%)**: Correct use of punctuation marks
5. **Language Flow (10%)**: Natural, fluent expression

## Scoring Guidelines

- **0.9-1.0 (Excellent)**: Perfect or near-perfect grammar, spelling, and syntax with natural language flow
- **0.7-0.8 (Good)**: Generally correct with minor errors that don't impede understanding
- **0.5-0.6 (Fair)**: Some grammatical errors or awkward phrasing but still comprehensible
- **0.3-0.4 (Poor)**: Multiple errors that may impede understanding
- **0.0-0.2 (Very Poor)**: Frequent errors that significantly hinder comprehension

## Special Considerations

- Consider the appropriate level of formality for the context
- Account for intentional stylistic choices vs. errors
- Focus on mechanics rather than content or factual accuracy

## Response Format

Provide your evaluation in this exact JSON format:

```json
{
  "score": 0.85,
  "reasoning": "Assessment of grammar, spelling, syntax, and language quality. Note specific errors if present and overall language fluency."
}
```

Focus specifically on language mechanics - do not evaluate content, accuracy, or other criteria.
            """,
            variables=[
                TemplateVariable("user_input", "The user's original query", required=True),
                TemplateVariable("model_output", "The AI assistant's response", required=True)
            ],
            instructions="This template evaluates language mechanics and grammatical correctness.",
            examples=[
                {
                    "user_input": "Can you help me with this problem?",
                    "model_output": "I'd be happy to help you with your problem. Please provide more details...",
                    "expected_score_range": "0.9-1.0"
                }
            ],
            tags=["grammar", "spelling", "syntax", "language"]
        )
        
        # Helpfulness evaluation template
        helpfulness_template = EvaluationTemplate(
            id="helpfulness_standard",
            name="Standard Helpfulness Evaluation",
            description="Evaluates how helpful and actionable the AI response is for the user",
            criteria=EvaluationCriteria.HELPFULNESS,
            template_text="""
You are evaluating the helpfulness of an AI assistant's response. Helpfulness measures how useful and actionable the response is for the user's needs.

## Evaluation Context
**User Query:** {user_input}
**AI Response:** {model_output}
{system_prompt_section}

## Evaluation Criteria for Helpfulness

Assess the response on the following dimensions:

1. **Practical Value (40%)**: Does the response provide practical, actionable information?
2. **Problem Solving (30%)**: Does the response help solve the user's problem or answer their question?
3. **Clarity of Guidance (20%)**: Are any instructions or advice clear and easy to follow?
4. **Additional Value (10%)**: Does the response provide useful context or related information?

## Scoring Guidelines

- **0.9-1.0 (Excellent)**: Extremely helpful with clear, actionable guidance that directly solves the user's problem
- **0.7-0.8 (Good)**: Helpful with good practical value and clear guidance
- **0.5-0.6 (Fair)**: Somewhat helpful but may lack actionable details or complete solution
- **0.3-0.4 (Poor)**: Limited helpfulness with vague or incomplete guidance
- **0.0-0.2 (Very Poor)**: Not helpful or potentially counterproductive

## Special Considerations

- Consider the user's apparent knowledge level and context
- Evaluate appropriate level of detail for the query
- Account for cases where being helpful means expressing uncertainty

## Response Format

Provide your evaluation in this exact JSON format:

```json
{
  "score": 0.80,
  "reasoning": "Assessment of how helpful and actionable the response is. Explain what makes it helpful or what could improve its helpfulness."
}
```

Focus specifically on helpfulness - evaluate the practical utility and actionable value of the response.
            """,
            variables=[
                TemplateVariable("user_input", "The user's original query", required=True),
                TemplateVariable("model_output", "The AI assistant's response", required=True),
                TemplateVariable("system_prompt_section", "System prompt context (optional)", required=False, 
                               default_value="")
            ],
            instructions="This template evaluates the practical utility and actionable value of responses.",
            examples=[
                {
                    "user_input": "How do I fix a leaky faucet?",
                    "model_output": "Here are step-by-step instructions to fix a leaky faucet: 1. Turn off the water supply...",
                    "expected_score_range": "0.8-0.9"
                }
            ],
            tags=["helpfulness", "actionable", "problem_solving", "utility"]
        )
        
        # Add templates to library
        templates = [
            coherence_template,
            relevance_template,
            factual_accuracy_template,
            grammar_template,
            helpfulness_template
        ]
        
        for template in templates:
            self.add_template(template)
    
    def add_template(self, template: EvaluationTemplate):
        """Add a template to the library."""
        self.templates[template.id] = template
        
        # Categorize template
        if template.criteria in [EvaluationCriteria.COHERENCE, EvaluationCriteria.RELEVANCE, EvaluationCriteria.COMPLETENESS]:
            self.categories[TemplateCategory.QUALITY_ASSESSMENT].append(template.id)
        elif template.criteria in [EvaluationCriteria.FACTUAL_ACCURACY, EvaluationCriteria.TRUTHFULNESS]:
            self.categories[TemplateCategory.FACTUAL_VERIFICATION].append(template.id)
        elif template.criteria in [EvaluationCriteria.GRAMMAR, EvaluationCriteria.STYLE]:
            self.categories[TemplateCategory.STYLE_ANALYSIS].append(template.id)
        elif template.criteria in [EvaluationCriteria.HARMFULNESS]:
            self.categories[TemplateCategory.SAFETY_EVALUATION].append(template.id)
        else:
            self.categories[TemplateCategory.TASK_SPECIFIC].append(template.id)
    
    def get_template(self, template_id: str) -> Optional[EvaluationTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_criteria(self, criteria: EvaluationCriteria) -> List[EvaluationTemplate]:
        """Get all templates for a specific evaluation criteria."""
        return [template for template in self.templates.values() if template.criteria == criteria]
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[EvaluationTemplate]:
        """Get all templates in a category."""
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def list_all_templates(self) -> List[EvaluationTemplate]:
        """List all available templates."""
        return list(self.templates.values())
    
    def render_template(self, template_id: str, variables: Dict[str, str]) -> str:
        """Render a template with provided variables."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Validate required variables
        missing_vars = []
        for var in template.variables:
            if var.required and var.name not in variables:
                if var.default_value is None:
                    missing_vars.append(var.name)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Create variables dict with defaults
        render_vars = {}
        for var in template.variables:
            if var.name in variables:
                value = variables[var.name]
                
                # Validate pattern if specified
                if var.validation_pattern and not re.match(var.validation_pattern, value):
                    raise ValueError(f"Variable {var.name} does not match required pattern")
                
                render_vars[var.name] = value
            elif var.default_value is not None:
                render_vars[var.name] = var.default_value
        
        # Handle special formatting for optional sections
        rendered_text = template.template_text
        
        # Process system prompt section
        if 'system_prompt_section' in render_vars:
            if render_vars['system_prompt_section']:
                system_section = f"\n**System Prompt:** {render_vars['system_prompt_section']}"
                rendered_text = rendered_text.replace("{system_prompt_section}", system_section)
            else:
                rendered_text = rendered_text.replace("{system_prompt_section}", "")
        
        # Process reference answer section
        if 'reference_answer_section' in render_vars:
            if render_vars['reference_answer_section']:
                ref_section = f"\n**Reference Answer:** {render_vars['reference_answer_section']}"
                rendered_text = rendered_text.replace("{reference_answer_section}", ref_section)
            else:
                rendered_text = rendered_text.replace("{reference_answer_section}", "")
        
        # Replace all other variables
        for var_name, value in render_vars.items():
            if var_name not in ['system_prompt_section', 'reference_answer_section']:
                rendered_text = rendered_text.replace(f"{{{var_name}}}", str(value))
        
        return rendered_text
    
    def create_custom_template(
        self, 
        name: str, 
        criteria: EvaluationCriteria, 
        template_text: str,
        description: str = "",
        variables: List[TemplateVariable] = None
    ) -> EvaluationTemplate:
        """Create a custom template."""
        template_id = f"custom_{name.lower().replace(' ', '_')}"
        
        template = EvaluationTemplate(
            id=template_id,
            name=name,
            description=description,
            criteria=criteria,
            template_text=template_text,
            variables=variables or [],
            tags=["custom"]
        )
        
        self.add_template(template)
        return template
    
    def validate_template(self, template: EvaluationTemplate) -> Tuple[bool, List[str]]:
        """Validate a template for common issues."""
        errors = []
        
        # Check for required fields
        if not template.name:
            errors.append("Template name is required")
        if not template.template_text:
            errors.append("Template text is required")
        
        # Check for variable placeholders in template text
        placeholders = re.findall(r'\{(\w+)\}', template.template_text)
        variable_names = {var.name for var in template.variables}
        
        for placeholder in placeholders:
            if placeholder not in variable_names:
                errors.append(f"Placeholder '{placeholder}' not defined in variables")
        
        # Check for unused variables
        for var in template.variables:
            if f"{{{var.name}}}" not in template.template_text:
                errors.append(f"Variable '{var.name}' defined but not used in template")
        
        return len(errors) == 0, errors

# Global template library instance
template_library = EvaluationTemplateLibrary() 