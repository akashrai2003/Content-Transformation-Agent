"""Base models and schemas for the Content Transformation System."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class ContentStyle(str, Enum):
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    CREATIVE = "creative"
    FACTUAL = "factual"
    PERSUASIVE = "persuasive"
    DESCRIPTIVE = "descriptive"


class ComplexityLevel(str, Enum):
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentFormat(str, Enum):
    PARAGRAPH = "paragraph"
    ARTICLE = "article"
    REPORT = "report"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    DOCUMENTATION = "documentation"
    MARKETING = "marketing"
    BLOG = "blog"
    ACADEMIC_PAPER = "academic_paper"
    ESSAY = "essay"
    BULLET_POINTS = "bullet_points"
    LIST = "list"


class TransformationStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    TRANSFORMING = "transforming"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentAnalysis(BaseModel):
    style: str = Field(..., description="Detected content style")
    complexity_level: str = Field(..., description="Detected complexity level")
    format_type: str = Field(..., description="Detected content format")
    word_count: int = Field(..., description="Number of words")
    sentence_count: int = Field(..., description="Number of sentences")
    paragraph_count: int = Field(..., description="Number of paragraphs")
    
    readability_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Various readability metrics"
    )
    
    key_entities: List[str] = Field(
        default_factory=list,
        description="Important entities found in content"
    )
    
    main_topics: List[str] = Field(
        default_factory=list,
        description="Main topics or themes"
    )
    
    sentiment_score: float = Field(
        default=0.0,
        description="Sentiment polarity (-1 to 1)"
    )
    
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in analysis (0 to 1)"
    )


class TransformationPlan(BaseModel):
    source_analysis: ContentAnalysis = Field(..., description="Analysis of source content")
    target_style: ContentStyle = Field(..., description="Target style")
    target_complexity: ComplexityLevel = Field(..., description="Target complexity")
    target_format: ContentFormat = Field(..., description="Target format")
    
    transformation_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered steps for transformation"
    )
    
    preservation_requirements: List[str] = Field(
        default_factory=list,
        description="Elements that must be preserved"
    )
    
    quality_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Target quality metrics"
    )
    
    estimated_difficulty: float = Field(
        default=0.5,
        description="Estimated transformation difficulty (0 to 1)"
    )
    
    similar_examples: List[str] = Field(
        default_factory=list,
        description="IDs of similar transformation examples"
    )


class QualityAssessment(BaseModel):
    overall_score: float = Field(..., description="Overall quality score (0 to 1)")
    
    readability_score: float = Field(
        default=0.0,
        description="Readability assessment score"
    )
    
    similarity_score: float = Field(
        default=0.0,
        description="Semantic similarity to original"
    )
    
    style_adherence: float = Field(
        default=0.0,
        description="Adherence to target style"
    )
    
    complexity_match: float = Field(
        default=0.0,
        description="Match to target complexity"
    )
    
    fact_preservation: float = Field(
        default=0.0,
        description="Preservation of factual content"
    )
    
    detailed_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed quality metrics"
    )
    
    quality_issues: List[str] = Field(
        default_factory=list,
        description="Identified quality issues"
    )
    
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )
    
    assessment_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When assessment was performed"
    )


class TransformationRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    content: str = Field(..., description="Original content to transform")
    target_style: ContentStyle = Field(..., description="Target style")
    target_complexity: ComplexityLevel = Field(..., description="Target complexity")
    target_format: ContentFormat = Field(..., description="Target format")
    
    user_instructions: Optional[str] = Field(
        default=None,
        description="Additional user instructions"
    )
    
    preserve_entities: bool = Field(
        default=True,
        description="Whether to preserve named entities"
    )
    
    preserve_facts: bool = Field(
        default=True,
        description="Whether to preserve factual information"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request timestamp"
    )


class TransformationResult(BaseModel):
    request_id: str = Field(..., description="Original request ID")
    original_content: str = Field(..., description="Original content")
    transformed_content: str = Field(..., description="Transformed content")
    
    source_analysis: Optional[ContentAnalysis] = Field(
        default=None,
        description="Analysis of source content"
    )
    
    transformation_plan: Optional[TransformationPlan] = Field(
        default=None,
        description="Transformation plan used"
    )
    
    quality_assessment: Optional[QualityAssessment] = Field(
        default=None,
        description="Quality assessment results"
    )
    
    status: TransformationStatus = Field(..., description="Transformation status")
    
    agent_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from individual agents"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )
    
    processing_time: float = Field(
        default=0.0,
        description="Processing time in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Result timestamp"
    )


class UserFeedback(BaseModel):
    transformation_id: str = Field(..., description="ID of the transformation")
    overall_rating: int = Field(..., description="Overall rating (1-5)")
    
    accuracy_rating: int = Field(
        default=3,
        description="Accuracy preservation rating (1-5)"
    )
    
    style_rating: int = Field(
        default=3,
        description="Style transformation rating (1-5)"
    )
    
    readability_rating: int = Field(
        default=3,
        description="Readability rating (1-5)"
    )
    
    comments: Optional[str] = Field(
        default=None,
        description="User comments"
    )
    
    suggested_improvements: Optional[str] = Field(
        default=None,
        description="User suggestions for improvement"
    )
    
    would_use_again: bool = Field(
        default=True,
        description="Whether user would use the service again"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Feedback timestamp"
    )

    @validator('overall_rating', 'accuracy_rating', 'style_rating', 'readability_rating')
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


class StyleExample(BaseModel):
    style: ContentStyle = Field(..., description="Content style")
    complexity: ComplexityLevel = Field(..., description="Complexity level")
    format_type: ContentFormat = Field(..., description="Content format")
    content: str = Field(..., description="Example content")
    description: str = Field(..., description="Description of the example")
    
    key_features: List[str] = Field(
        default_factory=list,
        description="Key stylistic features"
    )
    
    author: Optional[str] = Field(
        default=None,
        description="Author of the example"
    )
    
    source: Optional[str] = Field(
        default=None,
        description="Source of the example"
    )
    
    quality_score: float = Field(
        default=1.0,
        description="Quality score of the example"
    )


class AgentState(BaseModel):
    request_id: str = Field(..., description="Request identifier")
    original_content: str = Field(..., description="Original content")
    target_style: ContentStyle = Field(..., description="Target style")
    target_complexity: ComplexityLevel = Field(..., description="Target complexity")
    target_format: ContentFormat = Field(..., description="Target format")
    
    user_instructions: Optional[str] = Field(
        default=None,
        description="User instructions"
    )
    
    preserve_entities: bool = Field(
        default=True,
        description="Preserve named entities"
    )
    
    preserve_facts: bool = Field(
        default=True,
        description="Preserve facts"
    )
    
    current_step: str = Field(
        default="initialized",
        description="Current processing step"
    )
    
    status: TransformationStatus = Field(
        default=TransformationStatus.PENDING,
        description="Current status"
    )
    
    source_analysis: Optional[ContentAnalysis] = Field(
        default=None,
        description="Source content analysis"
    )
    
    transformation_plan: Optional[TransformationPlan] = Field(
        default=None,
        description="Transformation plan"
    )
    
    transformed_content: Optional[str] = Field(
        default=None,
        description="Transformed content"
    )
    
    quality_assessment: Optional[QualityAssessment] = Field(
        default=None,
        description="Quality assessment"
    )
    
    agent_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent outputs"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Processing errors"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retries"
    )

    class Config:
        arbitrary_types_allowed = True
