"""Core configuration for the Content Transformation System."""

import os
from typing import Dict, Any, List
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):    
    # US.inc API Configuration
    usinc_api_key: str = Field(default_factory=lambda: os.getenv("USINC_API_KEY", os.getenv("ULTRASAFE_API_KEY", "")))
    usinc_base_url: str = Field(default_factory=lambda: os.getenv("USINC_BASE_URL", os.getenv("ULTRASAFE_BASE_URL", "https://api.us.inc/usf/v1")))
    usinc_chat_model: str = Field(default_factory=lambda: os.getenv("USINC_CHAT_MODEL", "usf-mini"))
    usinc_embed_model: str = Field(default_factory=lambda: os.getenv("USINC_EMBED_MODEL", "usf-embed"))
    usinc_rerank_model: str = Field(default_factory=lambda: os.getenv("USINC_RERANK_MODEL", "usf-rerank"))
    
    # Fallback API Keys (for compatibility)
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    
    # Model Configuration
    sentence_transformer_model: str = Field(
        default_factory=lambda: os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    )
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    
    # API Configuration
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    
    # Quality Thresholds
    min_similarity_score: float = Field(
        default_factory=lambda: float(os.getenv("MIN_SIMILARITY_SCORE", "0.7"))
    )
    min_quality_score: float = Field(
        default_factory=lambda: float(os.getenv("MIN_QUALITY_SCORE", "0.8"))
    )
    max_transformation_time: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TRANSFORMATION_TIME", "10"))
    )
    
    # Vector Store Configuration
    collection_name: str = "content_transformation_knowledge"
    vector_size: int = 1024  # Updated to match UltraSafe API embedding dimension
    
    # Supported Styles and Formats
    supported_styles: List[str] = [
        "academic", "conversational", "formal", "informal", 
        "technical", "simplified", "professional", "casual",
        "creative", "factual", "persuasive", "descriptive"
    ]
    
    supported_complexity_levels: List[str] = [
        "elementary", "intermediate", "advanced", "expert"
    ]
    
    supported_formats: List[str] = [
        "article", "report", "email", "social_media", 
        "documentation", "marketing", "blog", "academic_paper"
    ]
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields during validation


# Global configuration instance
config = Config()


class StyleConfig:
    """Configuration for different content styles."""
    
    STYLE_DEFINITIONS = {
        "academic": {
            "description": "Formal, scholarly writing with citations and technical terminology",
            "characteristics": ["formal tone", "third person", "complex sentences", "citations"],
            "vocabulary": "advanced",
            "sentence_structure": "complex"
        },
        "conversational": {
            "description": "Informal, friendly writing as if speaking to a friend",
            "characteristics": ["casual tone", "first/second person", "simple sentences", "contractions"],
            "vocabulary": "everyday",
            "sentence_structure": "simple"
        },
        "formal": {
            "description": "Professional, business-appropriate writing",
            "characteristics": ["respectful tone", "clear structure", "proper grammar"],
            "vocabulary": "professional",
            "sentence_structure": "varied"
        },
        "informal": {
            "description": "Casual but not overly casual writing, friendly and approachable",
            "characteristics": ["relaxed tone", "personal pronouns", "contractions"],
            "vocabulary": "everyday",
            "sentence_structure": "varied"
        },
        "technical": {
            "description": "Precise, specialized writing for technical audiences",
            "characteristics": ["jargon", "specific terminology", "precise language"],
            "vocabulary": "technical",
            "sentence_structure": "complex"
        },
        "simplified": {
            "description": "Easy-to-understand content with clear explanations",
            "characteristics": ["simple words", "short sentences", "explanations of terms"],
            "vocabulary": "basic",
            "sentence_structure": "simple"
        },
        "professional": {
            "description": "Business-appropriate content with a focus on clarity",
            "characteristics": ["clear", "concise", "objective", "logical"],
            "vocabulary": "professional",
            "sentence_structure": "balanced"
        },
        "casual": {
            "description": "Very informal, uses slang and conversational elements",
            "characteristics": ["slang", "idioms", "colloquialisms", "contractions"],
            "vocabulary": "casual",
            "sentence_structure": "relaxed"
        },
        "creative": {
            "description": "Expressive, imaginative writing with figurative language",
            "characteristics": ["metaphors", "vivid descriptions", "unique voice"],
            "vocabulary": "expressive",
            "sentence_structure": "varied"
        },
        "factual": {
            "description": "Objective, fact-based content without opinions",
            "characteristics": ["data-driven", "objective", "precise", "neutral"],
            "vocabulary": "clear",
            "sentence_structure": "structured"
        },
        "persuasive": {
            "description": "Convincing content designed to influence the reader",
            "characteristics": ["rhetorical questions", "evidence", "emotional appeal"],
            "vocabulary": "impactful",
            "sentence_structure": "strategic"
        },
        "descriptive": {
            "description": "Detailed content that creates a vivid picture",
            "characteristics": ["sensory details", "rich adjectives", "show don't tell"],
            "vocabulary": "descriptive",
            "sentence_structure": "flowing"
        }
    }
    
    COMPLEXITY_DEFINITIONS = {
        "elementary": {
            "reading_level": "grades 1-5",
            "vocabulary": "basic everyday words",
            "sentence_length": "short (8-12 words)",
            "concepts": "concrete, familiar"
        },
        "intermediate": {
            "reading_level": "grades 6-9",
            "vocabulary": "common words with some technical terms",
            "sentence_length": "medium (12-18 words)",
            "concepts": "mix of concrete and abstract"
        },
        "advanced": {
            "reading_level": "grades 10-12",
            "vocabulary": "sophisticated with technical terms",
            "sentence_length": "varied (15-25 words)",
            "concepts": "abstract and complex"
        },
        "expert": {
            "reading_level": "college+",
            "vocabulary": "specialized terminology",
            "sentence_length": "complex and varied",
            "concepts": "highly specialized and nuanced"
        }
    }
    
    @classmethod
    def get_style_info(cls, style_name: str) -> Dict[str, Any]:
        """Get the style information for a given style name."""
        return cls.STYLE_DEFINITIONS.get(style_name.lower(), {})
    
    @classmethod
    def get_all_styles(cls) -> List[str]:
        """Get all supported style names."""
        return list(cls.STYLE_DEFINITIONS.keys())


class QualityMetrics:
    """Configuration for quality assessment metrics."""
    
    METRICS = {
        "readability": {
            "flesch_kincaid": {"weight": 0.3, "threshold": 0.0},
            "smog": {"weight": 0.2, "threshold": 0.0},
            "automated_readability": {"weight": 0.2, "threshold": 0.0},
            "coleman_liau": {"weight": 0.3, "threshold": 0.0}
        },
        "semantic_similarity": {
            "sentence_bert": {"weight": 0.4, "threshold": 0.7},
            "cosine_similarity": {"weight": 0.3, "threshold": 0.6},
            "rouge_l": {"weight": 0.3, "threshold": 0.5}
        },
        "style_adherence": {
            "vocabulary_match": {"weight": 0.4, "threshold": 0.7},
            "sentence_structure": {"weight": 0.3, "threshold": 0.6},
            "tone_consistency": {"weight": 0.3, "threshold": 0.7}
        },
        "factual_accuracy": {
            "entity_preservation": {"weight": 0.5, "threshold": 0.9},
            "fact_verification": {"weight": 0.5, "threshold": 0.85}
        }
    }
