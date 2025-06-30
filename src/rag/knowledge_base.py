"""Knowledge base management for content transformation system."""

import logging
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
from datetime import datetime

from src.rag.vector_st_new import VectorStore
from ..core.models import StyleExample, ContentStyle, ComplexityLevel, ContentFormat
from ..core.config_new import config

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self):
        self.vector_store = VectorStore()
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return
        
        try:
            await self._load_default_style_examples()
            await self._load_transformation_patterns()
            
            self.initialized = True
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def _load_default_style_examples(self):
        
        academic_examples = [
            {
                "style": ContentStyle.ACADEMIC,
                "complexity": ComplexityLevel.ADVANCED,
                "format_type": ContentFormat.ACADEMIC_PAPER,
                "content": "The phenomenon of climate change represents a significant challenge to contemporary environmental policy. Recent empirical studies have demonstrated that anthropogenic factors contribute substantially to global temperature variations (Smith et al., 2023). This research examines the correlation between industrial emissions and atmospheric carbon dioxide concentrations, utilizing a longitudinal dataset spanning three decades.",
                "description": "Academic writing sample with formal tone and citations",
                "key_features": ["formal tone", "third person", "citations", "technical terminology", "complex sentences"],
                "author": "System",
                "source": "Generated",
                "quality_score": 0.95
            },
            {
                "style": ContentStyle.ACADEMIC,
                "complexity": ComplexityLevel.INTERMEDIATE,
                "format_type": ContentFormat.ARTICLE,
                "content": "Research shows that regular exercise has many benefits for mental health. Studies have found that people who exercise regularly report lower levels of anxiety and depression. Physical activity increases the production of endorphins, which are natural mood boosters. Additionally, exercise can improve sleep quality and boost self-esteem.",
                "description": "Academic writing at intermediate level",
                "key_features": ["research-based", "factual", "clear structure", "evidence-based"],
                "author": "System",
                "source": "Generated",
                "quality_score": 0.90
            }
        ]
        
        conversational_examples = [
            {
                "style": ContentStyle.CONVERSATIONAL,
                "complexity": ComplexityLevel.INTERMEDIATE,
                "format_type": ContentFormat.BLOG,
                "content": "You know what's really interesting about climate change? It's not just about polar bears and melting ice caps (though those are definitely important!). The thing is, it's affecting all of us right now in ways we might not even realize. Have you noticed how the weather seems more unpredictable lately? That's not just your imagination.",
                "description": "Friendly, conversational tone with direct address",
                "key_features": ["direct address", "questions", "casual tone", "personal pronouns", "contractions"],
                "author": "System",
                "source": "Generated",
                "quality_score": 0.92
            }
        ]
        
        technical_examples = [
            {
                "style": ContentStyle.TECHNICAL,
                "complexity": ComplexityLevel.EXPERT,
                "format_type": ContentFormat.DOCUMENTATION,
                "content": "To implement OAuth 2.0 authentication, first configure the authorization server endpoint. Set the client_id and client_secret parameters in your application configuration. The authorization flow follows these steps: 1) Redirect user to authorization URL, 2) Handle callback with authorization code, 3) Exchange code for access token, 4) Use token for API requests. Ensure HTTPS is enabled for all endpoints.",
                "description": "Technical documentation with step-by-step instructions",
                "key_features": ["step-by-step", "technical terms", "imperative mood", "specific instructions"],
                "author": "System",
                "source": "Generated",
                "quality_score": 0.94
            }
        ]
        
        simplified_examples = [
            {
                "style": ContentStyle.SIMPLIFIED,
                "complexity": ComplexityLevel.ELEMENTARY,
                "format_type": ContentFormat.ARTICLE,
                "content": "Exercise is good for you. When you move your body, it makes you feel happy. This happens because your brain makes special chemicals called endorphins. These chemicals make you feel good. Exercise also helps you sleep better at night. Try to exercise for 30 minutes every day.",
                "description": "Simple language for elementary reading level",
                "key_features": ["simple words", "short sentences", "clear explanations", "basic concepts"],
                "author": "System",
                "source": "Generated",
                "quality_score": 0.88
            }
        ]
        
        all_examples = academic_examples + conversational_examples + technical_examples + simplified_examples
        
        for example_data in all_examples:
            example = StyleExample(**example_data)
            self.vector_store.add_style_example(example)
        
        logger.info(f"Loaded {len(all_examples)} default style examples")
    
    async def _load_transformation_patterns(self):
        
        transformation_cases = [
            {
                "original_content": "The research indicates that regular physical activity significantly improves cardiovascular health outcomes.",
                "transformed_content": "Studies show that exercise is really good for your heart!",
                "source_style": "academic",
                "target_style": "conversational",
                "source_complexity": "advanced",
                "target_complexity": "elementary",
                "quality_score": 0.89,
                "transformation_steps": [
                    "Simplify vocabulary (research → studies, indicates → show)",
                    "Change tone (formal → casual)",
                    "Add enthusiasm (exclamation)",
                    "Replace technical terms (cardiovascular → heart)"
                ]
            },
            {
                "original_content": "Hey everyone! Climate change is super scary, but we can totally fix it if we all work together!",
                "transformed_content": "Climate change presents significant challenges that require coordinated global action to address effectively.",
                "source_style": "conversational",
                "target_style": "formal",
                "source_complexity": "elementary",
                "target_complexity": "advanced",
                "quality_score": 0.91,
                "transformation_steps": [
                    "Remove casual greetings",
                    "Replace emotional language with neutral terms",
                    "Increase vocabulary complexity",
                    "Use formal sentence structure"
                ]
            }
        ]
        
        for case in transformation_cases:
            self.vector_store.add_transformation_case(**case)
        
        logger.info(f"Loaded {len(transformation_cases)} transformation patterns")
    
    def add_style_example(self, example: StyleExample) -> str:
        return self.vector_store.add_style_example(example)
    
    def add_transformation_case(self, 
                               original_content: str,
                               transformed_content: str,
                               source_style: str,
                               target_style: str,
                               source_complexity: str,
                               target_complexity: str,
                               quality_score: float,
                               transformation_steps: List[str] = None) -> str:
        return self.vector_store.add_transformation_case(
            original_content=original_content,
            transformed_content=transformed_content,
            source_style=source_style,
            target_style=target_style,
            source_complexity=source_complexity,
            target_complexity=target_complexity,
            quality_score=quality_score,
            transformation_steps=transformation_steps
        )
    
    def get_style_examples(self, 
                          content: str,
                          style: str,
                          complexity: str,
                          format_type: Optional[str] = None,
                          limit: int = 5) -> List[Dict[str, Any]]:
        return self.vector_store.search_similar_examples(
            content=content,
            style=style,
            complexity=complexity,
            format_type=format_type,
            limit=limit
        )
    
    def get_transformation_examples(self,
                                   content: str,
                                   source_style: str,
                                   target_style: str,
                                   source_complexity: Optional[str] = None,
                                   target_complexity: Optional[str] = None,
                                   limit: int = 5) -> List[Dict[str, Any]]:
        return self.vector_store.search_transformation_cases(
            content=content,            source_style=source_style,
            target_style=target_style,
            source_complexity=source_complexity,
            target_complexity=target_complexity,
            limit=limit
        )
    
    def get_style_guide(self, style: str, complexity: str) -> Dict[str, Any]:
        examples = self.vector_store.search_similar_examples(
            content=f"example of {style} style at {complexity} complexity",
            style=style,
            complexity=complexity,
            limit=3
        )
        
        return {
            "style": style,
            "complexity": complexity,
            "examples": examples,
            "guidelines": {
                "description": f"Style guide for {style} writing at {complexity} level",
                "key_features": []
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.vector_store.get_collection_stats()
    
    def learn_from_feedback(self, 
                           transformation_id: str,
                           original_content: str,
                           transformed_content: str,
                           transformation_metadata: Dict[str, Any],
                           quality_score: float,
                           user_feedback: Dict[str, Any]):
        try:
            if quality_score >= 0.8:
                self.add_transformation_case(
                    original_content=original_content,
                    transformed_content=transformed_content,
                    source_style=transformation_metadata.get("source_style", ""),
                    target_style=transformation_metadata.get("target_style", ""),
                    source_complexity=transformation_metadata.get("source_complexity", ""),
                    target_complexity=transformation_metadata.get("target_complexity", ""),
                    quality_score=quality_score,
                    transformation_steps=transformation_metadata.get("steps", [])
                )
                
                logger.info(f"Added successful transformation case from feedback: {transformation_id}")
            
            if user_feedback.get("suggested_improvements"):
                logger.info(f"Received improvement suggestions for {transformation_id}")
        
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
    
    def export_knowledge_base(self, file_path: str):
        try:
            stats = self.get_statistics()
            
            export_data = {
                "metadata": {
                    "export_timestamp": str(datetime.now()),
                    "statistics": stats
                },
                "note": "This is a summary export. Full vector data requires database backup."
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge base exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            raise
    
    def validate_knowledge_base(self) -> Dict[str, Any]:
        try:
            stats = self.get_statistics()
            
            validation_result = {
                "is_valid": True,
                "issues": [],
                "statistics": stats
            }
            
            if stats.get("style_examples", 0) < 4:
                validation_result["issues"].append("Insufficient style examples")
                validation_result["is_valid"] = False
            
            if stats.get("transformation_cases", 0) < 2:
                validation_result["issues"].append("Insufficient transformation cases")
                validation_result["is_valid"] = False
            
            required_styles = ["academic", "conversational", "technical", "simplified"]
            for style in required_styles:
                examples = self.vector_store.search_similar_examples(
                    content="sample text for validation",
                    style=style,
                    limit=1
                )
                if not examples:
                    validation_result["issues"].append(f"No examples for {style} style")
                    validation_result["is_valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate knowledge base: {e}")
            return {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "statistics": {}
            }


knowledge_base = KnowledgeBase()
