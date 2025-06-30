"""Transformation Planning Agent for content transformation system."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..core.models import (
    TransformationPlan, ContentAnalysis, AgentState, 
    ContentStyle, ComplexityLevel, ContentFormat
)
from ..core.config_new import config, StyleConfig, QualityMetrics
from ..rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)


class TransformationPlanningAgent:
    def __init__(self):
        self.name = "TransformationPlanningAgent"
        self.style_config = StyleConfig()
        self.quality_metrics = QualityMetrics()
    
    async def create_transformation_plan(self,
                                       source_analysis: ContentAnalysis,
                                       target_style: ContentStyle,
                                       target_complexity: ComplexityLevel,
                                       target_format: ContentFormat,
                                       user_instructions: Optional[str] = None) -> TransformationPlan:
        try:
            logger.info(f"Creating transformation plan: {source_analysis.style} -> {target_style.value}")
            
            similar_examples = knowledge_base.get_transformation_examples(
                content=f"example transformation from {source_analysis.style} to {target_style.value}",
                source_style=source_analysis.style,
                target_style=target_style.value,
                source_complexity=source_analysis.complexity_level,
                target_complexity=target_complexity.value,
                limit=5
            )
            
            style_guide = knowledge_base.get_style_guide(
                style=target_style.value,
                complexity=target_complexity.value
            )
            
            transformation_steps = self._create_transformation_steps(
                source_analysis=source_analysis,
                target_style=target_style,
                target_complexity=target_complexity,
                target_format=target_format,
                style_guide=style_guide,
                similar_examples=similar_examples,
                user_instructions=user_instructions
            )
            
            preservation_requirements = self._determine_preservation_requirements(
                source_analysis=source_analysis,
                target_style=target_style,
                target_format=target_format
            )
            
            quality_targets = self._set_quality_targets(
                source_analysis=source_analysis,
                target_complexity=target_complexity
            )
            
            difficulty = self._estimate_difficulty(
                source_analysis=source_analysis,
                target_style=target_style,
                target_complexity=target_complexity,
                target_format=target_format
            )
            
            similar_example_ids = [ex.get("id", "") for ex in similar_examples]
            
            plan = TransformationPlan(
                source_analysis=source_analysis,
                target_style=target_style,
                target_complexity=target_complexity,
                target_format=target_format,
                transformation_steps=transformation_steps,
                preservation_requirements=preservation_requirements,
                quality_targets=quality_targets,
                estimated_difficulty=difficulty,
                similar_examples=similar_example_ids
            )
            
            logger.info(f"Transformation plan created with {len(transformation_steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create transformation plan: {e}")
            raise
    
    def _create_transformation_steps(self,
                                   source_analysis: ContentAnalysis,
                                   target_style: ContentStyle,
                                   target_complexity: ComplexityLevel,
                                   target_format: ContentFormat,
                                   style_guide: Dict[str, Any],
                                   similar_examples: List[Dict[str, Any]],
                                   user_instructions: Optional[str]) -> List[Dict[str, Any]]:
        steps = []
        
        steps.append({
            "title": "Content Analysis",
            "description": f"Analyze source content style ({source_analysis.style}) and complexity ({source_analysis.complexity_level})",
            "actions": [
                "Identify key concepts and main ideas",
                "Extract important facts and entities",
                "Note current style characteristics"
            ]
        })
        
        if source_analysis.style != target_style.value:
            steps.append({
                "title": "Style Transformation",
                "description": f"Transform from {source_analysis.style} to {target_style.value} style",
                "actions": self._get_style_transformation_actions(source_analysis.style, target_style.value)
            })
        
        if source_analysis.complexity_level != target_complexity.value:
            steps.append({
                "title": "Complexity Adjustment",
                "description": f"Adjust complexity from {source_analysis.complexity_level} to {target_complexity.value}",
                "actions": self._get_complexity_adjustment_actions(source_analysis.complexity_level, target_complexity.value)
            })
        
        if target_format != ContentFormat.PARAGRAPH:
            steps.append({
                "title": "Format Adaptation",
                "description": f"Adapt content to {target_format.value} format",
                "actions": self._get_format_adaptation_actions(target_format.value)
            })
        
        if user_instructions:
            steps.append({
                "title": "Custom Requirements",
                "description": "Apply user-specific instructions",
                "actions": [f"Apply instruction: {user_instructions}"]
            })
        
        steps.append({
            "title": "Quality Review",
            "description": "Final review and polish",
            "actions": [
                "Check coherence and flow",
                "Verify fact preservation",
                "Ensure style consistency",
                "Final proofreading"
            ]
        })
        
        return steps
    
    def _get_style_transformation_actions(self, source_style: str, target_style: str) -> List[str]:
        actions = []
        
        if target_style == "conversational":
            actions.extend([
                "Add direct address (you/your)",
                "Use contractions where appropriate",
                "Include rhetorical questions",
                "Use informal connecting words"
            ])
        elif target_style == "academic":
            actions.extend([
                "Use formal tone and third person",
                "Add research-based language",
                "Include logical connectors",
                "Use precise terminology"
            ])
        elif target_style == "technical":
            actions.extend([
                "Include technical terminology",
                "Add step-by-step structure",
                "Use imperative mood for instructions",
                "Ensure precision and clarity"
            ])
        elif target_style == "simplified":
            actions.extend([
                "Replace complex words with simple alternatives",
                "Break down complex concepts",
                "Use everyday language",
                "Ensure clear explanations"
            ])
        elif target_style == "formal":
            actions.extend([
                "Use formal vocabulary",
                "Avoid contractions and colloquialisms",
                "Use complex sentence structures",
                "Maintain professional tone"
            ])
        
        return actions
    
    def _get_complexity_adjustment_actions(self, source_complexity: str, target_complexity: str) -> List[str]:
        actions = []
        
        if target_complexity == "elementary":
            actions.extend([
                "Use simple vocabulary (6th grade level)",
                "Shorten sentences (under 15 words)",
                "Explain technical terms",
                "Use basic sentence structures"
            ])
        elif target_complexity == "intermediate":
            actions.extend([
                "Use moderate vocabulary complexity",
                "Maintain medium sentence length (15-20 words)",
                "Balance simple and complex ideas",
                "Include some technical terms with context"
            ])
        elif target_complexity == "advanced":
            actions.extend([
                "Use sophisticated vocabulary",
                "Allow longer sentences (20-25 words)",
                "Include complex ideas and concepts",
                "Use technical terminology appropriately"
            ])
        elif target_complexity == "expert":
            actions.extend([
                "Use specialized terminology",
                "Allow complex sentence structures",
                "Include advanced concepts",
                "Assume domain expertise"
            ])
        
        return actions
    
    def _get_format_adaptation_actions(self, target_format: str) -> List[str]:
        actions = []
        
        if target_format == "email":
            actions.extend([
                "Add appropriate greeting",
                "Structure with clear paragraphs",
                "Include professional closing",
                "Use email-appropriate tone"
            ])
        elif target_format == "social_media":
            actions.extend([
                "Keep under character limit",
                "Use engaging language",
                "Include relevant hashtags if needed",
                "Make it shareable"
            ])
        elif target_format == "bullet_points":
            actions.extend([
                "Break into concise points",
                "Use parallel structure",
                "Ensure each point is complete",
                "Order logically"
            ])
        elif target_format == "blog":
            actions.extend([
                "Create engaging introduction",
                "Use subheadings for structure",
                "Write compelling conclusion",
                "Ensure readability"
            ])
        elif target_format == "report":
            actions.extend([
                "Use formal report structure",
                "Include clear sections",
                "Use professional language",
                "Ensure comprehensive coverage"
            ])
        
        return actions
    
    def _determine_preservation_requirements(self,
                                           source_analysis: ContentAnalysis,
                                           target_style: ContentStyle,
                                           target_format: ContentFormat) -> List[str]:
        requirements = ["factual_accuracy", "key_concepts"]
        
        if source_analysis.key_entities:
            requirements.append("named_entities")
        
        if target_style in [ContentStyle.ACADEMIC, ContentStyle.TECHNICAL, ContentStyle.FORMAL]:
            requirements.append("technical_terms")
        
        if target_format in [ContentFormat.REPORT, ContentFormat.ACADEMIC_PAPER, ContentFormat.DOCUMENTATION]:
            requirements.extend(["data_points", "citations"])
        
        if source_analysis.sentiment_score and abs(source_analysis.sentiment_score) > 0.3:
            requirements.append("sentiment_tone")
        
        return requirements
    
    def _set_quality_targets(self,
                           source_analysis: ContentAnalysis,
                           target_complexity: ComplexityLevel) -> Dict[str, float]:
        targets = {
            "readability_score": 0.8,
            "semantic_similarity": 0.7,
            "style_adherence": 0.85,
            "fact_preservation": 0.9,
            "overall_quality": 0.8
        }
        
        if target_complexity == ComplexityLevel.ELEMENTARY:
            targets["readability_score"] = 0.9
        elif target_complexity == ComplexityLevel.EXPERT:
            targets["semantic_similarity"] = 0.8
            targets["style_adherence"] = 0.9
        
        if source_analysis.confidence_score < 0.7:
            for key in targets:
                targets[key] *= 0.9
        
        return targets
    
    def _estimate_difficulty(self,
                           source_analysis: ContentAnalysis,
                           target_style: ContentStyle,
                           target_complexity: ComplexityLevel,
                           target_format: ContentFormat) -> float:
        difficulty = 0.5
        
        style_distance = self._calculate_style_distance(source_analysis.style, target_style.value)
        complexity_distance = self._calculate_complexity_distance(source_analysis.complexity_level, target_complexity.value)
        
        difficulty += style_distance * 0.3
        difficulty += complexity_distance * 0.25
        
        if target_format in [ContentFormat.SOCIAL_MEDIA, ContentFormat.BULLET_POINTS]:
            difficulty += 0.1
        elif target_format in [ContentFormat.ACADEMIC_PAPER, ContentFormat.REPORT]:
            difficulty += 0.15
        
        if source_analysis.word_count > 500:
            difficulty += 0.1
        
        if source_analysis.confidence_score < 0.7:
            difficulty += 0.1
        
        return min(difficulty, 1.0)
    
    def _calculate_style_distance(self, source_style: str, target_style: str) -> float:
        if source_style == target_style:
            return 0.0
        
        style_groups = {
            "formal_group": ["academic", "formal", "professional"],
            "casual_group": ["conversational", "informal", "casual"],
            "specialized_group": ["technical", "simplified"]
        }
        
        source_group = None
        target_group = None
        
        for group, styles in style_groups.items():
            if source_style in styles:
                source_group = group
            if target_style in styles:
                target_group = group
        
        if source_group == target_group:
            return 0.3
        elif source_group and target_group:
            return 0.6
        else:
            return 0.5
    
    def _calculate_complexity_distance(self, source_complexity: str, target_complexity: str) -> float:
        complexity_levels = ["elementary", "intermediate", "advanced", "expert"]
        
        try:
            source_index = complexity_levels.index(source_complexity)
            target_index = complexity_levels.index(target_complexity)
            distance = abs(target_index - source_index) / (len(complexity_levels) - 1)
            return distance
        except ValueError:
            return 0.5
    
    async def process_state(self, state: AgentState) -> AgentState:
        try:
            logger.info(f"Transformation Planning Agent processing request: {state.request_id}")
            
            if not state.source_analysis:
                raise ValueError("Source analysis required for transformation planning")
            
            plan = await self.create_transformation_plan(
                source_analysis=state.source_analysis,
                target_style=state.target_style,
                target_complexity=state.target_complexity,
                target_format=state.target_format,
                user_instructions=state.user_instructions
            )
            
            state.transformation_plan = plan
            state.current_step = "planning_completed"
            state.agent_outputs[self.name] = {
                "plan": plan.dict(),
                "timestamp": str(datetime.now()),
                "status": "completed"
            }
            
            logger.info(f"Transformation planning completed for request: {state.request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Transformation Planning Agent failed: {e}")
            state.errors.append(f"Transformation Planning Agent error: {str(e)}")
            state.agent_outputs[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": str(datetime.now())
            }
            return state
