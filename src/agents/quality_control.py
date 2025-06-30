"""Quality Control Agent for content transformation system."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import math

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from textstat import (
        flesch_kincaid_grade, smog_index, automated_readability_index,
        coleman_liau_index, flesch_reading_ease
    )
except ImportError:
    def flesch_kincaid_grade(text): return 0.0
    def smog_index(text): return 0.0
    def automated_readability_index(text): return 0.0
    def coleman_liau_index(text): return 0.0
    def flesch_reading_ease(text): return 0.0

import numpy as np

from ..core.models import QualityAssessment, AgentState, TransformationPlan
from ..core.config_new import config, QualityMetrics
from ..core.usinc_client_new import create_usinc_client

logger = logging.getLogger(__name__)


class QualityControlAgent:
    def __init__(self):
        self.name = "QualityControlAgent"
        self.quality_metrics = QualityMetrics()
        self.usinc_client = create_usinc_client(config)
        
        try:
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(config.sentence_transformer_model)
            else:
                self.embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")
            self.embedding_model = None
        
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    async def assess_quality(self,
                           original_content: str,
                           transformed_content: str,
                           transformation_plan: TransformationPlan) -> QualityAssessment:
        try:
            logger.info("Starting comprehensive quality assessment")
            
            readability_score = self._assess_readability(transformed_content, transformation_plan)
            similarity_score = await self._assess_semantic_similarity(original_content, transformed_content)
            style_adherence = self._assess_style_adherence(transformed_content, transformation_plan)
            complexity_match = self._assess_complexity_match(transformed_content, transformation_plan)
            quality_indicators = self._assess_quality_indicators(transformed_content)
            fact_preservation = self._assess_fact_preservation(original_content, transformed_content)
            
            overall_score = self._calculate_overall_score({
                'readability': readability_score,
                'similarity': similarity_score,
                'style_adherence': style_adherence,
                'complexity_match': complexity_match,
                'quality_indicators': quality_indicators,
                'fact_preservation': fact_preservation
            })
            
            detailed_metrics = self._generate_detailed_metrics(
                original_content, transformed_content, transformation_plan
            )
            
            issues = self._identify_quality_issues(
                original_content, transformed_content, transformation_plan,
                {
                    'readability': readability_score,
                    'similarity': similarity_score,
                    'style_adherence': style_adherence,
                    'complexity_match': complexity_match,
                    'quality_indicators': quality_indicators,
                    'fact_preservation': fact_preservation
                }
            )
            
            suggestions = self._generate_improvement_suggestions(issues, transformation_plan)
            
            assessment = QualityAssessment(
                overall_score=overall_score,
                readability_score=readability_score,
                similarity_score=similarity_score,
                style_adherence=style_adherence,
                complexity_match=complexity_match,
                fact_preservation=fact_preservation,
                detailed_metrics=detailed_metrics,
                quality_issues=issues,
                improvement_suggestions=suggestions,
                assessment_timestamp=datetime.now()
            )
            
            logger.info(f"Quality assessment completed. Overall score: {overall_score:.2f}")
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityAssessment(
                overall_score=0.5,
                readability_score=0.5,
                similarity_score=0.5,
                style_adherence=0.5,
                complexity_match=0.5,
                fact_preservation=0.5,
                detailed_metrics={},
                quality_issues=["Assessment failed due to error"],
                improvement_suggestions=["Manual review recommended"],
                assessment_timestamp=datetime.now()
            )
    
    def _assess_readability(self, content: str, transformation_plan: TransformationPlan) -> float:
        try:
            target_complexity = transformation_plan.target_complexity.value
            
            readability_scores = {}
            try:
                readability_scores['flesch_kincaid'] = flesch_kincaid_grade(content)
                readability_scores['flesch_reading_ease'] = flesch_reading_ease(content)
                readability_scores['smog'] = smog_index(content)
                readability_scores['automated_readability'] = automated_readability_index(content)
                readability_scores['coleman_liau'] = coleman_liau_index(content)
            except Exception as e:
                logger.warning(f"Failed to calculate readability scores: {e}")
                return 0.7
            
            avg_readability = np.mean([score for score in readability_scores.values() if score > 0])
            
            if target_complexity == "elementary":
                target_range = (3, 8)
            elif target_complexity == "intermediate":
                target_range = (8, 12)
            elif target_complexity == "advanced":
                target_range = (12, 16)
            else:
                target_range = (16, 20)
            
            if target_range[0] <= avg_readability <= target_range[1]:
                score = 1.0
            else:
                deviation = min(abs(avg_readability - target_range[0]), abs(avg_readability - target_range[1]))
                score = max(0.0, 1.0 - (deviation / 10.0))
            
            return score
            
        except Exception as e:
            logger.warning(f"Readability assessment failed: {e}")
            return 0.7
    
    async def _assess_semantic_similarity(self, original: str, transformed: str) -> float:
        try:
            try:
                original_embedding = await self.usinc_client.create_embedding(original)
                transformed_embedding = await self.usinc_client.create_embedding(transformed)
                
                if original_embedding and transformed_embedding:
                    original_vec = np.array(original_embedding)
                    transformed_vec = np.array(transformed_embedding)
                    
                    similarity = np.dot(original_vec, transformed_vec) / (
                        np.linalg.norm(original_vec) * np.linalg.norm(transformed_vec)
                    )
                    return float(similarity)
                    
            except Exception as usinc_error:
                logger.warning(f"US.inc embedding failed: {usinc_error}")
            
            return self._calculate_fallback_similarity(original, transformed)
            
        except Exception as e:
            logger.warning(f"Similarity assessment failed: {e}")
            return 0.7
    
    def _calculate_fallback_similarity(self, original: str, transformed: str) -> float:
        try:
            original_words = set(original.lower().split())
            transformed_words = set(transformed.lower().split())
            
            intersection = original_words.intersection(transformed_words)
            union = original_words.union(transformed_words)
            
            if not union:
                return 0.0
            
            jaccard_similarity = len(intersection) / len(union)
            return jaccard_similarity
            
        except Exception as e:
            logger.warning(f"Fallback similarity failed: {e}")
            return 0.5
    
    def _assess_style_adherence(self, content: str, transformation_plan: TransformationPlan) -> float:
        try:
            target_style = transformation_plan.target_style.value
            
            if target_style == "academic":
                return self._check_academic_style(content)
            elif target_style == "conversational":
                return self._check_conversational_style(content)
            elif target_style == "technical":
                return self._check_technical_style(content)
            elif target_style == "formal":
                return self._check_formal_style(content)
            elif target_style == "simplified":
                return self._check_simplified_style(content)
            else:
                return 0.7
                
        except Exception as e:
            logger.warning(f"Style adherence assessment failed: {e}")
            return 0.7
    
    def _check_academic_style(self, content: str) -> float:
        score = 0.0
        content_lower = content.lower()
        
        academic_indicators = [
            "research", "study", "analysis", "findings", "results", 
            "methodology", "data", "evidence", "conclusion", "therefore"
        ]
        
        for indicator in academic_indicators:
            if indicator in content_lower:
                score += 0.1
        
        if re.search(r'\b(the|this)\s+(study|research|analysis)', content_lower):
            score += 0.2
        
        if re.search(r'\bet al\.|according to', content_lower):
            score += 0.1
        
        if not re.search(r'\b(you|your|we|us|our)\b', content_lower):
            score += 0.1
        
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length > 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_conversational_style(self, content: str) -> float:
        score = 0.0
        content_lower = content.lower()
        
        if re.search(r'\b(you|your)\b', content_lower):
            score += 0.3
        
        if re.search(r'\b(we|us|our)\b', content_lower):
            score += 0.2
        
        conversational_words = ["really", "pretty", "quite", "actually", "honestly", "basically"]
        for word in conversational_words:
            if word in content_lower:
                score += 0.05
        
        if re.search(r'[?!]', content):
            score += 0.1
        
        contractions = ["don't", "won't", "can't", "isn't", "aren't", "doesn't"]
        for contraction in contractions:
            if contraction in content_lower:
                score += 0.05
        
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length < 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_technical_style(self, content: str) -> float:
        score = 0.0
        content_lower = content.lower()
        
        technical_terms = [
            "system", "process", "function", "method", "algorithm", "implementation",
            "configuration", "parameter", "specification", "protocol", "interface"
        ]
        
        for term in technical_terms:
            if term in content_lower:
                score += 0.08
        
        if re.search(r'\b(step|steps)\s+\d+', content_lower):
            score += 0.15
        
        if re.search(r'\b(first|second|third|finally)\b', content_lower):
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_formal_style(self, content: str) -> float:
        score = 0.0
        content_lower = content.lower()
        
        formal_words = ["shall", "hereby", "furthermore", "moreover", "consequently", "therefore"]
        for word in formal_words:
            if word in content_lower:
                score += 0.15
        
        if not re.search(r'\b(you|your|we|us|our)\b', content_lower):
            score += 0.2
        
        if not re.search(r"[!'?]", content):
            score += 0.1
        
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length > 25:
            score += 0.15
        
        return min(score, 1.0)
    
    def _check_simplified_style(self, content: str) -> float:
        score = 0.0
        
        words = content.split()
        if words:
            simple_words = [word for word in words if len(word) <= 6]
            simple_ratio = len(simple_words) / len(words)
            score += simple_ratio * 0.4
        
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length < 15:
            score += 0.3
        
        simple_connectors = ["and", "but", "so", "then", "also"]
        for connector in simple_connectors:
            if connector in content.lower():
                score += 0.06
        
        return min(score, 1.0)
    
    def _assess_complexity_match(self, content: str, transformation_plan: TransformationPlan) -> float:
        try:
            target_complexity = transformation_plan.target_complexity.value
            
            words = content.split()
            word_count = len(words)
            
            if word_count == 0:
                return 0.0
            
            complex_words = [word for word in words if len(word) > 6]
            complex_ratio = len(complex_words) / word_count
            
            sentences = content.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            score = 0.0
            
            if target_complexity == "elementary":
                if complex_ratio <= 0.15 and avg_sentence_length <= 15:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - abs(complex_ratio - 0.1) - abs(avg_sentence_length - 12) / 20)
            elif target_complexity == "intermediate":
                if 0.15 <= complex_ratio <= 0.25 and 15 <= avg_sentence_length <= 20:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - abs(complex_ratio - 0.2) - abs(avg_sentence_length - 17) / 25)
            elif target_complexity == "advanced":
                if 0.25 <= complex_ratio <= 0.35 and 20 <= avg_sentence_length <= 25:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 - abs(complex_ratio - 0.3) - abs(avg_sentence_length - 22) / 30)
            else:
                if complex_ratio >= 0.35 and avg_sentence_length >= 25:
                    score = 1.0
                else:
                    score = max(0.0, min(complex_ratio / 0.35, avg_sentence_length / 25))
            
            return score
            
        except Exception as e:
            logger.warning(f"Complexity assessment failed: {e}")
            return 0.7
    
    def _assess_quality_indicators(self, content: str) -> float:
        score = 0.0
        
        if len(content.strip()) > 10:
            score += 0.2
        
        if not re.search(r'(.)\1{3,}', content):
            score += 0.1
        
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 1:
            score += 0.2
        
        if re.search(r'^[A-Z]', content.strip()):
            score += 0.1
        
        if content.strip().endswith(('.', '!', '?')):
            score += 0.1
        
        words = content.split()
        if len(set(words)) / len(words) > 0.7 if words else True:
            score += 0.15
        
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.15
        
        return min(score, 1.0)
    
    def _assess_fact_preservation(self, original: str, transformed: str) -> float:
        try:
            original_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', original)
            transformed_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', transformed)
            
            original_caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original)
            transformed_caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', transformed)
            
            score = 0.0
            
            if original_numbers:
                preserved_numbers = sum(1 for num in original_numbers if num in transformed)
                score += (preserved_numbers / len(original_numbers)) * 0.4
            else:
                score += 0.4
            
            if original_caps:
                preserved_caps = sum(1 for cap in original_caps if cap in transformed)
                score += (preserved_caps / len(original_caps)) * 0.4
            else:
                score += 0.4
            
            original_words = set(original.lower().split())
            transformed_words = set(transformed.lower().split())
            key_words = original_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            if key_words:
                preserved_key_words = key_words.intersection(transformed_words)
                score += (len(preserved_key_words) / len(key_words)) * 0.2
            else:
                score += 0.2
            
            return score
            
        except Exception as e:
            logger.warning(f"Fact preservation assessment failed: {e}")
            return 0.7
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        weights = {
            'readability': 0.2,
            'similarity': 0.25,
            'style_adherence': 0.2,
            'complexity_match': 0.15,
            'quality_indicators': 0.1,
            'fact_preservation': 0.1
        }
        
        weighted_sum = sum(scores[metric] * weights[metric] for metric in scores if metric in weights)
        return weighted_sum
    
    def _generate_detailed_metrics(self, original: str, transformed: str, plan: TransformationPlan) -> Dict[str, Any]:
        metrics = {}
        
        try:
            metrics['word_count_original'] = len(original.split())
            metrics['word_count_transformed'] = len(transformed.split())
            metrics['length_ratio'] = metrics['word_count_transformed'] / metrics['word_count_original'] if metrics['word_count_original'] > 0 else 0
            
            metrics['sentence_count_original'] = len(re.findall(r'[.!?]+', original))
            metrics['sentence_count_transformed'] = len(re.findall(r'[.!?]+', transformed))
            
            if metrics['sentence_count_transformed'] > 0:
                metrics['avg_sentence_length'] = metrics['word_count_transformed'] / metrics['sentence_count_transformed']
            else:
                metrics['avg_sentence_length'] = 0
            
            try:
                metrics['flesch_kincaid_grade'] = flesch_kincaid_grade(transformed)
                metrics['flesch_reading_ease'] = flesch_reading_ease(transformed)
            except:
                metrics['flesch_kincaid_grade'] = 0
                metrics['flesch_reading_ease'] = 0
            
            original_words = set(original.lower().split())
            transformed_words = set(transformed.lower().split())
            metrics['vocabulary_overlap'] = len(original_words.intersection(transformed_words)) / len(original_words.union(transformed_words)) if original_words.union(transformed_words) else 0
            
            metrics['target_style'] = plan.target_style.value
            metrics['target_complexity'] = plan.target_complexity.value
            metrics['target_format'] = plan.target_format.value
            
        except Exception as e:
            logger.warning(f"Failed to generate some metrics: {e}")
        
        return metrics
    
    def _identify_quality_issues(self, original: str, transformed: str, plan: TransformationPlan, scores: Dict[str, float]) -> List[str]:
        issues = []
        
        try:
            if scores.get('readability', 1.0) < 0.6:
                issues.append("Readability does not match target complexity level")
            
            if scores.get('similarity', 1.0) < 0.5:
                issues.append("Semantic similarity too low - meaning may be lost")
            
            if scores.get('style_adherence', 1.0) < 0.6:
                issues.append(f"Content does not adhere well to {plan.target_style.value} style")
            
            if scores.get('complexity_match', 1.0) < 0.6:
                issues.append(f"Complexity level does not match target ({plan.target_complexity.value})")
            
            if scores.get('fact_preservation', 1.0) < 0.7:
                issues.append("Important factual information may be lost")
            
            if len(transformed.split()) < 10:
                issues.append("Transformed content is too short")
            
            word_count_ratio = len(transformed.split()) / len(original.split()) if len(original.split()) > 0 else 0
            if word_count_ratio > 2.0:
                issues.append("Transformed content is significantly longer than original")
            elif word_count_ratio < 0.3:
                issues.append("Transformed content is significantly shorter than original")
            
            if not transformed.strip().endswith(('.', '!', '?')):
                issues.append("Content appears incomplete (no proper ending)")
            
            if re.search(r'(.)\1{4,}', transformed):
                issues.append("Repetitive patterns detected")
            
        except Exception as e:
            logger.warning(f"Issue identification failed: {e}")
            issues.append("Unable to fully assess quality due to analysis error")
        
        return issues
    
    def _generate_improvement_suggestions(self, issues: List[str], plan: TransformationPlan) -> List[str]:
        suggestions = []
        
        try:
            for issue in issues:
                if "readability" in issue.lower():
                    if plan.target_complexity.value == "elementary":
                        suggestions.append("Use shorter sentences and simpler words")
                    elif plan.target_complexity.value == "advanced":
                        suggestions.append("Consider using more sophisticated vocabulary and complex sentence structures")
                
                if "similarity" in issue.lower():
                    suggestions.append("Ensure key concepts and main ideas are preserved from the original")
                
                if "style" in issue.lower():
                    style = plan.target_style.value
                    if style == "conversational":
                        suggestions.append("Add more direct address (you/your) and informal language")
                    elif style == "academic":
                        suggestions.append("Use more formal tone and research-based language")
                    elif style == "technical":
                        suggestions.append("Include more technical terminology and step-by-step explanations")
                
                if "complexity" in issue.lower():
                    suggestions.append(f"Adjust vocabulary and sentence complexity to match {plan.target_complexity.value} level")
                
                if "factual" in issue.lower():
                    suggestions.append("Verify all numbers, names, and key facts are preserved")
                
                if "short" in issue.lower():
                    suggestions.append("Expand on key points while maintaining clarity")
                
                if "long" in issue.lower():
                    suggestions.append("Focus on essential information and remove redundancy")
                
                if "incomplete" in issue.lower():
                    suggestions.append("Ensure content has proper conclusion and ending")
            
            if not suggestions:
                suggestions.append("Content meets quality standards")
                
        except Exception as e:
            logger.warning(f"Suggestion generation failed: {e}")
            suggestions.append("Manual review recommended for quality improvement")
        
        return suggestions
    
    async def process_state(self, state: AgentState) -> AgentState:
        try:
            logger.info(f"Quality Control Agent processing request: {state.request_id}")
            
            if not state.transformed_content:
                raise ValueError("No transformed content available for quality assessment")
            
            if not state.transformation_plan:
                raise ValueError("No transformation plan available for quality assessment")
            
            assessment = await self.assess_quality(
                original_content=state.original_content,
                transformed_content=state.transformed_content,
                transformation_plan=state.transformation_plan
            )
            
            state.quality_assessment = assessment
            state.current_step = "quality_control_completed"
            state.agent_outputs[self.name] = {
                "assessment": assessment.dict(),
                "timestamp": str(datetime.now()),
                "status": "completed"
            }
            
            logger.info(f"Quality assessment completed for request: {state.request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Quality Control Agent failed: {e}")
            state.errors.append(f"Quality Control Agent error: {str(e)}")
            state.agent_outputs[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": str(datetime.now())
            }
            return state
    
    async def _get_embedding_fallback(self, text: str) -> Optional[np.ndarray]:
        try:
            if self.embedding_model:
                embedding = self.embedding_model.encode([text])
                return embedding[0]
            return None
        except Exception as e:
            logger.warning(f"Fallback embedding failed: {e}")
            return None
