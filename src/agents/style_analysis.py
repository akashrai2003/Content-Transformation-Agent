"""Style Analysis Agent for content transformation system."""

import logging
from typing import Dict, List, Any, Optional
import re
from collections import Counter
import numpy as np
from datetime import datetime

try:
    import nltk
except ImportError:
    nltk = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    from textstat import flesch_kincaid_grade, smog_index, automated_readability_index, coleman_liau_index
except ImportError:
    def flesch_kincaid_grade(text): return 0.0
    def smog_index(text): return 0.0
    def automated_readability_index(text): return 0.0
    def coleman_liau_index(text): return 0.0

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..core.models import ContentAnalysis, AgentState, ContentStyle, ComplexityLevel, ContentFormat
from ..core.config_new import config, StyleConfig
from ..core.usinc_client_new import create_usinc_client
from ..rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class StyleAnalysisAgent:
    def __init__(self):
        self.name = "StyleAnalysisAgent"
        self.usinc_client = create_usinc_client(config)
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        try:
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(config.sentence_transformer_model)
            else:
                self.embedding_model = None
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")
            self.embedding_model = None
        
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    async def analyze_content(self, content: str) -> ContentAnalysis:
        try:
            logger.info(f"Analyzing content: {content[:100]}...")
            
            word_count = len(content.split())
            sentences = self._split_sentences(content)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            readability_scores = self._calculate_readability_scores(content)
            
            style_detection
            detected_style = self._detect_style(content)
            
            # Complexity level detection
            complexity_level = self._detect_complexity(content, readability_scores)
            
            # Format type detection
            format_type = self._detect_format(content)
            
            # Extract entities and topics
            key_entities = self._extract_entities(content)
            main_topics = self._extract_topics(content)
            
            sentiment_score = self._analyze_sentiment(content)
            
            confidence_score = self._calculate_confidence(content, readability_scores)
            
            analysis = ContentAnalysis(
                style=detected_style,
                complexity_level=complexity_level,
                format_type=format_type,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                readability_scores=readability_scores,
                key_entities=key_entities,
                main_topics=main_topics,
                sentiment_score=sentiment_score,
                confidence_score=confidence_score
            )
            
            logger.info(f"Content analysis completed: {detected_style} style, {complexity_level} complexity")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {e}")
            return ContentAnalysis(
                style="conversational",
                complexity_level="intermediate",
                format_type="article",
                word_count=len(content.split()),
                sentence_count=1,
                paragraph_count=1,
                readability_scores={},
                key_entities=[],
                main_topics=[],
                sentiment_score=0.0,
                confidence_score=0.5
            )
    
    def _split_sentences(self, content: str) -> List[str]:
        try:
            if self.nlp:
                doc = self.nlp(content)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                sentences = re.split(r'[.!?]+', content)
                return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"Sentence splitting failed: {e}")
            return [content]
    
    def _calculate_readability_scores(self, content: str) -> Dict[str, float]:
        scores = {}
        
        try:
            scores['flesch_kincaid'] = flesch_kincaid_grade(content)
        except Exception:
            scores['flesch_kincaid'] = 0.0
        
        try:
            scores['smog'] = smog_index(content)
        except Exception:
            scores['smog'] = 0.0
        
        try:
            scores['automated_readability'] = automated_readability_index(content)
        except Exception:
            scores['automated_readability'] = 0.0
        
        try:
            scores['coleman_liau'] = coleman_liau_index(content)
        except Exception:
            scores['coleman_liau'] = 0.0
        
        valid_scores = [score for score in scores.values() if score > 0]
        if valid_scores:
            scores['average'] = sum(valid_scores) / len(valid_scores)
        else:
            scores['average'] = 0.0
        
        return scores
    
    def _detect_style(self, content: str) -> str:
        try:
            style_indicators = {
                "academic": 0,
                "conversational": 0,
                "formal": 0,
                "technical": 0,
                "simplified": 0,
                "professional": 0
            }
            
            content_lower = content.lower()
            
            # Academic indicators
            academic_patterns = [
                r'\b(research|study|studies|analysis|examination|investigation)\b',
                r'\b(however|therefore|furthermore|moreover|consequently)\b',
                r'\b(et al\.|cite|reference|bibliography)\b',
                r'\([^)]*\d{4}[^)]*\)',  # Citations like (Smith, 2023)
            ]
            
            for pattern in academic_patterns:
                style_indicators["academic"] += len(re.findall(pattern, content_lower))
            
            # Conversational indicators
            conversational_patterns = [
                r'\b(you|your|we|us|our)\b',
                r'\b(hey|hi|hello|wow|great|awesome)\b',
                r"[?!]{1,3}",  # Questions and exclamations
                r"\b(can't|won't|don't|isn't|aren't)\b",  # Contractions
            ]
            
            for pattern in conversational_patterns:
                style_indicators["conversational"] += len(re.findall(pattern, content_lower))
            
            # Technical indicators
            technical_patterns = [
                r'\b(implement|configure|execute|process|function|method)\b',
                r'\b(step \d+|first|second|third|finally)\b',
                r'\b(api|sdk|algorithm|protocol|framework)\b',
                r'[{}()\[\]]',  # Code-like brackets
            ]
            
            for pattern in technical_patterns:
                style_indicators["technical"] += len(re.findall(pattern, content_lower))
            
            # Formal indicators
            formal_patterns = [
                r'\b(shall|will|must|should|may|might)\b',
                r'\b(request|require|recommend|suggest|propose)\b',
                r'\b(please|kindly|respectfully|sincerely)\b',
            ]
            
            for pattern in formal_patterns:
                style_indicators["formal"] += len(re.findall(pattern, content_lower))
            
            # Simplified indicators
            simplified_patterns = [
                r'\b(easy|simple|basic|clear|quick)\b',
                r'\b(this means|in other words|for example)\b',
            ]
            
            for pattern in simplified_patterns:
                style_indicators["simplified"] += len(re.findall(pattern, content_lower))
            
            # Professional indicators
            professional_patterns = [
                r'\b(business|company|organization|management|strategy)\b',
                r'\b(meeting|project|team|deadline|goals)\b',
            ]
            
            for pattern in professional_patterns:
                style_indicators["professional"] += len(re.findall(pattern, content_lower))
            
            # Determine dominant style
            if max(style_indicators.values()) == 0:
                return "conversational"  # Default
            
            detected_style = max(style_indicators, key=style_indicators.get)
            
            # Use RAG to validate against known examples
            similar_examples = knowledge_base.get_style_examples(
                content=content[:500],  # First 500 chars for efficiency
                style=detected_style,
                complexity="intermediate",
                limit=3
            )
            
            if similar_examples and similar_examples[0].get("score", 0) > 0.7:
                return detected_style
            
            # Fallback: check other styles
            for style in style_indicators:
                similar_examples = knowledge_base.get_style_examples(
                    content=content[:500],
                    style=style,
                    complexity="intermediate",
                    limit=1
                )
                if similar_examples and similar_examples[0].get("score", 0) > 0.8:
                    return style
            
            return detected_style
            
        except Exception as e:
            logger.warning(f"Style detection failed: {e}")
            return "conversational"
    
    def _detect_complexity(self, content: str, readability_scores: Dict[str, float]) -> str:
        """Detect the complexity level of the content."""
        try:
            # Use average readability score as primary indicator
            avg_readability = readability_scores.get('average', 0)
            
            if avg_readability <= 6:
                complexity = "elementary"
            elif avg_readability <= 9:
                complexity = "intermediate"
            elif avg_readability <= 13:
                complexity = "advanced"
            else:
                complexity = "expert"
            
            # Additional complexity indicators
            word_count = len(content.split())
            sentences = self._split_sentences(content)
            avg_sentence_length = word_count / len(sentences) if sentences else 0
            
            # Adjust based on sentence complexity
            if avg_sentence_length > 25:
                if complexity in ["elementary", "intermediate"]:
                    complexity = "advanced"
            elif avg_sentence_length < 10:
                if complexity in ["advanced", "expert"]:
                    complexity = "intermediate"
            
            # Check for technical vocabulary
            technical_words = len(re.findall(
                r'\b\w{10,}\b',  # Words longer than 10 characters
                content
            ))
            
            if technical_words > word_count * 0.1:  # More than 10% long words
                if complexity == "elementary":
                    complexity = "intermediate"
                elif complexity == "intermediate":
                    complexity = "advanced"
            
            return complexity
            
        except Exception as e:
            logger.warning(f"Complexity detection failed: {e}")
            return "intermediate"
    
    def _detect_format(self, content: str) -> str:
        """Detect the format type of the content."""
        try:
            format_indicators = {
                "email": 0,
                "article": 0,
                "report": 0,
                "documentation": 0,
                "blog": 0,
                "social_media": 0,
                "academic_paper": 0,
                "marketing": 0
            }
            
            content_lower = content.lower()
            
            # Email indicators
            if re.search(r'\b(dear|hi|hello|sincerely|best regards|subject:)\b', content_lower):
                format_indicators["email"] += 3
            
            # Academic paper indicators
            if re.search(r'\b(abstract|introduction|methodology|conclusion|references)\b', content_lower):
                format_indicators["academic_paper"] += 3
            
            # Documentation indicators
            if re.search(r'\b(installation|configuration|api|endpoint|parameter)\b', content_lower):
                format_indicators["documentation"] += 2
            
            # Blog indicators
            if re.search(r'\b(blog|post|share|comment|subscribe)\b', content_lower):
                format_indicators["blog"] += 2
            
            # Social media indicators
            if len(content) < 280 or re.search(r'[#@]\w+', content):
                format_indicators["social_media"] += 3
            
            # Marketing indicators
            if re.search(r'\b(buy|purchase|offer|discount|sale|limited time)\b', content_lower):
                format_indicators["marketing"] += 2
            
            # Report indicators
            if re.search(r'\b(summary|findings|recommendations|data|statistics)\b', content_lower):
                format_indicators["report"] += 2
            
            # Default to article if no strong indicators
            if max(format_indicators.values()) == 0:
                return "article"
            
            return max(format_indicators, key=format_indicators.get)
            
        except Exception as e:
            logger.warning(f"Format detection failed: {e}")
            return "article"
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract key entities from the content."""
        try:
            if not self.nlp:
                return []
            
            doc = self.nlp(content)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                    entities.append(ent.text)
            
            # Remove duplicates and limit to top 10
            unique_entities = list(dict.fromkeys(entities))[:10]
            return unique_entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from the content."""
        try:
            if not self.nlp:
                return []
            
            doc = self.nlp(content)
            
            # Extract noun phrases as potential topics
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Filter and count
            topic_counts = Counter()
            for phrase in noun_phrases:
                phrase_clean = phrase.lower().strip()
                if len(phrase_clean.split()) <= 3 and len(phrase_clean) > 3:
                    topic_counts[phrase_clean] += 1
            
            # Return top 5 topics
            topics = [topic for topic, count in topic_counts.most_common(5)]
            return topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of the content."""
        try:
            if not self.sentiment_analyzer:
                return 0.0
            
            scores = self.sentiment_analyzer.polarity_scores(content)
            return scores['compound']  # Compound score ranges from -1 to 1
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_confidence(self, content: str, readability_scores: Dict[str, float]) -> float:
        """Calculate confidence in the analysis."""
        try:
            confidence_factors = []
            
            # Content length factor
            word_count = len(content.split())
            if word_count >= 50:
                confidence_factors.append(0.9)
            elif word_count >= 20:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Readability score availability
            valid_scores = [score for score in readability_scores.values() if score > 0]
            if len(valid_scores) >= 3:
                confidence_factors.append(0.9)
            elif len(valid_scores) >= 1:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
            
            # NLP model availability
            if self.nlp:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def process_state(self, state: AgentState) -> AgentState:
        """Process agent state and add analysis results."""
        try:
            logger.info(f"Style Analysis Agent processing request: {state.request_id}")
            
            # Analyze the content
            analysis = await self.analyze_content(state.original_content)
            
            # Update state
            state.content_analysis = analysis
            state.current_step = "analysis_completed"
            state.agent_outputs[self.name] = {
                "analysis": analysis.dict(),
                "timestamp": str(datetime.now()),
                "status": "completed"
            }
            
            logger.info(f"Style analysis completed for request: {state.request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Style Analysis Agent failed: {e}")
            state.errors.append(f"Style Analysis Agent error: {str(e)}")
            state.agent_outputs[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": str(datetime.now())
            }
            return state
