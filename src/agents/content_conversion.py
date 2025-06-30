"""Content Conversion Agent for content transformation system."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import json

try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    HumanMessage = None
    SystemMessage = None

from ..core.models import AgentState, TransformationPlan, ContentAnalysis
from ..core.config_new import config
from ..core.usinc_client_new import create_usinc_client
from ..rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)


class ContentConversionAgent:
    def __init__(self):
        self.name = "ContentConversionAgent"
        self.usinc_client = create_usinc_client(config)

    async def transform_content(self,
                               original_content: str,
                               transformation_plan: TransformationPlan,
                               preserve_entities: bool = True,
                               preserve_facts: bool = True) -> str:
        try:
            logger.info("Starting content transformation")
            
            if not self.llm:
                return self._fallback_transformation(original_content, transformation_plan)
            
            # Get style examples and transformation cases
            style_examples = knowledge_base.get_style_examples(
                content=original_content,
                style=transformation_plan.target_style.value,
                complexity=transformation_plan.target_complexity.value,
                limit=3
            )
            
            transformation_examples = knowledge_base.get_transformation_examples(
                content=original_content,
                source_style=transformation_plan.source_analysis.style,
                target_style=transformation_plan.target_style.value,
                limit=2
            )
            
            prompt = self._create_transformation_prompt(
                original_content=original_content,
                transformation_plan=transformation_plan,
                style_examples=style_examples,
                transformation_examples=transformation_examples,
                preserve_entities=preserve_entities,
                preserve_facts=preserve_facts
            )
            
            #  transformation using LLM
            transformed_content = await self._execute_llm_transformation(prompt)
            
            # post-processing
            final_content = self._post_process_content(
                transformed_content,
                transformation_plan,
                original_content
            )
            
            logger.info("Content transformation completed successfully")
            return final_content
            
        except Exception as e:
            logger.error(f"Content transformation failed: {e}")
            return self._fallback_transformation(original_content, transformation_plan)
    
    def _create_transformation_prompt(self,
                                    original_content: str,
                                    transformation_plan: TransformationPlan,
                                    style_examples: List[Dict[str, Any]],
                                    transformation_examples: List[Dict[str, Any]],
                                    preserve_entities: bool,
                                    preserve_facts: bool) -> List[Any]:
        system_prompt = f"""You are an expert content transformation agent. Your task is to transform the given content according to specific requirements while maintaining accuracy and quality.

TRANSFORMATION REQUIREMENTS:
- Source Style: {transformation_plan.source_analysis.style}
- Target Style: {transformation_plan.target_style.value}
- Target Complexity: {transformation_plan.target_complexity.value}
- Target Format: {transformation_plan.target_format.value}

TRANSFORMATION STEPS:
"""
        
        for i, step in enumerate(transformation_plan.transformation_steps, 1):
            system_prompt += f"{i}. {step.get('title', 'Step')}: {step.get('description', '')}\n"
            if step.get('actions'):
                for action in step['actions']:
                    system_prompt += f"   - {action}\n"
        
        # Add requirements
        system_prompt += f"\nPRESERVATION REQUIREMENTS:\n"
        for req in transformation_plan.preservation_requirements:
            system_prompt += f"- {req.replace('_', ' ').title()}\n"
        
        if preserve_entities:
            system_prompt += "- Preserve all named entities (people, places, organizations)\n"
        
        if preserve_facts:
            system_prompt += "- Maintain factual accuracy and key information\n"
        
        # style examples
        if style_examples:
            system_prompt += f"\nTARGET STYLE EXAMPLES ({transformation_plan.target_style.value}):\n"
            for i, example in enumerate(style_examples[:2], 1):
                system_prompt += f"Example {i}: {example.get('content', '')[:200]}...\n"
                if example.get('key_features'):
                    system_prompt += f"Key features: {', '.join(example['key_features'][:3])}\n"
        
        # transformation examples
        if transformation_examples:
            system_prompt += f"\nSIMILAR TRANSFORMATION EXAMPLES:\n"
            for i, example in enumerate(transformation_examples[:1], 1):
                system_prompt += f"Example {i}:\n"
                system_prompt += f"Original: {example.get('original_content', '')[:150]}...\n"
                system_prompt += f"Transformed: {example.get('transformed_content', '')[:150]}...\n"
        
        # quality targets
        system_prompt += f"\nQUALITY TARGETS:\n"
        for metric, target in transformation_plan.quality_targets.items():
            system_prompt += f"- {metric.replace('_', ' ').title()}: {target:.2f}\n"
        
        # Final instructions
        system_prompt += f"""
INSTRUCTIONS:
1. Transform the content following the steps above
2. Maintain the original meaning and key information
3. Adapt the style, complexity, and format as specified
4. Ensure the output sounds natural and coherent
5. Do not add information not present in the original
6. Preserve all factual details and entities
7. Return only the transformed content, no explanations

Target audience: {transformation_plan.target_complexity.value} level readers
Expected output length: Similar to original (±20%)
"""
        
        human_prompt = f"""Please transform the following content according to the requirements above:

ORIGINAL CONTENT:
{original_content}

TRANSFORMED CONTENT:"""
        
        if SystemMessage and HumanMessage:            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        else:
            return [{"role": "system", "content": system_prompt}, 
                   {"role": "user", "content": human_prompt}]
    
    async def _execute_llm_transformation(self, prompt: List[Any]) -> str:
        """Execute the transformation using the LLM."""
        try:
            messages = []
            if isinstance(prompt[0], dict):
                messages = prompt
            else:
                for msg in prompt:
                    if hasattr(msg, 'content'):
                        if hasattr(msg, 'type') and msg.type == 'system':
                            messages.append({"role": "system", "content": msg.content})
                        else:
                            messages.append({"role": "user", "content": msg.content})
                    else:
                        messages.append({"role": "user", "content": str(msg)})
            
            try:
                response = await self.usinc_client.async_chat_completion(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                
                # Extract content from US.inc response
                if "choices" in response and len(response["choices"]) > 0:
                    transformed_content = response["choices"][0]["message"]["content"].strip()
                    # Clean up the response
                    transformed_content = self._clean_llm_response(transformed_content)
                    return transformed_content
                else:
                    raise Exception("No content in US.inc response")
                    
            except Exception as usinc_error:
                logger.warning(f"US.inc API failed: {usinc_error}")
            
        except Exception as e:
            logger.error(f"LLM transformation failed: {e}")
            raise
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean up the LLM response."""
        cleaned = response
        
        artifacts = [
            r"^(Here is|Here's) the transformed content:?\s*",
            r"^(The )?transformed content is:?\s*",
            r"^(Based on|Following) the requirements:?\s*",
            r"\*\*[^*]+\*\*",
            r"```[^`]*```",
        ]
        
        for artifact in artifacts:
            cleaned = re.sub(artifact, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned
    
    def _post_process_content(self,
                            transformed_content: str,
                            transformation_plan: TransformationPlan,
                            original_content: str) -> str:
        try:
            processed_content = transformed_content
            processed_content = self._fix_capitalization(processed_content)
            processed_content = self._fix_formatting(processed_content)
            if "named_entities" in transformation_plan.preservation_requirements:
                processed_content = self._ensure_entity_preservation(
                    processed_content,
                    original_content,
                    transformation_plan.source_analysis.key_entities
                )
            
            # post-processing
            processed_content = self._apply_format_specific_processing(
                processed_content,
                transformation_plan.target_format.value
            )
            
            return processed_content
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return transformed_content
    
    def _fix_capitalization(self, content: str) -> str:
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?]+)', content)
        fixed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                fixed_sentences.append(sentence)
            else:
                fixed_sentences.append(sentence)
        
        return ''.join(fixed_sentences)
    
    def _fix_formatting(self, content: str) -> str:
        # Fix spacing around punctuation
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        
        # Fix multiple spaces
        content = re.sub(r'\s+', ' ', content)
        
        # Fix paragraph spacing
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def _ensure_entity_preservation(self,
                                  transformed_content: str,
                                  original_content: str,
                                  key_entities: List[str]) -> str:
        if not key_entities:
            return transformed_content
        
        missing_entities = []
        for entity in key_entities:
            if entity.lower() not in transformed_content.lower():
                missing_entities.append(entity)
        
        if missing_entities:
            logger.warning(f"Missing entities detected: {missing_entities}")
        return transformed_content
    
    def _apply_format_specific_processing(self, content: str, target_format: str) -> str:
        if target_format == "email":
            if not content.startswith(("Dear", "Hi", "Hello")):
                content = "Dear Reader,\n\n" + content
            if not content.endswith(("Sincerely", "Best regards", "Thanks")):
                content = content + "\n\nBest regards"
        
        elif target_format == "social_media":
            if len(content) > 280:
                sentences = content.split('.')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + ".") <= 270:
                        truncated += sentence + "."
                    else:
                        break
                content = truncated + "..."
        
        elif target_format == "blog":
            if not content.strip().endswith(('?', '!', '.')):
                content = content.strip() + "."
        
        return content
    
    def _fallback_transformation(self, original_content: str, transformation_plan: TransformationPlan) -> str:
        logger.warning("Using fallback transformation method TODO: Implement fallback logic")
        pass
    
    async def process_state(self, state: AgentState) -> AgentState:
        try:
            logger.info(f"Content Conversion Agent processing request: {state.request_id}")
            
            if not state.transformation_plan:
                raise ValueError("Transformation plan required for conversion")
            
            # Execute transformation
            transformed_content = await self.transform_content(
                original_content=state.original_content,
                transformation_plan=state.transformation_plan,
                preserve_entities=state.preserve_entities,
                preserve_facts=state.preserve_facts
            )
            
            # Update state
            state.transformed_content = transformed_content
            state.current_step = "conversion_completed"
            state.agent_outputs[self.name] = {
                "transformed_content": transformed_content,
                "timestamp": str(datetime.now()),
                "status": "completed",
                "word_count_original": len(state.original_content.split()),
                "word_count_transformed": len(transformed_content.split())
            }
            
            logger.info(f"Content conversion completed for request: {state.request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Content Conversion Agent failed: {e}")
            state.errors.append(f"Content Conversion Agent error: {str(e)}")
            state.agent_outputs[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": str(datetime.now())
            }
            return state
