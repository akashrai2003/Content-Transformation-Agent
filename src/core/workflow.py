"""LangGraph workflow orchestrator for content transformation system."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.graph import CompiledGraph
except ImportError:
    StateGraph = None
    END = None
    CompiledGraph = None

from ..core.models import (
    AgentState, TransformationRequest, TransformationResult, 
    TransformationStatus, ContentStyle, ComplexityLevel, ContentFormat
)
from ..agents.style_analysis import StyleAnalysisAgent
from ..agents.transformation_planning import TransformationPlanningAgent
from ..agents.content_conversion import ContentConversionAgent
from ..agents.quality_control import QualityControlAgent
from ..rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)


class ContentTransformationWorkflow:
    def __init__(self):
        self.name = "ContentTransformationWorkflow"
        
        self.style_agent = StyleAnalysisAgent()
        self.planning_agent = TransformationPlanningAgent()
        self.conversion_agent = ContentConversionAgent()
        self.quality_agent = QualityControlAgent()
        
        self.workflow = None
        self._build_workflow()
    
    def _build_workflow(self):
        try:
            if not StateGraph:
                logger.warning("LangGraph not available, using fallback workflow")
                return
            
            workflow = StateGraph(AgentState)
            
            workflow.add_node("style_analysis", self._style_analysis_node)
            workflow.add_node("transformation_planning", self._transformation_planning_node)
            workflow.add_node("content_conversion", self._content_conversion_node)
            workflow.add_node("quality_control", self._quality_control_node)
            workflow.add_node("retry_conversion", self._retry_conversion_node)
            workflow.add_node("finalize", self._finalize_node)
            
            workflow.set_entry_point("style_analysis")
            
            workflow.add_edge("style_analysis", "transformation_planning")
            workflow.add_edge("transformation_planning", "content_conversion")
            workflow.add_edge("content_conversion", "quality_control")
            
            workflow.add_conditional_edges(
                "quality_control",
                self._should_retry,
                {
                    "retry": "retry_conversion",
                    "finalize": "finalize",
                    "end": END
                }
            )
            
            workflow.add_edge("retry_conversion", "quality_control")
            workflow.add_edge("finalize", END)
            
            self.workflow = workflow.compile()
            logger.info("LangGraph workflow built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            self.workflow = None
    
    async def _style_analysis_node(self, state: AgentState) -> AgentState:
        return await self.style_agent.process_state(state)
    
    async def _transformation_planning_node(self, state: AgentState) -> AgentState:
        return await self.planning_agent.process_state(state)
    
    async def _content_conversion_node(self, state: AgentState) -> AgentState:
        return await self.conversion_agent.process_state(state)
    
    async def _quality_control_node(self, state: AgentState) -> AgentState:
        return await self.quality_agent.process_state(state)
    
    async def _retry_conversion_node(self, state: AgentState) -> AgentState:
        logger.info(f"Retrying conversion for request: {state.request_id}")
        state.retry_count += 1
        return await self.conversion_agent.process_state(state)
    
    async def _finalize_node(self, state: AgentState) -> AgentState:
        state.current_step = "completed"
        state.status = TransformationStatus.COMPLETED
        logger.info(f"Workflow completed for request: {state.request_id}")
        return state
    
    def _should_retry(self, state: AgentState) -> str:
        if state.quality_assessment and state.quality_assessment.overall_score < 0.6:
            if state.retry_count < 2:
                return "retry"
            else:
                return "finalize"
        else:
            return "finalize"
    
    async def transform_content(self, request: TransformationRequest) -> TransformationResult:
        try:
            logger.info(f"Starting content transformation: {request.request_id}")
            
            await knowledge_base.initialize()
            
            initial_state = AgentState(
                request_id=request.request_id,
                original_content=request.content,
                target_style=request.target_style,
                target_complexity=request.target_complexity,
                target_format=request.target_format,
                user_instructions=request.user_instructions,
                preserve_entities=request.preserve_entities,
                preserve_facts=request.preserve_facts,
                current_step="initialized",
                status=TransformationStatus.IN_PROGRESS,
                agent_outputs={},
                errors=[],
                retry_count=0
            )
            
            if self.workflow:
                try:
                    final_state = await self.workflow.ainvoke(initial_state)
                except Exception as workflow_error:
                    logger.warning(f"LangGraph workflow failed: {workflow_error}")
                    final_state = await self._fallback_execution(initial_state)
            else:
                final_state = await self._fallback_execution(initial_state)
            
            result = TransformationResult(
                request_id=request.request_id,
                original_content=request.content,
                transformed_content=final_state.transformed_content or "",
                source_analysis=final_state.source_analysis,
                transformation_plan=final_state.transformation_plan,
                quality_assessment=final_state.quality_assessment,
                status=final_state.status,
                agent_outputs=final_state.agent_outputs,
                errors=final_state.errors,
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            logger.info(f"Content transformation completed: {request.request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Content transformation failed: {e}")
            return TransformationResult(
                request_id=request.request_id,
                original_content=request.content,
                transformed_content="",
                source_analysis=None,
                transformation_plan=None,
                quality_assessment=None,
                status=TransformationStatus.FAILED,
                agent_outputs={},
                errors=[str(e)],
                processing_time=0.0,
                timestamp=datetime.now()
            )
    
    async def _fallback_execution(self, state: AgentState) -> AgentState:
        try:
            logger.info("Using fallback sequential execution")
            
            state = await self.style_agent.process_state(state)
            if state.errors:
                logger.warning(f"Style analysis errors: {state.errors}")
            
            state = await self.planning_agent.process_state(state)
            if state.errors:
                logger.warning(f"Planning errors: {state.errors}")
            
            state = await self.conversion_agent.process_state(state)
            if state.errors:
                logger.warning(f"Conversion errors: {state.errors}")
            
            state = await self.quality_agent.process_state(state)
            if state.errors:
                logger.warning(f"Quality control errors: {state.errors}")
            
            if (state.quality_assessment and 
                state.quality_assessment.overall_score < 0.6 and 
                state.retry_count < 2):
                logger.info("Retrying conversion due to low quality score")
                state.retry_count += 1
                state = await self.conversion_agent.process_state(state)
                state = await self.quality_agent.process_state(state)
            
            state.status = TransformationStatus.COMPLETED
            state.current_step = "completed"
            
            return state
            
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            state.errors.append(f"Fallback execution error: {str(e)}")
            state.status = TransformationStatus.FAILED
            return state
    
    async def batch_transform(self, requests: list[TransformationRequest]) -> list[TransformationResult]:
        results = []
        
        for request in requests:
            try:
                result = await self.transform_content(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch transformation failed for {request.request_id}: {e}")
                results.append(TransformationResult(
                    request_id=request.request_id,
                    original_content=request.content,
                    transformed_content="",
                    source_analysis=None,
                    transformation_plan=None,
                    quality_assessment=None,
                    status=TransformationStatus.FAILED,
                    agent_outputs={},
                    errors=[str(e)],
                    processing_time=0.0,
                    timestamp=datetime.now()
                ))
        
        return results
    
    def get_workflow_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "workflow_available": self.workflow is not None,
            "langgraph_available": StateGraph is not None,
            "agents": {
                "style_analysis": self.style_agent.name,
                "transformation_planning": self.planning_agent.name,
                "content_conversion": self.conversion_agent.name,
                "quality_control": self.quality_agent.name
            }
        }
