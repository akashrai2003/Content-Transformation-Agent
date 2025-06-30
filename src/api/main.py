"""FastAPI application for content transformation system."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uvicorn

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    FastAPI = None
    HTTPException = None
    BackgroundTasks = None
    CORSMiddleware = None
    JSONResponse = None

from ..core.models import (
    TransformationRequest, TransformationResult, UserFeedback,
    ContentStyle, ComplexityLevel, ContentFormat, StyleExample
)
from ..core.config import config
from ..core.workflow import workflow_orchestrator
from ..rag.knowledge_base import knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if FastAPI:
    app = FastAPI(
        title="Content Transformation System",
        description="A multi-agent system for transforming content between formats, styles, and complexity levels",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None
    logger.error("FastAPI not available")


@app.get("/")
async def root():
    return {
        "message": "Content Transformation System API",
        "documentation": "/docs",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    try:
        kb_stats = knowledge_base.get_statistics()
        
        # Check workflow status
        workflow_info = workflow_orchestrator.get_supported_transformations()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "knowledge_base": {
                    "status": "operational",
                    "documents": kb_stats.get("total_documents", 0)
                },
                "workflow": {
                    "status": "operational",
                    "langgraph_available": workflow_info["workflow_capabilities"]["langgraph_available"]
                },
                "agents": {
                    "count": len(workflow_info["workflow_capabilities"]["agents"]),
                    "agents": workflow_info["workflow_capabilities"]["agents"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/transform", response_model=TransformationResult)
async def transform_content(request: TransformationRequest):
    """Transform content according to specified parameters."""
    try:
        logger.info(f"Received transformation request: {request.target_style} style")
        
        # Validate request
        if not request.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if len(request.content) > 10000:
            raise HTTPException(status_code=400, detail="Content too long (max 10,000 characters)")
        
        # Initialize knowledge base if needed
        if not knowledge_base.initialized:
            await knowledge_base.initialize()
        
        # Execute transformation
        result = await workflow_orchestrator.execute_transformation(request)
        
        logger.info(f"Transformation completed: {result.request_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")


@app.get("/supported-options")
async def get_supported_options():
    """Get supported transformation options."""
    try:
        return {
            "styles": [style.value for style in ContentStyle],
            "complexity_levels": [level.value for level in ComplexityLevel],
            "formats": [format_type.value for format_type in ContentFormat]
        }
    except Exception as e:
        logger.error(f"Failed to get supported options: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported options: {str(e)}")


@app.get("/system/status")
async def get_system_status():
    """Get detailed system status."""
    try:
        # Check knowledge base status
        kb_stats = knowledge_base.get_statistics()
        kb_validation = knowledge_base.validate_knowledge_base()
        
        # Check workflow status
        workflow_info = workflow_orchestrator.get_supported_transformations()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "knowledge_base": {
                    "status": "operational" if kb_validation["is_valid"] else "issues_detected",
                    "documents": kb_stats.get("total_documents", 0),
                    "style_examples": kb_stats.get("style_examples", 0),
                    "transformation_cases": kb_stats.get("transformation_cases", 0),
                    "validation": kb_validation
                },
                "workflow": {
                    "status": "operational",
                    "langgraph_available": workflow_info["workflow_capabilities"]["langgraph_available"],
                    "agents": workflow_info["workflow_capabilities"]["agents"]
                },
                "vector_store": {
                    "status": "operational",
                    "collection": "content_transformation_knowledge"
                }
            }
        }
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/styles")
async def get_available_styles():
    """Get information about available styles and formats."""
    try:
        return workflow_orchestrator.get_supported_transformations()
    except Exception as e:
        logger.error(f"Failed to get styles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get styles: {str(e)}")


@app.get("/styles/{style}/examples")
async def get_style_examples(style: str, complexity: Optional[str] = None, limit: int = 5):
    """Get examples for a specific style."""
    try:
        # Validate style
        if style not in [s.value for s in ContentStyle]:
            raise HTTPException(status_code=400, detail=f"Invalid style: {style}")
        
        # Validate complexity if provided
        if complexity and complexity not in [c.value for c in ComplexityLevel]:
            raise HTTPException(status_code=400, detail=f"Invalid complexity: {complexity}")
        
        # Get examples
        examples = knowledge_base.get_style_examples(
            content="",  # Empty content for general examples
            style=style,
            complexity=complexity or "intermediate",
            limit=limit
        )
        
        return {
            "style": style,
            "complexity": complexity,
            "examples": examples,
            "count": len(examples)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get style examples: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get style examples: {str(e)}")


@app.post("/analyze")
async def analyze_content(content_data: Dict[str, str]):
    """Analyze content characteristics."""
    try:
        content = content_data.get("content", "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Use style analysis agent
        from ..agents.style_analysis import StyleAnalysisAgent
        agent = StyleAnalysisAgent()
        
        analysis = await agent.analyze_content(content)
        
        return {
            "analysis": analysis.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")


@app.get("/transformations/{transformation_id}/status")
async def get_transformation_status(transformation_id: str):
    """Get status of a transformation (placeholder for async transformations)."""
    try:
        status = await workflow_orchestrator.get_workflow_status(transformation_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get transformation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get transformation status: {str(e)}")


@app.post("/feedback")
async def submit_feedback(feedback: UserFeedback, background_tasks: BackgroundTasks):
    """Submit feedback for a transformation result."""
    try:
        # In a real implementation, you would store feedback and use it for learning
        background_tasks.add_task(process_feedback, feedback)
        
        return {
            "message": "Feedback received successfully",
            "feedback_id": feedback.result_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.post("/knowledge-base/examples")
async def add_style_example(example: StyleExample):
    """Add a new style example to the knowledge base."""
    try:
        if not knowledge_base.initialized:
            await knowledge_base.initialize()
        
        example_id = knowledge_base.add_style_example(example)
        
        return {
            "message": "Style example added successfully",
            "example_id": example_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to add style example: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add style example: {str(e)}")


@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics."""
    try:
        stats = knowledge_base.get_statistics()
        validation = knowledge_base.validate_knowledge_base()
        
        return {
            "statistics": stats,
            "validation": validation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base stats: {str(e)}")


@app.post("/knowledge-base/initialize")
async def initialize_knowledge_base():
    """Initialize or reinitialize the knowledge base."""
    try:
        await knowledge_base.initialize()
        
        return {
            "message": "Knowledge base initialized successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize knowledge base: {str(e)}")


@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents."""
    try:
        agents_info = workflow_orchestrator.get_supported_transformations()["workflow_capabilities"]["agents"]
        
        # Test each agent (basic check)
        agent_status = {}
        
        # Style Analysis Agent
        try:
            from ..agents.style_analysis import StyleAnalysisAgent
            StyleAnalysisAgent()
            agent_status["StyleAnalysisAgent"] = "operational"
        except Exception as e:
            agent_status["StyleAnalysisAgent"] = f"error: {str(e)}"
        
        # Transformation Planning Agent
        try:
            from ..agents.transformation_planning import TransformationPlanningAgent
            TransformationPlanningAgent()
            agent_status["TransformationPlanningAgent"] = "operational"
        except Exception as e:
            agent_status["TransformationPlanningAgent"] = f"error: {str(e)}"
        
        # Content Conversion Agent
        try:
            from ..agents.content_conversion import ContentConversionAgent
            ContentConversionAgent()
            agent_status["ContentConversionAgent"] = "operational"
        except Exception as e:
            agent_status["ContentConversionAgent"] = f"error: {str(e)}"
        
        # Quality Control Agent
        try:
            from ..agents.quality_control import QualityControlAgent
            QualityControlAgent()
            agent_status["QualityControlAgent"] = "operational"
        except Exception as e:
            agent_status["QualityControlAgent"] = f"error: {str(e)}"
        
        return {
            "agents": agent_status,
            "total_agents": len(agent_status),
            "operational_agents": len([s for s in agent_status.values() if s == "operational"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agents status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agents status: {str(e)}")


async def process_feedback(feedback: UserFeedback):
    """Process user feedback in the background."""
    try:
        logger.info(f"Processing feedback for result: {feedback.result_id}")
        
        # In a real implementation, you would:
        # 1. Store the feedback in a database
        # 2. Use it to improve the system
        # 3. Update agent performance metrics
        # 4. Retrain models if necessary
        
        # For now, just log the feedback
        logger.info(f"Feedback: Rating={feedback.rating}, Comments={feedback.comments}")
        
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not app:
        raise RuntimeError("FastAPI not available")
    
    return app


def run_server():
    """Run the API server."""
    if not app:
        raise RuntimeError("FastAPI not available")
    
    logger.info(f"Starting Content Transformation System API server")
    logger.info(f"Server will be available at: http://{config.api_host}:{config.api_port}")
    logger.info(f"API documentation: http://{config.api_host}:{config.api_port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
