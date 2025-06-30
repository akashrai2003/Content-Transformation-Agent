"""Main entry point for the Content Transformation System."""

import asyncio
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config_new import config
from src.rag.knowledge_base import knowledge_base
from src.core.workflow import workflow_orchestrator
from src.core.models import TransformationRequest, ContentStyle, ComplexityLevel, ContentFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentTransformationSystem:
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return
        
        try:
            logger.info("Initializing Content Transformation System...")
            
            # Initialize knowledge base
            logger.info("Initializing knowledge base...")
            await knowledge_base.initialize()
            
            # Validate system components
            logger.info("Validating system components...")
            validation = await self._validate_system()
            
            if not validation["is_valid"]:
                logger.warning(f"System validation issues: {validation['issues']}")
            
            self.initialized = True
            logger.info("Content Transformation System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def transform_content(self,
                               content: str,
                               target_style: str,
                               target_complexity: str = "intermediate",
                               target_format: str = "article",
                               user_instructions: str = None,
                               preserve_entities: bool = True,
                               preserve_facts: bool = True,
                               quality_threshold: float = 0.6) -> dict:
        try:
            if not self.initialized:
                logger.info("System not initialized. Initializing ...")
                await self.initialize()
            
            # Create transformation request
            request = TransformationRequest(
                content=content,
                target_style=ContentStyle(target_style),
                target_complexity=ComplexityLevel(target_complexity),
                target_format=ContentFormat(target_format),
                user_instructions=user_instructions,
                preserve_entities=preserve_entities,
                preserve_facts=preserve_facts,
                quality_threshold=quality_threshold
            )
            
            # Execute transformation
            result = await workflow_orchestrator.execute_transformation(request)
            
            return {
                "success": result.status.value == "completed",
                "request_id": result.request_id,
                "original_content": result.original_content,
                "transformed_content": result.transformed_content,
                "processing_time": result.processing_time,
                "quality_score": result.quality_assessment.overall_score if result.quality_assessment else None,
                "issues": result.quality_assessment.issues_found if result.quality_assessment else [],
                "suggestions": result.quality_assessment.suggestions if result.quality_assessment else [],
                "error": result.error_message
            }
            
        except Exception as e:
            logger.error(f"Content transformation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": None,
                "original_content": content,
                "transformed_content": None
            }
    
    async def analyze_content(self, content: str) -> dict:
        try:
            if not self.initialized:
                await self.initialize()
            
            from src.agents.style_analysis import StyleAnalysisAgent
            agent = StyleAnalysisAgent()
            
            analysis = await agent.analyze_content(content)
            
            return {
                "success": True,
                "analysis": analysis.dict()
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_supported_options(self) -> dict:
        return {
            "styles": [style.value for style in ContentStyle],
            "complexity_levels": [level.value for level in ComplexityLevel],
            "formats": [format_type.value for format_type in ContentFormat]
        }
    
    async def _validate_system(self) -> dict:
        validation = {
            "is_valid": True,
            "issues": [],
            "components": {}
        }
        
        try:
            # Validate knowledge base
            kb_validation = knowledge_base.validate_knowledge_base()
            validation["components"]["knowledge_base"] = kb_validation
            
            if not kb_validation["is_valid"]:
                validation["is_valid"] = False
                validation["issues"].extend(kb_validation["issues"])
            
            # Validate workflow
            workflow_info = workflow_orchestrator.get_supported_transformations()
            validation["components"]["workflow"] = {
                "langgraph_available": workflow_info["workflow_capabilities"]["langgraph_available"],
                "agents_count": len(workflow_info["workflow_capabilities"]["agents"])
            }
            
            if self.initialized:
                try:
                    test_result = await self.transform_content(
                        content="This is a test sentence.",
                        target_style="conversational",
                        target_complexity="elementary"
                    )
                    validation["components"]["transformation_test"] = {
                        "success": test_result["success"],
                        "error": test_result.get("error")
                    }
                    
                    if not test_result["success"]:
                        validation["issues"].append(f"Transformation test failed: {test_result.get('error')}")
                
                except Exception as e:
                    validation["issues"].append(f"Transformation test error: {str(e)}")
            else:
                validation["components"]["transformation_test"] = {
                    "success": False,
                    "error": "System not initialized - skipping transformation test"
                }
            
        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"System validation error: {str(e)}")
        
        return validation
    
    async def run_api_server(self):
        try:
            if not self.initialized:
                await self.initialize()
            
            from src.api.main import run_server
            run_server()
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise


async def main():
    """Main function for command-line usage."""
    print("Content Transformation System")
    print("=" * 50)
    
    # Initialize system
    system = ContentTransformationSystem()
    await system.initialize()
    
    options = system.get_supported_options()
    print(f"Supported styles: {', '.join(options['styles'])}")
    print(f"Supported complexity levels: {', '.join(options['complexity_levels'])}")
    print(f"Supported formats: {', '.join(options['formats'])}")
    print()
    
    # Example transformations
    examples = [
        {
            "content": "The research indicates that regular physical activity significantly improves cardiovascular health outcomes and reduces the risk of chronic diseases.",
            "target_style": "conversational",
            "target_complexity": "elementary",
            "description": "Academic to conversational, simplified"
        },
        {
            "content": "Hey everyone! Climate change is super scary, but we can totally fix it if we all work together!",
            "target_style": "formal",
            "target_complexity": "advanced",
            "description": "Conversational to formal, more complex"
        },
        {
            "content": "To configure the API endpoint, first set your authentication credentials in the config file, then initialize the client with the proper parameters.",
            "target_style": "simplified",
            "target_complexity": "elementary",
            "description": "Technical to simplified"
        }
    ]
    
    print("Running transformations...")
    print("-" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['description']}")
        print(f"Original: {example['content']}")
        
        result = await system.transform_content(
            content=example["content"],
            target_style=example["target_style"],
            target_complexity=example["target_complexity"]
        )
        
        if result["success"]:
            print(f"Transformed: {result['transformed_content']}")
            print(f"Quality Score: {result['quality_score']:.3f}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            
            if result["issues"]:
                print(f"Issues: {', '.join(result['issues'])}")
            
            if result["suggestions"]:
                print(f"Suggestions: {', '.join(result['suggestions'])}")
        else:
            print(f"Transformation failed: {result['error']}")
        
        print("-" * 30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run API server
        system = ContentTransformationSystem()
        asyncio.run(system.run_api_server())
    else:
        # Run examples
        asyncio.run(main())
