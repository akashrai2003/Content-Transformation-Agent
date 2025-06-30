"""Vector store implementation for content transformation knowledge base."""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid
import json
import hashlib
from datetime import datetime
import asyncio

from ..core.config_new import config
from ..core.models import StyleExample, ContentStyle, ComplexityLevel, ContentFormat
from ..core.usinc_client_new import create_usinc_client

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = None
        self.encoder = None
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size
        self.usinc_client = create_usinc_client(config)
        self._initialize()
        
    def _initialize(self):
        try:
            self.client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port
            )
            
            self.encoder = SentenceTransformer(config.sentence_transformer_model)
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            self.client = None
            self.encoder = None
    
    def _ensure_collection_exists(self):
        if not self.client:
            return
            
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        # Validate input text
        if not text or not text.strip():
            logger.warning("Empty or invalid text provided for embedding generation")
            return [0.0] * self.vector_size
        
        text = text.strip()
        if len(text) < 1:
            logger.warning("Text too short for embedding generation")
            return [0.0] * self.vector_size
            
        logger.info(f"Generating embedding for text: {text[:50]}...")
        try:
            embeddings = self.usinc_client.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                if embedding and len(embedding) == self.vector_size:
                    logger.info(f"Successfully generated embedding of size {len(embedding)}")
                    return embedding
            
            logger.warning(f"Failed to generate embedding for text: '{text[:50]}...', returning zero vector")
            return [0.0] * self.vector_size
        except Exception as e:
            logger.error(f"UltraSafe embedding failed, generating zero vector: {e}")
            return [0.0] * self.vector_size

    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        combined_hash = f"{content_hash}{metadata_hash}"
        
        # Generate UUID from has
        uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, combined_hash)
        return str(uuid_obj)
    
    def add_style_example(self, example: StyleExample) -> str:
        if not self.client or not self.encoder:
            logger.warning("Vector store not initialized, skipping add_style_example")
            return "dummy_id"
        
        try:
            embedding = self._generate_embedding(example.content)
            
            metadata = {
                "id": str(uuid.uuid4()),
                "type": "style_example",
                "style": example.style.value if example.style else "",
                "complexity": example.complexity.value if example.complexity else "",
                "format_type": example.format_type.value if example.format_type else "",
                "timestamp": datetime.now().isoformat(),
                "key_features": example.key_features or []
            }
            
            point_id = self._generate_id(example.content, metadata)
            metadata["id"] = point_id
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "content": example.content,
                            "metadata": metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added style example to vector store: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add style example: {e}")
            return "error_id"
    
    def add_transformation_example(self, source_content: str, transformed_content: str, metadata: Dict[str, Any]) -> str:
        if not self.client or not self.encoder:
            logger.warning("Vector store not initialized, skipping add_transformation_example")
            return "dummy_id"
        
        try:
            source_embedding = self._generate_embedding(source_content)
            
            combined_content = f"Source: {source_content}\n\nTransformed: {transformed_content}"
            combined_embedding = self._generate_embedding(combined_content)
            
            if "id" not in metadata:
                metadata["id"] = str(uuid.uuid4())
            
            metadata.update({
                "type": "transformation_example",
                "timestamp": datetime.now().isoformat()
            })
            
            point_id = self._generate_id(combined_content, metadata)
            metadata["id"] = point_id
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=combined_embedding,
                        payload={
                            "source_content": source_content,
                            "transformed_content": transformed_content,
                            "metadata": metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added transformation example to vector store: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add transformation example: {e}")
            return "error_id"
    
    def add_transformation_case(self, **case_data: Any) -> str:
        if not self.client or not self.encoder:
            logger.warning("Vector store not initialized, skipping add_transformation_case")
            return "dummy_id"
        
        try:
            original_content = case_data.get("original_content", "")
            transformed_content = case_data.get("transformed_content", "")
            
            combined_content = f"Original: {original_content}\n\nTransformed: {transformed_content}"
            embedding = self._generate_embedding(combined_content)
            
            metadata = {
                "id": str(uuid.uuid4()),
                "type": "transformation_case",
                "source_style": case_data.get("source_style"),
                "target_style": case_dataget("target_style"),
                "source_complexity": case_data.get("source_complexity"),
                "target_complexity": case_data.get("target_complexity"),
                "quality_score": case_data.get("quality_score"),
                "timestamp": datetime.now().isoformat()
            }
            
            if "transformation_steps" in case_data:
                metadata["transformation_steps"] = case_data["transformation_steps"]
            
            point_id = self._generate_id(combined_content, metadata)
            metadata["id"] = point_id
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "original_content": original_content,
                            "transformed_content": transformed_content,
                            "metadata": metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added transformation case to vector store: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add transformation case: {e}")
            return "error_id"
    
    def search_similar_examples(self, 
                             content: str, 
                             style: Optional[str] = None,
                             complexity: Optional[str] = None,
                             format_type: Optional[str] = None,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """Search for style examples based on query and filters."""
        if not self.client or not self.encoder:
            logger.warning("Vector store not initialized, returning empty results")
            return []
            
        try:
            query_embedding = self._generate_embedding(content)
            
            filter_conditions = []
            
            filter_conditions.append(
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="style_example")
                )
            )
            
            if style:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.style",
                        match=MatchValue(value=style)
                    )
                )
                
            if complexity:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.complexity",
                        match=MatchValue(value=complexity)
                    )
                )
                
            if format_type:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.format_type",
                        match=MatchValue(value=format_type)
                    )
                )
            
            query_filter = Filter(
                must=filter_conditions
            ) if filter_conditions else None
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter
            )
            
            # Process results
            results = []
            for result in search_results:
                payload = result.payload
                results.append({
                    "id": result.id,
                    "content": payload.get("content", ""),
                    "score": result.score,
                    "metadata": payload.get("metadata", {})
                })
            
            logger.info(f"Found {len(results)} style examples for query: {content}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search style examples: {e}")
            return []
    
    def search_transformation_cases(self,
                                     content: str,
                                     source_style: Optional[str] = None,
                                     target_style: Optional[str] = None,
                                     source_complexity: Optional[str] = None,
                                     target_complexity: Optional[str] = None,
                                     source_format: Optional[str] = None,
                                     target_format: Optional[str] = None,
                                     limit: int = 5) -> List[Dict[str, Any]]:
        if not self.client or not self.encoder:
            logger.warning("Vector store not initialized, returning empty results")
            return []
            
        try:
            query_embedding = self._generate_embedding(content)
            
            filter_conditions = []
            
            filter_conditions.append(
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="transformation_case")
                )
            )
            
            if source_style:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.source_style",
                        match=MatchValue(value=source_style)
                    )
                )
                
            if target_style:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.target_style",
                        match=MatchValue(value=target_style)
                    )
                )
                
            if source_complexity:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.source_complexity",
                        match=MatchValue(value=source_complexity)
                    )
                )
                
            if target_complexity:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.target_complexity",
                        match=MatchValue(value=target_complexity)
                    )
                )
                
            if source_format:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.source_format",
                        match=MatchValue(value=source_format)
                    )
                )
                
            if target_format:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.target_format",
                        match=MatchValue(value=target_format)
                    )
                )
            
            filter_obj = Filter(
                must=filter_conditions
            ) if filter_conditions else None
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_obj
            )
            
            results = []
            for result in search_results:
                payload = result.payload
                results.append({
                    "id": result.id,
                    "source_content": payload.get("source_content", ""),
                    "transformed_content": payload.get("transformed_content", ""),
                    "score": result.score,
                    "metadata": payload.get("metadata", {})
                })
            
            logger.info(f"Found {len(results)} transformation examples")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search transformation examples: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.client:
            logger.warning("Vector store not initialized, returning empty stats")
            return {}
            
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            style_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.type",
                        match=MatchValue(value="style_example")
                    )
                ]
            )
            style_count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=style_filter,
                exact=True
            )
            style_count = style_count_result.count if style_count_result else 0


            transformation_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.type",
                        match=MatchValue(value="transformation_case")
                    )
                ]
            )
            transformation_count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=transformation_filter,
                exact=True
            )
            transformation_count = transformation_count_result.count if transformation_count_result else 0

            return {
                "total_points": collection_info.points_count if collection_info and collection_info.points_count is not None else 0,
                "style_examples": style_count,
                "transformation_cases": transformation_count,
                "vector_size": self.vector_size,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
