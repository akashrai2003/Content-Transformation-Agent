"""US.inc API client for chat completions, embeddings, and reranking."""

import requests
import json
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from .config_new import Config


class USIncClient:
    """Client for US.inc API services."""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.usinc_api_key
        self.base_url = config.usinc_base_url
        
        # Just store the base URL, we'll construct the full endpoint URLs when needed
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate chat completion using US.inc API."""
        
        url = f"{self.base_url}/hiring/chat/completions"
        
        payload = {
            "model": self.config.usinc_chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"US.inc chat completion error: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using US.inc API."""
        
        url = f"{self.base_url}/hiring/embed/embeddings"
        embeddings = []
        
        for text in texts:
            # Validate text input
            if not text or not text.strip():
                print(f"Warning: Empty text provided for embedding, returning zero vector")
                embeddings.append([0.0] * 1024)  # Default embedding size
                continue
                
            # Clean text
            text = text.strip()
            
            payload = {
                "model": self.config.usinc_embed_model,
                "input": text  # API accepts either string or array
            }
            
            # Try up to 3 times with increasing delays
            for attempt in range(3):
                try:
                    response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract embedding from response
                    if "result" in result and "data" in result["result"]:
                        embedding = result["result"]["data"][0]["embedding"]
                        embeddings.append(embedding)
                        break
                    else:
                        print(f"Warning: Unexpected response structure from embedding API")
                        # Fallback: return zeros if embedding extraction fails
                        embeddings.append([0.0] * 1024)  # Default embedding size
                        break
                        
                except requests.exceptions.RequestException as e:
                    if attempt == 2:  # Last attempt
                        print(f"Warning: US.inc embedding failed after 3 attempts for text '{text[:50]}...': {str(e)}")
                        embeddings.append([0.0] * 1024)
                    else:
                        # Wait before retry (exponential backoff)
                        import time
                        wait_time = (2 ** attempt)  # 1s, 2s, 4s
                        print(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
        
        return embeddings
    
    def rerank(self, query: str, texts: List[str]) -> List[Dict[str, Any]]:
        """Rerank texts based on query relevance using US.inc API."""
        
        url = f"{self.base_url}/hiring/embed/reranker"
        
        payload = {
            "model": self.config.usinc_rerank_model,
            "query": query,
            "texts": texts
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract reranking results
            if "result" in result and "data" in result["result"]:
                return result["result"]["data"]
            else:
                # Fallback: return original order with default scores
                return [{"index": i, "text": text, "score": 0.5} for i, text in enumerate(texts)]
                
        except requests.exceptions.RequestException as e:
            print(f"Warning: US.inc rerank error: {str(e)}")
            # Fallback: return original order with default scores
            return [{"index": i, "text": text, "score": 0.5} for i, text in enumerate(texts)]
    
    async def async_chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Async version of chat completion."""
        
        url = f"{self.base_url}/hiring/chat/completions"
        
        payload = {
            "model": self.config.usinc_chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                raise Exception(f"US.inc async chat completion error: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test if the API connection is working."""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completion(test_messages, max_tokens=10)
            return "choices" in result and len(result["choices"]) > 0
        except Exception as e:
            print(f"US.inc API connection test failed: {str(e)}")
            return False


def create_usinc_client(config: Config) -> USIncClient:
    """Factory function to create US.inc client."""
    return USIncClient(config)
