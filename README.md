# Content Transformation System

A multi-agent system that transforms content between different writing styles and complexity levels.

## Quick Start

### Requirements
- Python 3.8+
- US.inc API key

### Setup

1. **Install**
   ```bash
   git clone <repository-url>
   cd content_transformation_system
   pip install -r requirements.txt
   ```

2. **Configure**
   Create `.env` file:
   ```env
   USINC_API_KEY=your_api_key_here
   ```

3. **Run**
   ```bash
   python main.py
   ```

### How It Works

The system uses 4 agents:

1. **Style Analysis** - Analyzes your text
2. **Planning** - Creates transformation plan  
3. **Conversion** - Transforms the text
4. **Quality Control** - Validates results

### Example
```
Input: "The research indicates that regular physical activity significantly improves cardiovascular health."

Output: "Studies show that exercising regularly is really good for your heart!"
```


### Why LangGraph?
- **Professional tool** - Used by real companies
- **Handles errors well** - Automatically retries if something fails
- **Good for learning** - Shows how modern AI systems work

### Trade-offs Made
- **Internet required** - System needs internet to work (but most apps do these days)
- **API costs** - Uses paid API calls (but keeps infrastructure simple)
- **Multiple steps** - Takes longer than single-step solutions (but produces better quality)


## Features

**Styles:** Academic, Conversational, Technical, Simplified, Formal, Casual

**Complexity:** Elementary, Intermediate, Advanced, Expert

**Formats:** Articles, emails, reports, documentation

## Architecture

```
src/
├── agents/            # 4 agent files
├── api/              # FastAPI endpoints
├── core/             # Core models and workflow
└── rag/              # Knowledge base
```

## API Usage

```bash
curl -X POST "http://localhost:8000/transform" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your text here",
    "target_style": "conversational",
    "target_complexity": "elementary"
  }'
```

## Tech Stack

- Python
- LangGraph
- FastAPI
- US.inc API
- Qdrant
