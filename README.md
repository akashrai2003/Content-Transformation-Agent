# Content Transformation System

A multi-agent system that transforms content between different writing styles and complexity levels. Built with Python, LangGraph, and US.inc API.

## 🚀 Quick Start

### What You Need
- Python 3.8+
- US.inc API key
- Internet connection

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
   # Test with examples
   python main.py
   
   # Start web API
   python main.py --server
   ```

4. **Use the API**
   - Web interface: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## 📋 How It Works

### The System Components

The system uses 4 different agents that work together:

1. **Style Analysis Agent** - Figures out what style your text is currently in
2. **Planning Agent** - Decides how to change it to the style you want  
3. **Conversion Agent** - Actually transforms the text
4. **Quality Control Agent** - Checks if the result is good enough

### What It Can Transform

**Writing Styles:**
- Academic → Conversational
- Technical → Simplified  
- Formal → Casual
- And many more combinations

**Complexity Levels:**
- Elementary, Intermediate, Advanced, Expert

**Text Formats:**
- Articles, emails, documentation, reports, etc.

### Example
```
Input: "The research indicates that regular physical activity significantly improves cardiovascular health."
Style: Academic → Conversational, Advanced → Elementary

Output: "Studies show that exercising regularly is really good for your heart!"
```

## 🔧 Technical Details

### Built With
- **Python** - Main programming language
- **LangGraph** - Manages the workflow between agents
- **FastAPI** - Web API framework
- **US.inc API** - AI language model for text processing
- **Qdrant** - Vector database for storing examples

### Project Structure
```
content_transformation_system/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Configuration file
├── src/
│   ├── agents/            # The 4 agent files
│   ├── api/              # Web API code
│   ├── core/             # Core system files
│   └── rag/              # Knowledge base system
```

### Key Files
- `main.py` - Run this to start the application
- `src/api/main.py` - Web API endpoints
- `src/core/workflow.py` - Coordinates all the agents
- `src/rag/knowledge_base.py` - Stores transformation examples

## 🎯 Design Choices

### Why Multiple Agents?
I chose to split the work into 4 different agents instead of using one big system because:
- **Easier to debug** - Each agent handles one specific task
- **Better quality** - Each agent is specialized for its job
- **Easier to improve** - Can update one agent without breaking others

### Why US.inc API?
- **Consistent results** - Same quality every time
- **No local setup** - Runs entirely in the cloud
- **Easy to use** - Simple API calls

### Why LangGraph?
- **Professional tool** - Used by real companies
- **Handles errors well** - Automatically retries if something fails
- **Good for learning** - Shows how modern AI systems work

### Trade-offs Made
- **Internet required** - System needs internet to work (but most apps do these days)
- **API costs** - Uses paid API calls (but keeps infrastructure simple)
- **Multiple steps** - Takes longer than single-step solutions (but produces better quality)

## 🌐 API Endpoints

### Main Endpoints
- `POST /transform` - Transform your text
- `GET /health` - Check if system is working
- `GET /supported-options` - See available styles and formats
- `POST /analyze` - Analyze text characteristics

### Example API Usage
```bash
# Transform text
curl -X POST "http://localhost:8000/transform" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your text here",
    "target_style": "conversational",
    "target_complexity": "elementary"
  }'
```

## 🚧 Future Improvements

### Things I'd Add With More Time

**Short-term (1-2 weeks):**
- Support for more languages (Spanish, French, etc.)
- Better quality scoring
- Batch processing for multiple documents
- Result caching to make it faster

**Medium-term (1-2 months):**
- User accounts and login
- Save transformation history
- Custom style creation
- Better web interface

**Long-term (3+ months):**
- Mobile app
- Integration with Google Docs/Word
- AI that learns from user feedback
- Enterprise features for companies

### Known Issues
- Sometimes quality scores are lower than expected
- Processing can be slow for very long texts
- Requires internet connection

## 🛠️ Development

### Testing
```bash
# Test basic functionality
python main.py

# Start API server
python main.py --server

# Check API docs
# http://localhost:8000/docs
```

### Environment Variables
```env
# Required
USINC_API_KEY=your_api_key_here

# Optional (defaults work fine)
MIN_QUALITY_SCORE=0.5
API_HOST=0.0.0.0
API_PORT=8000
```

## 📝 About This Project

This project was built as a technical demonstration of multi-agent AI systems. It shows how to:
- Build a system with multiple AI agents working together
- Use modern workflow frameworks like LangGraph
- Create web APIs with FastAPI
- Integrate with cloud AI services

**Built with**: Python, LangGraph, FastAPI, US.inc API

---

*For more technical details, see IMPLEMENTATION.md*
