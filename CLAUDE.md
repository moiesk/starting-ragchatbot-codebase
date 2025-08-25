# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
./run.sh
```
Or manually:
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Add new dependencies:**
uv add package_name

**Code Quality Commands:**
```bash
# Format code with Black
./scripts/format.sh

# Check code formatting
./scripts/check-format.sh

# Run all quality checks (formatting + tests)
./scripts/quality-check.sh
```

**Environment setup:**
Create `.env` file in root with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** for course materials with a FastAPI backend and vanilla HTML/CSS/JS frontend.

### Core Architecture Pattern
The system follows a **tool-based RAG architecture** where Claude AI uses search tools to query a vector database instead of direct RAG retrieval:

1. **User Query** → Frontend → FastAPI endpoint
2. **RAG System** orchestrates: AI Generator + Tools + Session Management  
3. **Claude AI** decides whether to use search tools based on query type
4. **Search Tools** perform semantic search via ChromaDB vector store
5. **Response Generation** using search results as context

### Key Components

**Backend (`/backend/`):**
- `app.py` - FastAPI server with CORS, serves static files, handles `/api/query` and `/api/courses`
- `rag_system.py` - Main orchestrator, coordinates all components
- `ai_generator.py` - Claude API integration with tool calling support
- `search_tools.py` - Tool system for Claude to search course content
- `vector_store.py` - ChromaDB wrapper with semantic search
- `document_processor.py` - Text chunking and course document parsing
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic models for Course, Lesson, CourseChunk

**Frontend (`/frontend/`):**
- `index.html` - Single-page chat interface
- `script.js` - API calls, chat UI, session management
- `style.css` - Responsive styling

### Document Processing Flow
1. **Course documents** (`/docs/`) follow structured format:
   - Line 1: `Course Title: [title]`
   - Line 2: `Course Link: [url]` 
   - Line 3: `Course Instructor: [instructor]`
   - Content organized by `Lesson N: [title]` markers

2. **Text chunking** creates overlapping chunks with course/lesson context
3. **Vector storage** in ChromaDB with metadata for filtering
4. **Search** supports course name and lesson number filtering

### Configuration (`config.py`)
- **Model:** `claude-sonnet-4-20250514` 
- **Embeddings:** `all-MiniLM-L6-v2`
- **Chunk size:** 800 chars with 100 char overlap
- **ChromaDB:** Stored in `./chroma_db`
- **Session history:** Limited to 2 exchanges

### API Endpoints
- `POST /api/query` - Main chat endpoint (query, session_id) → (answer, sources, session_id)
- `GET /api/courses` - Course statistics → (total_courses, course_titles)
- `GET /` - Serves frontend static files

### Session Management
- Auto-created session IDs for new conversations
- Conversation history maintained for context
- History limited by MAX_HISTORY config

## Important Implementation Details

**Tool-based Search:** Claude uses `search_course_content` tool rather than receiving pre-filtered context. This allows dynamic, query-appropriate searches.

**Document Structure:** Course documents must follow the expected format with title/link/instructor headers, otherwise parsing may fail.

**Vector Search Filtering:** Search can filter by partial course name matches and specific lesson numbers via tool parameters.

**Sources Tracking:** Search tools track sources separately from results to display in UI collapsible sections.

**CORS Configuration:** Set to allow all origins (`"*"`) for development - should be restricted in production.

- Always use uv to run the server do not use pip directly
- Make sure to use uv to manage dependencies
- Use uv to run any python file instead of calling python directly