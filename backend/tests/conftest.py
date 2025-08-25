"""
Pytest configuration and shared fixtures for RAG system tests
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio

# Import the actual modules for mocking/testing
from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from session_manager import SessionManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_course_content():
    """Sample course content for testing"""
    return {
        "course1.txt": """Course Title: Introduction to Python
Course Link: https://example.com/python-intro
Course Instructor: Jane Doe

Lesson 1: Python Basics
This is the first lesson about Python basics.
Variables and data types are fundamental concepts.

Lesson 2: Control Flow
Learn about if statements and loops.
Control flow manages program execution.

Lesson 3: Functions
Functions help organize your code.
They make code reusable and modular.
""",
        "course2.txt": """Course Title: Advanced JavaScript
Course Link: https://example.com/js-advanced  
Course Instructor: John Smith

Lesson 1: Async Programming
Asynchronous programming with promises and async/await.
Handle multiple operations concurrently.

Lesson 2: Module Systems
ES6 modules and CommonJS patterns.
Organize large JavaScript applications.
"""
    }


@pytest.fixture
def sample_docs_folder(temp_dir, sample_course_content):
    """Create a temporary docs folder with sample course files"""
    docs_path = Path(temp_dir) / "docs"
    docs_path.mkdir()
    
    for filename, content in sample_course_content.items():
        (docs_path / filename).write_text(content)
    
    return str(docs_path)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    mock_client = Mock()
    
    # Mock successful API response
    mock_response = Mock()
    mock_response.content = [
        Mock(text="This is a test response from Claude AI")
    ]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Mock AI generator with tool support"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        ai_gen = AIGenerator(config.ANTHROPIC_MODEL, config.ANTHROPIC_API_KEY)
        
        # Mock the generate method
        async def mock_generate(messages, tools=None):
            if tools:
                # Mock tool call response
                return "Based on the search results, here's the answer.", [{"source": "test_source"}]
            else:
                return "This is a test response.", []
        
        ai_gen.generate = AsyncMock(side_effect=mock_generate)
        return ai_gen


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_vs = Mock(spec=VectorStore)
    
    # Mock successful initialization
    mock_vs.collection_name = "test_collection"
    mock_vs.client = Mock()
    mock_vs.collection = Mock()
    
    # Mock search results
    mock_search_results = [
        {
            "content": "Sample content about Python basics",
            "course_name": "Introduction to Python",
            "lesson_number": 1,
            "source": "course1.txt"
        }
    ]
    
    mock_vs.search.return_value = mock_search_results
    mock_vs.add_documents.return_value = True
    mock_vs.get_collection_info.return_value = {
        "total_documents": 10,
        "total_courses": 2
    }
    
    return mock_vs


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_sm = Mock(spec=SessionManager)
    
    # Mock session operations
    mock_sm.create_session.return_value = "test_session_123"
    mock_sm.get_session_history.return_value = []
    mock_sm.add_to_session.return_value = None
    mock_sm.clear_session.return_value = None
    
    return mock_sm


@pytest.fixture
def mock_rag_system(mock_ai_generator, mock_vector_store, mock_session_manager):
    """Mock RAG system with all dependencies"""
    mock_rag = Mock(spec=RAGSystem)
    
    # Set up the mocked components
    mock_rag.ai_generator = mock_ai_generator
    mock_rag.vector_store = mock_vector_store
    mock_rag.session_manager = mock_session_manager
    
    # Mock main query method
    async def mock_query(query, session_id):
        return "Test response from RAG system", [{"source": "test_source", "content": "test content"}]
    
    mock_rag.query = AsyncMock(side_effect=mock_query)
    
    # Mock analytics method
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Python", "Advanced JavaScript"]
    }
    
    # Mock document loading
    mock_rag.add_course_folder.return_value = (2, 5)  # 2 courses, 5 chunks
    
    return mock_rag


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    
    # Create a minimal test app with just the API endpoints
    app = FastAPI(title="Test RAG API")
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (copied from app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, Any]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class ClearSessionRequest(BaseModel):
        session_id: str

    class ClearSessionResponse(BaseModel):
        success: bool
        message: str
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.session_manager = Mock()
    
    # Create a counter for unique session IDs
    session_counter = 0
    def create_unique_session():
        nonlocal session_counter
        session_counter += 1
        return f"test_session_{session_counter}"
    
    mock_rag_system.session_manager.create_session.side_effect = create_unique_session
    
    async def mock_query_func(query, session_id):
        return "Test response", [{"source": "test.txt", "content": "test content"}]
    
    mock_rag_system.query = AsyncMock(side_effect=mock_query_func)
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag_system.session_manager.clear_session.return_value = None
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = await mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/clear-session", response_model=ClearSessionResponse)
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.session_id} cleared successfully"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "RAG Chatbot API is running"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)




@pytest.fixture
def api_test_data():
    """Test data for API endpoint tests"""
    return {
        "valid_query": {
            "query": "What is Python?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "How do Python functions work?"
        },
        "invalid_query": {
            "query": "",  # Empty query
            "session_id": "test_session"
        },
        "clear_session_request": {
            "session_id": "test_session_123"
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    # Ensure test environment variables are set
    os.environ.setdefault("ANTHROPIC_API_KEY", "test_api_key")
    os.environ.setdefault("ENVIRONMENT", "test")
    
    yield
    
    # Clean up after tests
    pass


# Pytest markers for organizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
pytest.mark.slow = pytest.mark.slow