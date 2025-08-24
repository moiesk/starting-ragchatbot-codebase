"""
System Health Tests - Verify basic configuration and system initialization
"""

import unittest
import os
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
import chromadb


class TestSystemHealth(unittest.TestCase):
    """Test basic system health and configuration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_results = []
    
    def log_result(self, test_name, status, details=""):
        """Log test results for analysis"""
        self.test_results.append({
            'test': test_name,
            'status': status,
            'details': details
        })
    
    def test_config_loaded(self):
        """Test if configuration is properly loaded"""
        try:
            # Check critical config values
            self.assertIsNotNone(config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY not set")
            self.assertNotEqual(config.ANTHROPIC_API_KEY, "", "ANTHROPIC_API_KEY is empty")
            self.assertIsNotNone(config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL not set")
            self.assertIsNotNone(config.EMBEDDING_MODEL, "EMBEDDING_MODEL not set")
            self.assertIsNotNone(config.CHROMA_PATH, "CHROMA_PATH not set")
            
            self.log_result("test_config_loaded", "PASS", 
                          f"API Key: {'SET' if config.ANTHROPIC_API_KEY else 'MISSING'}")
        except AssertionError as e:
            self.log_result("test_config_loaded", "FAIL", str(e))
            raise
    
    def test_chroma_db_exists(self):
        """Test if ChromaDB directory and files exist"""
        try:
            chroma_path = Path(config.CHROMA_PATH)
            self.assertTrue(chroma_path.exists(), f"ChromaDB path does not exist: {chroma_path}")
            
            # Check for ChromaDB files
            sqlite_file = chroma_path / "chroma.sqlite3"
            if sqlite_file.exists():
                self.log_result("test_chroma_db_exists", "PASS", 
                              f"ChromaDB found at {chroma_path}")
            else:
                self.log_result("test_chroma_db_exists", "WARN", 
                              "ChromaDB directory exists but no sqlite file found")
                
        except AssertionError as e:
            self.log_result("test_chroma_db_exists", "FAIL", str(e))
            raise
    
    def test_chroma_db_connection(self):
        """Test if we can connect to ChromaDB"""
        try:
            client = chromadb.PersistentClient(
                path=config.CHROMA_PATH,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            # Try to list collections
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            self.log_result("test_chroma_db_connection", "PASS", 
                          f"Connected. Collections: {collection_names}")
            
            # Check for expected collections
            if 'course_catalog' in collection_names and 'course_content' in collection_names:
                self.log_result("expected_collections", "PASS", "Both collections found")
            else:
                self.log_result("expected_collections", "WARN", 
                              f"Missing collections. Found: {collection_names}")
                
        except Exception as e:
            self.log_result("test_chroma_db_connection", "FAIL", str(e))
            raise
    
    def test_vector_store_initialization(self):
        """Test if VectorStore can be initialized"""
        try:
            vector_store = VectorStore(
                config.CHROMA_PATH, 
                config.EMBEDDING_MODEL, 
                config.MAX_RESULTS
            )
            
            # Check if collections are accessible
            course_count = vector_store.get_course_count()
            course_titles = vector_store.get_existing_course_titles()
            
            self.log_result("test_vector_store_initialization", "PASS", 
                          f"Courses: {course_count}, Titles: {course_titles[:3]}..." if course_titles else "No courses")
            
            # Check if we have actual data
            if course_count > 0:
                self.log_result("vector_store_data", "PASS", f"{course_count} courses loaded")
            else:
                self.log_result("vector_store_data", "FAIL", "No courses found in vector store")
                
        except Exception as e:
            self.log_result("test_vector_store_initialization", "FAIL", str(e))
            raise
    
    def test_ai_generator_initialization(self):
        """Test if AIGenerator can be initialized"""
        try:
            ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            
            # Check if the client is properly initialized
            self.assertIsNotNone(ai_generator.client, "Anthropic client not initialized")
            self.assertEqual(ai_generator.model, config.ANTHROPIC_MODEL)
            
            self.log_result("test_ai_generator_initialization", "PASS", 
                          f"Model: {ai_generator.model}")
            
        except Exception as e:
            self.log_result("test_ai_generator_initialization", "FAIL", str(e))
            raise
    
    def test_search_tools_initialization(self):
        """Test if search tools can be initialized"""
        try:
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            
            # Initialize tools
            search_tool = CourseSearchTool(vector_store)
            outline_tool = CourseOutlineTool(vector_store)
            tool_manager = ToolManager()
            
            # Register tools
            tool_manager.register_tool(search_tool)
            tool_manager.register_tool(outline_tool)
            
            # Check tool definitions
            tool_definitions = tool_manager.get_tool_definitions()
            tool_names = [tool['name'] for tool in tool_definitions]
            
            expected_tools = ['search_course_content', 'get_course_outline']
            for tool_name in expected_tools:
                self.assertIn(tool_name, tool_names, f"Tool {tool_name} not found")
            
            self.log_result("test_search_tools_initialization", "PASS", 
                          f"Tools registered: {tool_names}")
            
        except Exception as e:
            self.log_result("test_search_tools_initialization", "FAIL", str(e))
            raise
    
    def test_rag_system_initialization(self):
        """Test if RAGSystem can be fully initialized"""
        try:
            rag_system = RAGSystem(config)
            
            # Check if all components are initialized
            self.assertIsNotNone(rag_system.document_processor)
            self.assertIsNotNone(rag_system.vector_store)
            self.assertIsNotNone(rag_system.ai_generator)
            self.assertIsNotNone(rag_system.session_manager)
            self.assertIsNotNone(rag_system.tool_manager)
            
            # Check analytics
            analytics = rag_system.get_course_analytics()
            
            self.log_result("test_rag_system_initialization", "PASS", 
                          f"Analytics: {analytics}")
            
        except Exception as e:
            self.log_result("test_rag_system_initialization", "FAIL", str(e))
            raise
    
    def test_docs_folder_exists(self):
        """Test if docs folder exists with course files"""
        try:
            docs_path = Path("../docs")
            self.assertTrue(docs_path.exists(), "Docs folder not found")
            
            # List doc files
            doc_files = list(docs_path.glob("*.txt"))
            doc_files.extend(list(docs_path.glob("*.pdf")))
            doc_files.extend(list(docs_path.glob("*.docx")))
            
            self.log_result("test_docs_folder_exists", "PASS" if doc_files else "WARN", 
                          f"Doc files found: {[f.name for f in doc_files]}")
            
        except Exception as e:
            self.log_result("test_docs_folder_exists", "FAIL", str(e))
            raise
    
    def tearDown(self):
        """Print test results summary"""
        if hasattr(self, 'test_results'):
            print("\n" + "="*60)
            print("SYSTEM HEALTH TEST RESULTS")
            print("="*60)
            for result in self.test_results:
                status_symbol = "✓" if result['status'] == "PASS" else "✗" if result['status'] == "FAIL" else "⚠"
                print(f"{status_symbol} {result['test']}: {result['status']}")
                if result['details']:
                    print(f"   Details: {result['details']}")
            print("="*60)


if __name__ == '__main__':
    unittest.main(verbosity=2)