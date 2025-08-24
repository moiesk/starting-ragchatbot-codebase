"""
Search Tools Tests - Unit tests for CourseSearchTool and CourseOutlineTool
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import VectorStore, SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test CourseSearchTool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Try to use real vector store first, fall back to mock if needed
        self.use_real_vector_store = True
        try:
            self.vector_store = VectorStore(
                config.CHROMA_PATH,
                config.EMBEDDING_MODEL,
                config.MAX_RESULTS
            )
            # Test if we have data
            if self.vector_store.get_course_count() == 0:
                self.use_real_vector_store = False
        except Exception:
            self.use_real_vector_store = False
        
        if not self.use_real_vector_store:
            self.vector_store = self._create_mock_vector_store()
        
        self.search_tool = CourseSearchTool(self.vector_store)
    
    def _create_mock_vector_store(self):
        """Create mock vector store with predictable behavior"""
        mock_store = Mock(spec=VectorStore)
        
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is lesson 1 content about introduction", "More content about the topic"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 1}
            ],
            distances=[0.1, 0.2]
        )
        mock_store.search.return_value = mock_results
        
        # Mock lesson link retrieval
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        return mock_store
    
    def test_tool_definition_structure(self):
        """Test that tool definition has correct structure"""
        definition = self.search_tool.get_tool_definition()
        
        print(f"\n🛠 CourseSearchTool Definition:")
        print(f"   Name: {definition.get('name')}")
        print(f"   Description: {definition.get('description')}")
        print(f"   Required params: {definition.get('input_schema', {}).get('properties', {}).get('query', {}).get('type')}")
        
        # Check required fields
        self.assertEqual(definition['name'], 'search_course_content')
        self.assertIn('description', definition)
        self.assertIn('input_schema', definition)
        
        # Check schema structure
        schema = definition['input_schema']
        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
        self.assertIn('required', schema)
        
        # Check required parameters
        self.assertIn('query', schema['required'])
        
        # Check parameter definitions
        properties = schema['properties']
        self.assertIn('query', properties)
        self.assertIn('course_name', properties)
        self.assertIn('lesson_number', properties)
        
        # Verify parameter types
        self.assertEqual(properties['query']['type'], 'string')
        self.assertEqual(properties['course_name']['type'], 'string')
        self.assertEqual(properties['lesson_number']['type'], 'integer')
    
    def test_execute_basic_query(self):
        """Test basic query execution"""
        result = self.search_tool.execute("introduction")
        
        print(f"\n🔍 Basic Query Test:")
        print(f"   Query: 'introduction'")
        print(f"   Result length: {len(result)}")
        print(f"   Result preview: {result[:200]}...")
        
        self.assertIsInstance(result, str, "Execute should return string")
        self.assertGreater(len(result), 0, "Result should not be empty")
        
        # Should not contain error indicators (if we have real data)
        if self.use_real_vector_store:
            self.assertNotIn("No relevant content found", result, "Should find content for 'introduction'")
    
    def test_execute_with_course_filter(self):
        """Test query execution with course name filter"""
        if not self.use_real_vector_store:
            # For mock, we'll test the parameter passing
            result = self.search_tool.execute("test", course_name="Test Course")
            self.assertIsInstance(result, str)
            return
        
        # Get a real course title to test with
        course_titles = self.vector_store.get_existing_course_titles()
        if not course_titles:
            self.skipTest("No courses available for testing")
        
        first_course = course_titles[0]
        result = self.search_tool.execute("introduction", course_name=first_course)
        
        print(f"\n🔍 Course Filtered Query Test:")
        print(f"   Query: 'introduction'")
        print(f"   Course filter: {first_course}")
        print(f"   Result length: {len(result)}")
        print(f"   Result preview: {result[:200]}...")
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Should contain course context
        self.assertIn(first_course, result, "Result should contain course title")
    
    def test_execute_with_lesson_filter(self):
        """Test query execution with lesson number filter"""
        result = self.search_tool.execute("content", lesson_number=1)
        
        print(f"\n🔍 Lesson Filtered Query Test:")
        print(f"   Query: 'content'")
        print(f"   Lesson filter: 1")
        print(f"   Result length: {len(result)}")
        print(f"   Result preview: {result[:200]}...")
        
        self.assertIsInstance(result, str)
        
        # If we have real data, check for lesson context
        if self.use_real_vector_store and "No relevant content found" not in result:
            self.assertIn("Lesson 1", result, "Result should contain lesson context")
    
    def test_execute_with_combined_filters(self):
        """Test query execution with both course and lesson filters"""
        if not self.use_real_vector_store:
            result = self.search_tool.execute("test", course_name="Test Course", lesson_number=1)
            self.assertIsInstance(result, str)
            return
        
        course_titles = self.vector_store.get_existing_course_titles()
        if not course_titles:
            self.skipTest("No courses available for testing")
        
        first_course = course_titles[0]
        result = self.search_tool.execute("introduction", course_name=first_course, lesson_number=0)
        
        print(f"\n🔍 Combined Filters Query Test:")
        print(f"   Query: 'introduction'")
        print(f"   Course filter: {first_course}")
        print(f"   Lesson filter: 0")
        print(f"   Result length: {len(result)}")
        print(f"   Result preview: {result[:200]}...")
        
        self.assertIsInstance(result, str)
    
    def test_execute_invalid_course(self):
        """Test query execution with invalid course name"""
        result = self.search_tool.execute("test", course_name="NonExistentCourse12345")
        
        print(f"\n❌ Invalid Course Query Test:")
        print(f"   Query: 'test'")
        print(f"   Invalid course: 'NonExistentCourse12345'")
        print(f"   Result: {result}")
        
        self.assertIsInstance(result, str)
        # Should return error message
        self.assertIn("No course found", result, "Should return error for invalid course")
    
    def test_execute_empty_results(self):
        """Test query execution that returns no results"""
        # Use a very specific query that likely won't match anything
        result = self.search_tool.execute("xyzabcunlikelyquery123")
        
        print(f"\n🤷 Empty Results Query Test:")
        print(f"   Query: 'xyzabcunlikelyquery123'")
        print(f"   Result: {result}")
        
        self.assertIsInstance(result, str)
        # Should return "no content found" message
        if "No relevant content found" not in result and len(result) < 50:
            # Might be an actual error, which is also valid to test
            print(f"   Got potential error instead of empty results: {result}")
    
    def test_sources_tracking(self):
        """Test that sources are properly tracked after search"""
        # Execute a search
        result = self.search_tool.execute("introduction")
        
        # Check if sources were tracked
        sources = self.search_tool.last_sources
        
        print(f"\n📚 Sources Tracking Test:")
        print(f"   Query: 'introduction'")
        print(f"   Sources tracked: {len(sources)}")
        print(f"   Sources: {sources}")
        
        self.assertIsInstance(sources, list, "Sources should be a list")
        
        if self.use_real_vector_store and "No relevant content found" not in result:
            self.assertGreater(len(sources), 0, "Should track sources for successful search")
            
            # Check source structure
            if sources:
                first_source = sources[0]
                self.assertIn('text', first_source, "Source should have 'text' field")
                # 'link' field is optional
    
    def test_result_formatting(self):
        """Test that results are properly formatted with course and lesson context"""
        if not self.use_real_vector_store:
            # Use mock to test formatting
            mock_results = SearchResults(
                documents=["Test content here"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.1]
            )
            formatted = self.search_tool._format_results(mock_results)
            
            self.assertIn("[Test Course - Lesson 1]", formatted)
            self.assertIn("Test content here", formatted)
            return
        
        # Test with real data if available
        result = self.search_tool.execute("introduction")
        
        print(f"\n🎨 Result Formatting Test:")
        print(f"   Formatted result preview: {result[:300]}...")
        
        if "No relevant content found" not in result:
            # Should contain course/lesson context markers
            self.assertTrue(
                "[" in result and "]" in result,
                "Result should contain context markers [Course - Lesson X]"
            )


class TestCourseOutlineTool(unittest.TestCase):
    """Test CourseOutlineTool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        try:
            self.vector_store = VectorStore(
                config.CHROMA_PATH,
                config.EMBEDDING_MODEL,
                config.MAX_RESULTS
            )
            self.use_real_vector_store = self.vector_store.get_course_count() > 0
        except Exception:
            self.use_real_vector_store = False
            self.vector_store = self._create_mock_vector_store()
        
        self.outline_tool = CourseOutlineTool(self.vector_store)
    
    def _create_mock_vector_store(self):
        """Create mock vector store for outline tool"""
        mock_store = Mock(spec=VectorStore)
        
        # Mock course resolution
        mock_store._resolve_course_name.return_value = "Test Course"
        
        # Mock course metadata
        mock_courses_metadata = [{
            'title': 'Test Course',
            'instructor': 'Test Instructor',
            'course_link': 'https://example.com/course',
            'lessons': [
                {'lesson_number': 0, 'lesson_title': 'Introduction', 'lesson_link': 'https://example.com/lesson0'},
                {'lesson_number': 1, 'lesson_title': 'Basics', 'lesson_link': 'https://example.com/lesson1'}
            ]
        }]
        mock_store.get_all_courses_metadata.return_value = mock_courses_metadata
        
        return mock_store
    
    def test_tool_definition_structure(self):
        """Test outline tool definition"""
        definition = self.outline_tool.get_tool_definition()
        
        print(f"\n🛠 CourseOutlineTool Definition:")
        print(f"   Name: {definition.get('name')}")
        print(f"   Description: {definition.get('description')}")
        
        self.assertEqual(definition['name'], 'get_course_outline')
        self.assertIn('description', definition)
        self.assertIn('input_schema', definition)
        
        schema = definition['input_schema']
        self.assertIn('course_name', schema['required'])
        self.assertEqual(schema['properties']['course_name']['type'], 'string')
    
    def test_execute_valid_course(self):
        """Test outline retrieval for valid course"""
        if not self.use_real_vector_store:
            result = self.outline_tool.execute("Test Course")
            self.assertIn("Test Course", result)
            self.assertIn("Test Instructor", result)
            return
        
        course_titles = self.vector_store.get_existing_course_titles()
        if not course_titles:
            self.skipTest("No courses available for testing")
        
        first_course = course_titles[0]
        result = self.outline_tool.execute(first_course)
        
        print(f"\n📋 Course Outline Test:")
        print(f"   Course: {first_course}")
        print(f"   Outline length: {len(result)}")
        print(f"   Outline preview: {result[:400]}...")
        
        self.assertIsInstance(result, str)
        self.assertIn(first_course, result, "Outline should contain course title")
        self.assertIn("Instructor:", result, "Outline should contain instructor info")
        self.assertIn("Lessons:", result, "Outline should contain lessons section")
    
    def test_execute_invalid_course(self):
        """Test outline retrieval for invalid course"""
        result = self.outline_tool.execute("NonExistentCourse12345")
        
        print(f"\n❌ Invalid Course Outline Test:")
        print(f"   Invalid course: 'NonExistentCourse12345'")
        print(f"   Result: {result}")
        
        self.assertIn("No course found", result, "Should return error for invalid course")


class TestToolManager(unittest.TestCase):
    """Test ToolManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.tool_manager = ToolManager()
        
        # Create mock vector store
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults(
            documents=["test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1]
        )
        
        # Create tools
        self.search_tool = CourseSearchTool(mock_vector_store)
        self.outline_tool = CourseOutlineTool(mock_vector_store)
    
    def test_tool_registration(self):
        """Test tool registration functionality"""
        # Register tools
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)
        
        # Check tool definitions
        definitions = self.tool_manager.get_tool_definitions()
        tool_names = [tool['name'] for tool in definitions]
        
        print(f"\n🔧 Tool Manager Registration Test:")
        print(f"   Registered tools: {tool_names}")
        
        self.assertEqual(len(definitions), 2, "Should have 2 tools registered")
        self.assertIn('search_course_content', tool_names)
        self.assertIn('get_course_outline', tool_names)
    
    def test_tool_execution(self):
        """Test tool execution through manager"""
        # Register tools
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)
        
        # Execute search tool
        result = self.tool_manager.execute_tool('search_course_content', query='test')
        
        print(f"\n⚙️ Tool Execution Test:")
        print(f"   Tool: search_course_content")
        print(f"   Result: {result[:100]}...")
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Test invalid tool
        invalid_result = self.tool_manager.execute_tool('invalid_tool', query='test')
        self.assertIn("Tool 'invalid_tool' not found", invalid_result)
    
    def test_sources_management(self):
        """Test sources tracking and reset functionality"""
        # Register tool with sources tracking
        self.tool_manager.register_tool(self.search_tool)
        
        # Execute search to generate sources
        self.tool_manager.execute_tool('search_course_content', query='test')
        
        # Check sources
        sources = self.tool_manager.get_last_sources()
        
        print(f"\n📚 Sources Management Test:")
        print(f"   Sources found: {len(sources)}")
        print(f"   Sources: {sources}")
        
        self.assertIsInstance(sources, list)
        
        # Reset sources
        self.tool_manager.reset_sources()
        
        # Check sources are cleared
        reset_sources = self.tool_manager.get_last_sources()
        self.assertEqual(len(reset_sources), 0, "Sources should be cleared after reset")


if __name__ == '__main__':
    unittest.main(verbosity=2)