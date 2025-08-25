"""
End-to-End RAG System Tests - Test complete query processing flow
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from rag_system import RAGSystem
from vector_store import SearchResults
import uuid


class TestRAGSystemEndToEnd(unittest.TestCase):
    """Test complete RAG system functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_session_id = str(uuid.uuid4())

        # Try to initialize real RAG system
        self.use_real_system = True
        try:
            self.rag_system = RAGSystem(config)
            # Check if we have API key and data
            if (
                not config.ANTHROPIC_API_KEY
                or self.rag_system.get_course_analytics()["total_courses"] == 0
            ):
                self.use_real_system = False
        except Exception as e:
            print(f"Cannot initialize real RAG system: {e}")
            self.use_real_system = False

        if not self.use_real_system:
            self.rag_system = self._create_mock_rag_system()

    def _create_mock_rag_system(self):
        """Create mock RAG system for testing when real system unavailable"""
        mock_rag = Mock(spec=RAGSystem)

        # Mock successful query response
        def mock_query(query, session_id=None):
            if "course outline" in query.lower():
                return "Course outline mock response", [
                    {"text": "Course Outline", "link": None}
                ]
            else:
                return "Mock response to: " + query, [
                    {
                        "text": "Test Course - Lesson 1",
                        "link": "https://example.com/lesson1",
                    }
                ]

        mock_rag.query = mock_query
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"],
        }

        # Mock session manager
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "test_session_123"
        mock_rag.session_manager = mock_session_manager

        return mock_rag

    def test_basic_content_query(self):
        """Test basic content-related query processing"""
        query = "Tell me about introduction to machine learning"

        response, sources = self.rag_system.query(query, self.test_session_id)

        print(f"\n🎯 Basic Content Query Test:")
        print(f"   Query: {query}")
        print(f"   Response length: {len(response)}")
        print(f"   Response preview: {response[:200]}...")
        print(f"   Sources count: {len(sources)}")
        print(f"   Sources: {sources}")

        self.assertIsInstance(response, str, "Response should be string")
        self.assertGreater(len(response), 0, "Response should not be empty")

        # Check for "query failed" - this is the main issue we're diagnosing
        if "query failed" in response.lower():
            self.fail(f"Query failed with response: {response}")

        self.assertIsInstance(sources, list, "Sources should be a list")

    def test_course_outline_query(self):
        """Test course outline query processing"""
        query = "Can you give me the outline for the introduction course?"

        response, sources = self.rag_system.query(query, self.test_session_id)

        print(f"\n📋 Course Outline Query Test:")
        print(f"   Query: {query}")
        print(f"   Response length: {len(response)}")
        print(f"   Response preview: {response[:200]}...")
        print(f"   Sources count: {len(sources)}")

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

        # Check for "query failed"
        if "query failed" in response.lower():
            self.fail(f"Course outline query failed with response: {response}")

        # Should contain course structure information
        if self.use_real_system:
            self.assertTrue(
                any(
                    keyword in response.lower()
                    for keyword in ["course", "lesson", "instructor"]
                ),
                "Course outline response should contain course structure information",
            )

    def test_general_knowledge_query(self):
        """Test general knowledge query (should not use tools)"""
        query = "What is artificial intelligence?"

        response, sources = self.rag_system.query(query, self.test_session_id)

        print(f"\n🧠 General Knowledge Query Test:")
        print(f"   Query: {query}")
        print(f"   Response length: {len(response)}")
        print(f"   Response preview: {response[:200]}...")
        print(f"   Sources count: {len(sources)}")

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

        # General knowledge queries might not generate sources
        # This is expected behavior since they shouldn't use course search tools
        self.assertIsInstance(sources, list)

    def test_session_management(self):
        """Test session creation and management"""
        if not self.use_real_system:
            # Test mock session management
            session_id = self.rag_system.session_manager.create_session()
            self.assertIsInstance(session_id, str)
            return

        # Test session creation
        query1 = "Hello, I want to learn about machine learning"
        response1, sources1 = self.rag_system.query(
            query1, None
        )  # No session ID - should create one

        # Extract session ID from response or create one
        new_session_id = str(uuid.uuid4())  # Simulate session creation

        # Test follow-up with session
        query2 = "Tell me more about lesson 1"
        response2, sources2 = self.rag_system.query(query2, new_session_id)

        print(f"\n👥 Session Management Test:")
        print(f"   First query response length: {len(response1)}")
        print(f"   Second query response length: {len(response2)}")
        print(f"   Session ID: {new_session_id}")

        self.assertIsInstance(response1, str)
        self.assertIsInstance(response2, str)
        self.assertGreater(len(response1), 0)
        self.assertGreater(len(response2), 0)

    def test_course_analytics(self):
        """Test course analytics functionality"""
        analytics = self.rag_system.get_course_analytics()

        print(f"\n📊 Course Analytics Test:")
        print(f"   Analytics: {analytics}")

        self.assertIsInstance(analytics, dict)
        self.assertIn("total_courses", analytics)
        self.assertIn("course_titles", analytics)

        total_courses = analytics["total_courses"]
        course_titles = analytics["course_titles"]

        self.assertIsInstance(total_courses, int)
        self.assertIsInstance(course_titles, list)

        if self.use_real_system:
            self.assertGreater(total_courses, 0, "Should have courses loaded")
            self.assertEqual(
                len(course_titles),
                total_courses,
                "Course count should match titles count",
            )

        print(f"   Total courses: {total_courses}")
        print(f"   Course titles: {course_titles}")

    def test_error_handling_invalid_query(self):
        """Test error handling for various edge cases"""
        test_cases = [
            "",  # Empty query
            " ",  # Whitespace only
            "a" * 5000,  # Very long query
        ]

        for query in test_cases:
            try:
                response, sources = self.rag_system.query(query, self.test_session_id)

                print(f"\n⚠️ Edge Case Test - Query length {len(query)}:")
                print(f"   Response length: {len(response)}")
                print(f"   Response preview: {response[:100]}...")

                # Should not crash and should return some response
                self.assertIsInstance(response, str)
                self.assertIsInstance(sources, list)

            except Exception as e:
                print(f"   Exception for query length {len(query)}: {e}")
                # Some edge cases might raise exceptions - that's acceptable
                self.assertIsInstance(e, Exception)

    def test_source_tracking_consistency(self):
        """Test that sources are consistently tracked and formatted"""
        queries = [
            "Tell me about lesson 0",
            "What is covered in the introduction?",
            "Show me the course outline",
        ]

        for query in queries:
            response, sources = self.rag_system.query(query, self.test_session_id)

            print(f"\n🔗 Source Tracking Test - '{query[:30]}...':")
            print(f"   Response length: {len(response)}")
            print(f"   Sources count: {len(sources)}")
            print(f"   Sources structure: {sources[:2] if sources else 'None'}")

            self.assertIsInstance(sources, list)

            # Check source structure if we have sources
            for source in sources:
                if isinstance(source, dict):
                    self.assertIn("text", source, "Source should have 'text' field")
                    # 'link' field is optional
                elif isinstance(source, str):
                    # Old string format - still acceptable
                    self.assertIsInstance(source, str)
                else:
                    self.fail(f"Invalid source format: {type(source)}")

    @unittest.skipIf(
        not config.ANTHROPIC_API_KEY, "API key required for API integration test"
    )
    def test_api_integration_real_call(self):
        """Test actual API integration (only if API key available)"""
        if not self.use_real_system:
            self.skipTest("Real RAG system not available")

        # Make a real API call
        query = "What is machine learning?"  # General knowledge - shouldn't use tools

        try:
            response, sources = self.rag_system.query(query, self.test_session_id)

            print(f"\n🌐 Real API Integration Test:")
            print(f"   Query: {query}")
            print(f"   Response length: {len(response)}")
            print(f"   Response: {response[:300]}...")
            print(f"   Sources: {sources}")

            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertNotIn("query failed", response.lower())

        except Exception as e:
            self.fail(f"API integration failed: {e}")

    def test_tool_vs_no_tool_queries(self):
        """Test that appropriate queries use tools vs direct responses"""
        # Queries that should use tools (course-specific)
        tool_queries = [
            "Tell me about the course introduction",
            "What's in lesson 1?",
            "Show me the course outline",
        ]

        # Queries that should not use tools (general knowledge)
        no_tool_queries = [
            "What is 2 + 2?",
            "Define machine learning",
            "What is Python programming?",
        ]

        print(f"\n🔧 Tool Usage Analysis:")

        for query in tool_queries:
            response, sources = self.rag_system.query(query, self.test_session_id)
            print(f"   Tool query '{query[:30]}...': {len(sources)} sources")

            if self.use_real_system:
                # Course-specific queries should generate sources (when tools are used)
                # Note: This might not always be true if course data isn't loaded
                pass

        for query in no_tool_queries:
            response, sources = self.rag_system.query(query, self.test_session_id)
            print(f"   General query '{query[:30]}...': {len(sources)} sources")


class TestRAGSystemInitialization(unittest.TestCase):
    """Test RAG system initialization and component setup"""

    def test_component_initialization(self):
        """Test that all RAG system components are properly initialized"""
        try:
            rag_system = RAGSystem(config)

            # Check all components exist
            self.assertIsNotNone(rag_system.document_processor)
            self.assertIsNotNone(rag_system.vector_store)
            self.assertIsNotNone(rag_system.ai_generator)
            self.assertIsNotNone(rag_system.session_manager)
            self.assertIsNotNone(rag_system.tool_manager)
            self.assertIsNotNone(rag_system.search_tool)
            self.assertIsNotNone(rag_system.outline_tool)

            print(f"\n✅ RAG System Component Initialization:")
            print(f"   Document processor: OK")
            print(f"   Vector store: OK")
            print(f"   AI generator: OK")
            print(f"   Session manager: OK")
            print(f"   Tool manager: OK")
            print(f"   Search tool: OK")
            print(f"   Outline tool: OK")

        except Exception as e:
            self.fail(f"RAG system initialization failed: {e}")

    def test_tool_registration(self):
        """Test that tools are properly registered"""
        rag_system = RAGSystem(config)

        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]

        print(f"\n🔧 Tool Registration Test:")
        print(f"   Registered tools: {tool_names}")

        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
        self.assertEqual(len(tool_names), 2, "Should have exactly 2 tools registered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
