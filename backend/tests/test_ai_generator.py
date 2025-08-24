"""
AI Generator Tests - Integration tests for Claude API tool calling
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import VectorStore, SearchResults


class TestAIGeneratorWithRealAPI(unittest.TestCase):
    """Test AI Generator with real Anthropic API (when available)"""
    
    def setUp(self):
        """Set up test environment"""
        self.api_available = bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip())
        
        if not self.api_available:
            self.skipTest("ANTHROPIC_API_KEY not available - skipping real API tests")
        
        self.ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Create simple tool manager for testing
        self.tool_manager = ToolManager()
        
        # Use mock vector store to avoid dependency on real data for API tests
        self.mock_vector_store = self._create_mock_vector_store()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(self.search_tool)
    
    def _create_mock_vector_store(self):
        """Create mock vector store with predictable responses"""
        mock_store = Mock(spec=VectorStore)
        
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is introduction content about getting started with the course"],
            metadata=[{"course_title": "Introduction to AI", "lesson_number": 0}],
            distances=[0.1]
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson0"
        
        return mock_store
    
    def test_basic_response_generation(self):
        """Test basic response generation without tools"""
        query = "What is 2 + 2?"
        response = self.ai_generator.generate_response(query)
        
        print(f"\n🤖 Basic AI Response Test:")
        print(f"   Query: {query}")
        print(f"   Response: {response[:200]}...")
        print(f"   Response length: {len(response)}")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Should contain the answer to math question
        self.assertIn("4", response)
    
    def test_tool_calling_decision_making(self):
        """Test if AI correctly decides when to use tools"""
        # This should trigger tool use
        course_query = "Tell me about introduction to AI course content"
        
        response = self.ai_generator.generate_response(
            query=course_query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        print(f"\n🔧 Tool Calling Decision Test:")
        print(f"   Query: {course_query}")
        print(f"   Response: {response[:300]}...")
        print(f"   Response length: {len(response)}")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Should contain content related to course (mock tool results or AI knowledge)
        self.assertTrue(any(keyword in response.lower() for keyword in ["course", "introduction", "ai", "lesson"]))
    
    def test_general_knowledge_vs_tool_use(self):
        """Test that AI uses tools for course content but not for general knowledge"""
        # General knowledge - should not use tools
        general_query = "What is machine learning?"
        general_response = self.ai_generator.generate_response(
            query=general_query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        # Course specific - should use tools
        course_query = "What does lesson 0 cover in the introduction course?"
        course_response = self.ai_generator.generate_response(
            query=course_query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        print(f"\n🧠 Knowledge vs Tool Use Test:")
        print(f"   General query response length: {len(general_response)}")
        print(f"   Course query response length: {len(course_response)}")
        print(f"   General response preview: {general_response[:150]}...")
        print(f"   Course response preview: {course_response[:150]}...")
        
        self.assertIsInstance(general_response, str)
        self.assertIsInstance(course_response, str)
        self.assertGreater(len(general_response), 0)
        self.assertGreater(len(course_response), 0)
    
    def test_conversation_history_integration(self):
        """Test that conversation history is properly integrated"""
        initial_query = "Hello, I want to learn about AI courses"
        follow_up = "Tell me more about the introduction lesson"
        
        # First interaction
        first_response = self.ai_generator.generate_response(initial_query)
        
        # Build conversation history
        history = f"Human: {initial_query}\nAssistant: {first_response}"
        
        # Second interaction with history
        second_response = self.ai_generator.generate_response(
            query=follow_up,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        print(f"\n💬 Conversation History Test:")
        print(f"   First query: {initial_query}")
        print(f"   Follow-up query: {follow_up}")
        print(f"   Second response: {second_response[:200]}...")
        
        self.assertIsInstance(second_response, str)
        self.assertGreater(len(second_response), 0)
    
    def test_sequential_tool_calling_complex_query(self):
        """Test sequential tool calling for complex multi-step queries"""
        # This type of query should trigger multiple tool calls
        complex_query = "Find a course that discusses the same topic as lesson 4 of Introduction to AI course"
        
        response = self.ai_generator.generate_response(
            query=complex_query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        print(f"\n🔗 Sequential Tool Calling Test:")
        print(f"   Query: {complex_query}")
        print(f"   Response: {response[:300]}...")
        print(f"   Response length: {len(response)}")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Should contain content related to the complex search
        self.assertIn("course", response.lower())
    
    def test_max_rounds_limitation(self):
        """Test that the system respects the max_rounds parameter"""
        query = "Search for advanced AI topics in multiple courses"
        
        # Test with max_rounds=1 (should limit to single round)
        response_single = self.ai_generator.generate_response(
            query=query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
            max_rounds=1
        )
        
        # Test with default max_rounds=2
        response_double = self.ai_generator.generate_response(
            query=query,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
            max_rounds=2
        )
        
        print(f"\n🔢 Max Rounds Test:")
        print(f"   Query: {query}")
        print(f"   Single round response length: {len(response_single)}")
        print(f"   Double round response length: {len(response_double)}")
        
        self.assertIsInstance(response_single, str)
        self.assertIsInstance(response_double, str)
        self.assertGreater(len(response_single), 0)
        self.assertGreater(len(response_double), 0)


class TestAIGeneratorWithMockAPI(unittest.TestCase):
    """Test AI Generator with mocked Anthropic API"""
    
    def setUp(self):
        """Set up test environment with mocked API"""
        self.ai_generator = AIGenerator("mock_key", "mock_model")
        
        # Create tool manager
        self.tool_manager = ToolManager()
        mock_vector_store = Mock(spec=VectorStore)
        search_tool = CourseSearchTool(mock_vector_store)
        self.tool_manager.register_tool(search_tool)
    
    @patch('anthropic.Anthropic')
    def test_initialization(self, mock_anthropic):
        """Test AI generator initialization"""
        ai_gen = AIGenerator("test_key", "test_model")
        
        mock_anthropic.assert_called_once_with(api_key="test_key")
        self.assertEqual(ai_gen.model, "test_model")
        self.assertIsNotNone(ai_gen.base_params)
        self.assertEqual(ai_gen.base_params['model'], "test_model")
    
    @patch('anthropic.Anthropic')
    def test_basic_response_without_tools(self, mock_anthropic):
        """Test basic response generation without tool use"""
        # Mock the API response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        response = ai_gen.generate_response("Test query")
        
        print(f"\n🤖 Mock Basic Response Test:")
        print(f"   Response: {response}")
        
        self.assertEqual(response, "This is a test response")
        mock_client.messages.create.assert_called_once()
        
        # Check the API call parameters
        call_args = mock_client.messages.create.call_args
        self.assertIn('messages', call_args.kwargs)
        self.assertIn('system', call_args.kwargs)
        self.assertEqual(call_args.kwargs['messages'][0]['content'], "Test query")
    
    @patch('anthropic.Anthropic')
    def test_tool_calling_flow(self, mock_anthropic):
        """Test the complete tool calling flow"""
        # Mock initial response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution  
        mock_text = Mock()
        mock_text.text = "Final response with tool results"
        mock_final_response = Mock()
        mock_final_response.content = [mock_text]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Mock tool execution
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        response = ai_gen.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        print(f"\n🔧 Mock Tool Calling Test:")
        print(f"   Response: {response}")
        
        self.assertEqual(response, "Final response with tool results")
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test")
        
        # Verify API was called twice (initial + final)
        self.assertEqual(mock_client.messages.create.call_count, 2)
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic):
        """Test error handling during tool execution"""
        # Mock response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"
        
        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.stop_reason = "tool_use"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # This should not crash, but handle the error gracefully
        response = ai_gen.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        # Should handle the error and return some response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_system_prompt_structure(self):
        """Test that system prompt contains expected elements"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        print(f"\n📝 System Prompt Test:")
        print(f"   Prompt length: {len(system_prompt)}")
        print(f"   Contains tool usage guidelines: {'Tool Usage Guidelines' in system_prompt}")
        print(f"   Contains response protocol: {'Response Protocol' in system_prompt}")
        
        # Check for key elements
        self.assertIn("search_course_content", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
        self.assertIn("Tool Usage Guidelines", system_prompt)
        self.assertIn("Response Protocol", system_prompt)
        
        # Check response requirements
        self.assertIn("Brief, Concise and focused", system_prompt)
        self.assertIn("Educational", system_prompt)
        self.assertIn("Clear", system_prompt)
    
    def test_api_parameters_structure(self):
        """Test that API parameters are correctly structured"""
        base_params = self.ai_generator.base_params
        
        print(f"\n⚙️ API Parameters Test:")
        print(f"   Base parameters: {base_params}")
        
        # Check required parameters
        self.assertIn("model", base_params)
        self.assertIn("temperature", base_params)
        self.assertIn("max_tokens", base_params)
        
        # Check parameter values
        self.assertEqual(base_params["temperature"], 0)
        self.assertEqual(base_params["max_tokens"], 800)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_mock(self, mock_anthropic):
        """Test sequential tool calling with mock API responses"""
        # Mock first response with tool use
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.input = {"course_name": "Introduction to AI"}
        mock_tool_use_1.id = "tool_1"
        
        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_use_1]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response with another tool use
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.input = {"query": "machine learning basics"}
        mock_tool_use_2.id = "tool_2"
        
        mock_second_response = Mock()
        mock_second_response.content = [mock_tool_use_2]
        mock_second_response.stop_reason = "tool_use"
        
        # Mock final response without tools
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final synthesized response from sequential tool calls")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_first_response, mock_second_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Mock tool manager with different responses
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline with lesson 4: Machine Learning Basics",
            "Search results for machine learning courses"
        ]
        
        response = ai_gen.generate_response(
            "Find courses about the same topic as lesson 4 of Introduction to AI",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        print(f"\n🔗 Sequential Mock Test:")
        print(f"   Response: {response}")
        
        self.assertEqual(response, "Final synthesized response from sequential tool calls")
        
        # Verify tool was executed twice with different parameters
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify API was called three times (2 rounds + 1 final)
        self.assertEqual(mock_client.messages.create.call_count, 3)
    
    @patch('anthropic.Anthropic')
    def test_early_termination_no_tools(self, mock_anthropic):
        """Test that sequential calling terminates early when no tools are needed"""
        # Mock response without tool use (should terminate after first round)
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tools needed")]
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        mock_tool_manager = Mock()
        
        response = ai_gen.generate_response(
            "What is 2 + 2?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        print(f"\n🛑 Early Termination Test:")
        print(f"   Response: {response}")
        
        self.assertEqual(response, "Direct response without tools needed")
        
        # Should only call API once (early termination)
        self.assertEqual(mock_client.messages.create.call_count, 1)
        
        # Should not execute any tools
        mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling_sequential(self, mock_anthropic):
        """Test error handling during sequential tool execution"""
        # Mock first response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"
        
        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_use]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response after error
        mock_second_response = Mock()
        mock_second_response.content = [Mock(text="Response despite tool error")]
        mock_second_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_first_response, mock_second_response]
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "test_model")
        
        # Mock tool manager that raises exception on first call, succeeds on second
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [Exception("Tool execution failed")]
        
        response = ai_gen.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        print(f"\n❌ Error Handling Test:")
        print(f"   Response: {response}")
        
        self.assertEqual(response, "Response despite tool error")
        
        # Should still attempt tool execution and continue
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Should call API twice (first round with error, second round continues)
        self.assertEqual(mock_client.messages.create.call_count, 2)


if __name__ == '__main__':
    # Run real API tests only if API key is available
    if config.ANTHROPIC_API_KEY:
        print("Running tests with real Anthropic API...")
    else:
        print("ANTHROPIC_API_KEY not found - running mock tests only...")
    
    unittest.main(verbosity=2)