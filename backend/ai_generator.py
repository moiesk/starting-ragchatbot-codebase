import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Course content questions**: Use `search_course_content` for specific content, materials, or detailed educational information
- **Course outline questions**: Use `get_course_outline` for course structure, lesson lists, course links, or overview information
- **Sequential tool calls allowed**: Use up to 2 rounds of tool calls to gather comprehensive information for complex queries
- **Multi-round strategy**: Make initial tool call, analyze results, then optionally make additional targeted tool calls for follow-up questions
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Usage Examples:
- "Find a course discussing the same topic as lesson 4 of course X": First get course outline for course X to find lesson 4 title, then search for courses with that topic
- Complex comparisons: Search multiple courses or lessons sequentially to make comparisons
- Multi-part questions: Break down into sequential searches as needed

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search tool first, then answer
- **Course outline/structure questions**: Use outline tool first, then answer
- **Complex queries**: Use multiple tool calls strategically to gather complete information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

For course outline responses, always include:
- Course title and instructor
- Course link (if available)
- Complete list of lessons with numbers, titles, and links

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize message history for multi-round conversation
        messages = [{"role": "user", "content": query}]
        rounds_completed = 0
        
        # Multi-round tool calling loop
        while rounds_completed < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools if available and not on final round
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                # Handle API errors gracefully
                if rounds_completed == 0:
                    raise e  # Re-raise on first round failure
                else:
                    # Return best available response if later rounds fail
                    return "Unable to complete additional searches. Please try rephrasing your query."
            
            rounds_completed += 1
            
            # Check if tool use is requested
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and update message history
                try:
                    messages = self._handle_tool_execution_and_continue(response, messages, tool_manager)
                    # Continue loop for potential additional rounds
                    continue
                except Exception as e:
                    # Tool execution failed - return partial response if available
                    if rounds_completed == 1:
                        raise e  # Re-raise if first round fails
                    else:
                        return "Search encountered an error. Please try rephrasing your query."
            else:
                # No tool use requested - return Claude's response
                return response.content[0].text
        
        # Max rounds reached - make final call without tools to get synthesized response
        try:
            final_api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            final_response = self.client.messages.create(**final_api_params)
            return final_response.content[0].text
        except Exception as e:
            return "Unable to synthesize final response. Please try rephrasing your query."
    
    def _handle_tool_execution_and_continue(self, initial_response, messages: List, tool_manager) -> List:
        """
        Handle execution of tool calls and return updated message history for continuation.
        
        Args:
            initial_response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools
            
        Returns:
            Updated messages list with tool execution results
        """
        # Add AI's tool use response to message history
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle individual tool failures gracefully
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}",
                        "is_error": True
                    })
        
        # Add tool results as single message if any results were collected
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        return messages
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Legacy method for backward compatibility - handles single-round tool execution.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Use the new method to get updated messages
        messages = self._handle_tool_execution_and_continue(
            initial_response, 
            base_params["messages"].copy(), 
            tool_manager
        )
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text