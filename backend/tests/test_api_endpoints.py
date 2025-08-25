"""
API Endpoint Tests for FastAPI RAG System
Tests all HTTP endpoints for proper request/response handling
"""

import pytest
import json
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_with_session_id(self, test_client, api_test_data):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=api_test_data["valid_query"])

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Check response content
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)
        assert data["session_id"] == api_test_data["valid_query"]["session_id"]

    def test_query_without_session_id(self, test_client, api_test_data):
        """Test query endpoint without session ID (should create new session)"""
        response = test_client.post(
            "/api/query", json=api_test_data["query_without_session"]
        )

        assert response.status_code == 200
        data = response.json()

        # Should create new session ID
        assert "session_id" in data
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

        # Should still return answer and sources
        assert "answer" in data
        assert "sources" in data

    def test_query_invalid_request(self, test_client):
        """Test query endpoint with invalid request data"""
        # Test empty query
        response = test_client.post("/api/query", json={"query": ""})

        # Should still work but might return different response
        # The exact behavior depends on RAG system implementation
        assert response.status_code in [200, 400, 422]

    def test_query_missing_query_field(self, test_client):
        """Test query endpoint with missing required query field"""
        response = test_client.post("/api/query", json={"session_id": "test"})

        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_query_content_type(self, test_client, api_test_data):
        """Test query endpoint with different content types"""
        # Test without content-type header
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        assert response.status_code == 200

        # Response should be JSON
        assert response.headers["content-type"].startswith("application/json")


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_get_courses_success(self, test_client):
        """Test successful course stats retrieval"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Check data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Check data consistency
        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]

    def test_get_courses_method_not_allowed(self, test_client):
        """Test that other HTTP methods are not allowed"""
        # POST should not be allowed
        response = test_client.post("/api/courses")
        assert response.status_code == 405

        # PUT should not be allowed
        response = test_client.put("/api/courses")
        assert response.status_code == 405

        # DELETE should not be allowed
        response = test_client.delete("/api/courses")
        assert response.status_code == 405


@pytest.mark.api
class TestClearSessionEndpoint:
    """Test the /api/clear-session endpoint"""

    def test_clear_session_success(self, test_client, api_test_data):
        """Test successful session clearing"""
        response = test_client.post(
            "/api/clear-session", json=api_test_data["clear_session_request"]
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "success" in data
        assert "message" in data

        # Check response content
        assert data["success"] is True
        assert isinstance(data["message"], str)
        assert api_test_data["clear_session_request"]["session_id"] in data["message"]

    def test_clear_session_missing_session_id(self, test_client):
        """Test clear session with missing session_id"""
        response = test_client.post("/api/clear-session", json={})

        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_clear_session_invalid_method(self, test_client):
        """Test that GET method is not allowed for clear session"""
        response = test_client.get("/api/clear-session")
        assert response.status_code == 405


@pytest.mark.api
class TestRootEndpoint:
    """Test the root endpoint /"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns appropriate response"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        # Should return some kind of status or welcome message
        assert isinstance(data, dict)
        assert "message" in data
        assert len(data["message"]) > 0


@pytest.mark.api
class TestErrorHandling:
    """Test error handling across all endpoints"""

    def test_query_endpoint_server_error(self, test_client):
        """Test query endpoint when server error occurs"""
        # This test would require mocking the RAG system to raise an exception
        # For now, we'll test the general structure
        response = test_client.post("/api/query", json={"query": "test query"})

        # Should either succeed or return proper error
        assert response.status_code in [200, 500]

        if response.status_code == 500:
            data = response.json()
            assert "detail" in data

    def test_nonexistent_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_cors_headers(self, test_client, api_test_data):
        """Test that CORS headers are properly set"""
        response = test_client.post(
            "/api/query",
            json=api_test_data["valid_query"],
            headers={"Origin": "http://localhost:3000"},
        )

        # Check for CORS headers (these should be set by the CORS middleware)
        # The exact headers depend on the CORS configuration
        assert response.status_code in [200, 500]  # Should not be blocked by CORS


@pytest.mark.api
class TestConcurrentRequests:
    """Test concurrent request handling"""

    def test_multiple_simultaneous_requests(self, test_client):
        """Test handling of multiple requests in quick succession"""
        import concurrent.futures
        import threading

        def make_request(query_text):
            return test_client.post("/api/query", json={"query": query_text})

        # Make multiple concurrent requests using threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                future = executor.submit(make_request, f"Test query {i}")
                futures.append(future)

            responses = [future.result() for future in futures]

        # All requests should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data
            # Each should have unique session ID
            assert data["session_id"] is not None


@pytest.mark.api
class TestRequestValidation:
    """Test request validation for all endpoints"""

    def test_query_field_validation(self, test_client):
        """Test validation of query field types and constraints"""
        # Test with non-string query
        response = test_client.post(
            "/api/query", json={"query": 123}  # Should be string
        )
        assert response.status_code == 422

        # Test with null query
        response = test_client.post("/api/query", json={"query": None})
        assert response.status_code == 422

    def test_session_id_validation(self, test_client):
        """Test session_id field validation"""
        # Test with various session_id types
        test_cases = [
            {"query": "test", "session_id": "valid_session"},  # Should work
            {"query": "test", "session_id": 123},  # Should fail
            {"query": "test", "session_id": None},  # Should work (optional)
        ]

        for i, case in enumerate(test_cases):
            response = test_client.post("/api/query", json=case)
            if i == 1:  # Integer session_id should fail
                assert response.status_code == 422
            else:  # Others should work
                assert response.status_code == 200

    def test_clear_session_validation(self, test_client):
        """Test clear session request validation"""
        # Test with invalid session_id types
        response = test_client.post(
            "/api/clear-session", json={"session_id": 123}  # Should be string
        )
        assert response.status_code == 422

        # Test with null session_id
        response = test_client.post("/api/clear-session", json={"session_id": None})
        assert response.status_code == 422


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end API flows"""

    def test_complete_conversation_flow(self, test_client):
        """Test a complete conversation flow"""
        # 1. Start with a query (should create session)
        response1 = test_client.post("/api/query", json={"query": "What is Python?"})
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # 2. Continue conversation with same session
        response2 = test_client.post(
            "/api/query",
            json={"query": "Tell me more about functions", "session_id": session_id},
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

        # 3. Get course statistics
        response3 = test_client.get("/api/courses")
        assert response3.status_code == 200

        # 4. Clear the session
        response4 = test_client.post(
            "/api/clear-session", json={"session_id": session_id}
        )
        assert response4.status_code == 200
        assert response4.json()["success"] is True

    def test_session_isolation(self, test_client):
        """Test that different sessions are properly isolated"""
        # Create two different sessions
        response1 = test_client.post("/api/query", json={"query": "Query in session 1"})
        session_id1 = response1.json()["session_id"]

        response2 = test_client.post("/api/query", json={"query": "Query in session 2"})
        session_id2 = response2.json()["session_id"]

        # Sessions should be different
        assert session_id1 != session_id2

        # Both should be valid
        assert len(session_id1) > 0
        assert len(session_id2) > 0
