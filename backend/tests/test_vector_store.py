"""
Vector Store Tests - Verify data loading and search functionality
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore(unittest.TestCase):
    """Test VectorStore functionality and data integrity"""

    def setUp(self):
        """Set up test environment"""
        self.vector_store = None
        try:
            self.vector_store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )
        except Exception as e:
            self.skipTest(f"Cannot initialize VectorStore: {e}")

    def test_vector_store_has_data(self):
        """Test if vector store contains course data"""
        course_count = self.vector_store.get_course_count()
        course_titles = self.vector_store.get_existing_course_titles()

        print(f"\n📊 Vector Store Status:")
        print(f"   Course count: {course_count}")
        print(f"   Course titles: {course_titles}")

        # We should have some courses loaded
        self.assertGreater(course_count, 0, "No courses found in vector store")
        self.assertGreater(len(course_titles), 0, "No course titles found")

    def test_basic_search_functionality(self):
        """Test basic search without filters"""
        # Test with a general query
        results = self.vector_store.search("introduction")

        print(f"\n🔍 Basic Search Results for 'introduction':")
        print(f"   Documents found: {len(results.documents)}")
        print(f"   Error: {results.error}")

        if results.error:
            self.fail(f"Search failed with error: {results.error}")

        # Should find some results for "introduction"
        self.assertGreater(
            len(results.documents), 0, "No results found for 'introduction'"
        )

        # Check metadata structure
        if results.documents:
            self.assertEqual(
                len(results.documents),
                len(results.metadata),
                "Metadata count doesn't match document count",
            )

            # Print first result for debugging
            print(
                f"   First result metadata: {results.metadata[0] if results.metadata else 'None'}"
            )
            print(f"   First result content preview: {results.documents[0][:100]}...")

    def test_course_name_resolution(self):
        """Test course name resolution functionality"""
        # Get existing course titles
        course_titles = self.vector_store.get_existing_course_titles()

        if not course_titles:
            self.skipTest("No course titles available for testing")

        # Test exact match
        first_title = course_titles[0]
        resolved = self.vector_store._resolve_course_name(first_title)

        print(f"\n📝 Course Name Resolution:")
        print(f"   Original title: {first_title}")
        print(f"   Resolved title: {resolved}")

        self.assertEqual(resolved, first_title, "Exact course name resolution failed")

        # Test partial match (if we have a multi-word title)
        if len(first_title.split()) > 1:
            partial = first_title.split()[0]  # First word
            resolved_partial = self.vector_store._resolve_course_name(partial)
            print(f"   Partial search '{partial}' resolved to: {resolved_partial}")

            # Should resolve to something (may not be exact match due to semantic search)
            self.assertIsNotNone(
                resolved_partial, "Partial course name resolution failed"
            )

    def test_filtered_search(self):
        """Test search with course name filter"""
        # Get a course title to filter by
        course_titles = self.vector_store.get_existing_course_titles()

        if not course_titles:
            self.skipTest("No course titles available for testing")

        first_title = course_titles[0]

        # Search with course filter
        results = self.vector_store.search("introduction", course_name=first_title)

        print(f"\n🔍 Filtered Search Results:")
        print(f"   Course filter: {first_title}")
        print(f"   Documents found: {len(results.documents)}")
        print(f"   Error: {results.error}")

        if results.error:
            self.fail(f"Filtered search failed: {results.error}")

        # Check that all results are from the specified course
        for metadata in results.metadata:
            self.assertEqual(
                metadata.get("course_title"),
                first_title,
                "Result from wrong course found",
            )

    def test_lesson_number_filter(self):
        """Test search with lesson number filter"""
        # Search with lesson filter
        results = self.vector_store.search("lesson", lesson_number=1)

        print(f"\n🔍 Lesson Filtered Search Results:")
        print(f"   Lesson filter: 1")
        print(f"   Documents found: {len(results.documents)}")
        print(f"   Error: {results.error}")

        if results.error and "No results found" not in results.error:
            self.fail(f"Lesson filtered search failed: {results.error}")

        # Check that all results are from lesson 1 (if any found)
        for metadata in results.metadata:
            self.assertEqual(
                metadata.get("lesson_number"), 1, "Result from wrong lesson found"
            )

    def test_combined_filters(self):
        """Test search with both course and lesson filters"""
        course_titles = self.vector_store.get_existing_course_titles()

        if not course_titles:
            self.skipTest("No course titles available for testing")

        first_title = course_titles[0]

        results = self.vector_store.search(
            "content", course_name=first_title, lesson_number=0
        )

        print(f"\n🔍 Combined Filter Search Results:")
        print(f"   Course filter: {first_title}")
        print(f"   Lesson filter: 0")
        print(f"   Documents found: {len(results.documents)}")
        print(f"   Error: {results.error}")

        if results.error and "No results found" not in results.error:
            self.fail(f"Combined filtered search failed: {results.error}")

        # Check filters are applied correctly
        for metadata in results.metadata:
            self.assertEqual(metadata.get("course_title"), first_title)
            self.assertEqual(metadata.get("lesson_number"), 0)

    def test_invalid_course_name(self):
        """Test search with invalid course name"""
        results = self.vector_store.search("test", course_name="NonExistentCourse12345")

        print(f"\n❌ Invalid Course Name Test:")
        print(f"   Error: {results.error}")
        print(f"   Documents found: {len(results.documents)}")

        # Should return error for invalid course
        self.assertIsNotNone(results.error, "Expected error for invalid course name")
        self.assertIn("No course found", results.error)
        self.assertEqual(
            len(results.documents), 0, "Should not return documents for invalid course"
        )

    def test_empty_query(self):
        """Test search with empty query"""
        results = self.vector_store.search("")

        print(f"\n❓ Empty Query Test:")
        print(f"   Error: {results.error}")
        print(f"   Documents found: {len(results.documents)}")

        # Empty query might still work with semantic search, but check for reasonable behavior
        if results.error:
            print(
                f"   Empty query returned error (this may be expected): {results.error}"
            )

    def test_course_metadata_retrieval(self):
        """Test course metadata retrieval functionality"""
        all_courses_metadata = self.vector_store.get_all_courses_metadata()

        print(f"\n📚 Course Metadata Test:")
        print(f"   Total courses with metadata: {len(all_courses_metadata)}")

        if all_courses_metadata:
            first_course = all_courses_metadata[0]
            print(f"   First course keys: {list(first_course.keys())}")
            print(f"   First course title: {first_course.get('title')}")
            print(f"   First course instructor: {first_course.get('instructor')}")
            print(f"   First course has lessons: {'lessons' in first_course}")

            if "lessons" in first_course:
                lessons = first_course["lessons"]
                print(f"   Number of lessons: {len(lessons) if lessons else 0}")
                if lessons:
                    print(f"   First lesson: {lessons[0]}")

        self.assertGreater(len(all_courses_metadata), 0, "No course metadata found")

    def test_lesson_link_retrieval(self):
        """Test lesson link retrieval functionality"""
        course_titles = self.vector_store.get_existing_course_titles()

        if not course_titles:
            self.skipTest("No course titles available for testing")

        first_title = course_titles[0]

        # Try to get lesson 0 link
        lesson_link = self.vector_store.get_lesson_link(first_title, 0)

        print(f"\n🔗 Lesson Link Test:")
        print(f"   Course: {first_title}")
        print(f"   Lesson 0 link: {lesson_link}")

        # Link might be None if not set, but method should not crash
        if lesson_link:
            self.assertIsInstance(lesson_link, str, "Lesson link should be string")

    def test_search_results_structure(self):
        """Test SearchResults class functionality"""
        # Test empty results
        empty_results = SearchResults.empty("Test error")

        self.assertTrue(empty_results.is_empty())
        self.assertEqual(empty_results.error, "Test error")
        self.assertEqual(len(empty_results.documents), 0)

        # Test from_chroma conversion
        mock_chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(mock_chroma_results)

        self.assertFalse(results.is_empty())
        self.assertEqual(len(results.documents), 2)
        self.assertEqual(len(results.metadata), 2)
        self.assertEqual(len(results.distances), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
