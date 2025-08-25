#!/usr/bin/env python3
"""
Comprehensive Test Runner for RAG Chatbot System
Runs all tests and provides detailed diagnostic report
"""

import unittest
import sys
import os
import time
from io import StringIO
import traceback

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class DetailedTestResult(unittest.TextTestResult):
    """Enhanced test result that captures detailed information"""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_details = []
        self.current_test = None
        self.start_time = None

    def startTest(self, test):
        super().startTest(test)
        self.current_test = test
        self.start_time = time.time()

    def stopTest(self, test):
        super().stopTest(test)
        duration = time.time() - self.start_time if self.start_time else 0

        test_info = {
            "name": str(test),
            "duration": duration,
            "status": "PASS",
            "error": None,
            "output": None,
        }

        self.test_details.append(test_info)
        self.current_test = None

    def addError(self, test, err):
        super().addError(test, err)
        if self.test_details:
            self.test_details[-1]["status"] = "ERROR"
            self.test_details[-1]["error"] = self._exc_info_to_string(err, test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.test_details:
            self.test_details[-1]["status"] = "FAIL"
            self.test_details[-1]["error"] = self._exc_info_to_string(err, test)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.test_details:
            self.test_details[-1]["status"] = "SKIP"
            self.test_details[-1]["error"] = reason


class TestRunner:
    """Comprehensive test runner with detailed reporting"""

    def __init__(self):
        self.test_modules = [
            "test_system_health",
            "test_vector_store",
            "test_search_tools",
            "test_ai_generator",
            "test_rag_system",
        ]

    def run_all_tests(self):
        """Run all test modules and return detailed results"""
        print("🚀 Starting Comprehensive RAG Chatbot Test Suite")
        print("=" * 70)

        all_results = {}
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skips = 0

        for module_name in self.test_modules:
            print(f"\n📋 Running {module_name}...")
            print("-" * 50)

            try:
                # Import the test module
                test_module = __import__(module_name)

                # Create test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(test_module)

                # Capture output
                stream = StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream, verbosity=2, resultclass=DetailedTestResult
                )

                # Run tests
                result = runner.run(suite)

                # Store results
                all_results[module_name] = {
                    "result": result,
                    "output": stream.getvalue(),
                    "test_count": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "skips": len(result.skipped) if hasattr(result, "skipped") else 0,
                }

                # Update totals
                total_tests += result.testsRun
                total_failures += len(result.failures)
                total_errors += len(result.errors)
                if hasattr(result, "skipped"):
                    total_skips += len(result.skipped)

                # Print summary for this module
                status = (
                    "✅ PASS"
                    if (len(result.failures) == 0 and len(result.errors) == 0)
                    else "❌ FAIL"
                )
                print(
                    f"{status} {module_name}: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors"
                )

            except Exception as e:
                print(f"❌ ERROR importing {module_name}: {e}")
                all_results[module_name] = {
                    "result": None,
                    "output": f"Import error: {e}",
                    "test_count": 0,
                    "failures": 0,
                    "errors": 1,
                    "skips": 0,
                }
                total_errors += 1

        # Generate comprehensive report
        self.generate_report(
            all_results, total_tests, total_failures, total_errors, total_skips
        )

        return all_results

    def generate_report(
        self, all_results, total_tests, total_failures, total_errors, total_skips
    ):
        """Generate detailed diagnostic report"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST RESULTS REPORT")
        print("=" * 70)

        # Overall summary
        overall_status = (
            "✅ SYSTEM HEALTHY"
            if (total_failures == 0 and total_errors == 0)
            else "❌ ISSUES DETECTED"
        )
        print(f"\n🎯 OVERALL STATUS: {overall_status}")
        print(f"   Total Tests: {total_tests}")
        print(f"   Failures: {total_failures}")
        print(f"   Errors: {total_errors}")
        print(f"   Skipped: {total_skips}")

        # Module-by-module breakdown
        print(f"\n📋 MODULE BREAKDOWN:")
        for module_name, result_data in all_results.items():
            status_icon = (
                "✅"
                if (result_data["failures"] == 0 and result_data["errors"] == 0)
                else "❌"
            )
            print(f"   {status_icon} {module_name}:")
            print(f"      Tests: {result_data['test_count']}")
            print(f"      Failures: {result_data['failures']}")
            print(f"      Errors: {result_data['errors']}")
            print(f"      Skipped: {result_data['skips']}")

        # Detailed failure analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        critical_issues = []

        for module_name, result_data in all_results.items():
            result = result_data["result"]
            if result is None:
                critical_issues.append(f"❌ {module_name}: Module import failed")
                continue

            # Report failures
            if result.failures:
                for test, error in result.failures:
                    issue = f"❌ FAILURE in {module_name}: {test}"
                    critical_issues.append(issue)
                    print(f"   {issue}")
                    print(
                        f"      Error: {error.split('AssertionError:')[-1].strip()[:200]}..."
                    )

            # Report errors
            if result.errors:
                for test, error in result.errors:
                    issue = f"🚨 ERROR in {module_name}: {test}"
                    critical_issues.append(issue)
                    print(f"   {issue}")
                    print(
                        f"      Error: {str(error).split('Exception:')[-1].strip()[:200]}..."
                    )

        # Diagnostic suggestions
        print(f"\n🩺 DIAGNOSTIC SUGGESTIONS:")

        if "test_system_health" in all_results:
            health_result = all_results["test_system_health"]["result"]
            if health_result and (health_result.failures or health_result.errors):
                print("   ⚠️  System Health Issues Detected:")
                print("      - Check ANTHROPIC_API_KEY configuration")
                print("      - Verify ChromaDB database exists and has data")
                print("      - Ensure all dependencies are installed")

        if "test_vector_store" in all_results:
            vs_result = all_results["test_vector_store"]["result"]
            if vs_result and (vs_result.failures or vs_result.errors):
                print("   ⚠️  Vector Store Issues Detected:")
                print("      - Check if course documents are loaded")
                print("      - Verify ChromaDB collections exist")
                print("      - Test embedding model availability")

        if "test_search_tools" in all_results:
            tools_result = all_results["test_search_tools"]["result"]
            if tools_result and (tools_result.failures or tools_result.errors):
                print("   ⚠️  Search Tools Issues Detected:")
                print("      - Verify tool definitions are correct")
                print("      - Check vector store integration")
                print("      - Test result formatting")

        if "test_ai_generator" in all_results:
            ai_result = all_results["test_ai_generator"]["result"]
            if ai_result and (ai_result.failures or ai_result.errors):
                print("   ⚠️  AI Generator Issues Detected:")
                print("      - Verify Anthropic API key and access")
                print("      - Check tool calling configuration")
                print("      - Test model parameters")

        if "test_rag_system" in all_results:
            rag_result = all_results["test_rag_system"]["result"]
            if rag_result and (rag_result.failures or rag_result.errors):
                print("   ⚠️  RAG System Issues Detected:")
                print("      - Check end-to-end integration")
                print("      - Verify session management")
                print("      - Test query processing flow")

        # Final recommendations
        print(f"\n💡 NEXT STEPS:")
        if critical_issues:
            print("   1. Address the critical issues listed above")
            print("   2. Run individual test modules for detailed debugging")
            print("   3. Check system logs for additional error information")
            print("   4. Verify all configuration settings")
        else:
            print("   ✅ All tests passed - system appears healthy!")
            print("   🔍 If you're still seeing 'query failed' errors:")
            print("      - Check the actual application logs")
            print("      - Test with real user queries")
            print("      - Verify frontend-backend communication")

        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    runner = TestRunner()
    results = runner.run_all_tests()

    # Return exit code based on test results
    total_failures = sum(r["failures"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())

    if total_failures > 0 or total_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
