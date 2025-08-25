#!/bin/bash
# Run all code quality checks

echo "🚀 Running code quality checks..."
echo

# Check formatting
echo "📝 Checking code formatting..."
./scripts/check-format.sh
if [ $? -ne 0 ]; then
    echo "❌ Formatting check failed!"
    exit 1
fi
echo

# Run tests
echo "🧪 Running tests..."
cd backend && uv run python tests/run_all_tests.py
if [ $? -ne 0 ]; then
    echo "❌ Tests failed!"
    exit 1
fi
cd ..
echo

echo "✅ All quality checks passed!"