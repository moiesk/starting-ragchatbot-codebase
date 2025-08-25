#!/bin/bash
# Check if Python code is properly formatted with Black

echo "🔍 Checking Python code formatting..."
uv run black --check --diff backend/ main.py

if [ $? -eq 0 ]; then
    echo "✅ All Python files are properly formatted!"
else
    echo "❌ Some files need formatting. Run './scripts/format.sh' to fix."
    exit 1
fi