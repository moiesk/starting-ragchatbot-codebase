#!/bin/bash
# Format Python code with Black

echo "🎨 Formatting Python code with Black..."
uv run black backend/ main.py

echo "✅ Code formatting complete!"