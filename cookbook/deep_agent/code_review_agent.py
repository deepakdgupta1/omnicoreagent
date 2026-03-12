"""
Code Review Agent using DeepAgent.

This example demonstrates DeepAgent for:
- Code quality analysis
- Security vulnerability detection
- Performance recommendations

Run: python cookbook/deep_agent/code_review_agent.py
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

from omnicoreagent import DeepAgent  # noqa: E402
from omnicoreagent.core.tools.local_tools_registry import ToolRegistry  # noqa: E402


# =============================================================================
# TOOLS - Code Review capabilities
# =============================================================================

tools = ToolRegistry()


@tools.register_tool(
    name="analyze_code_quality",
    description="Analyze code quality metrics like complexity, docstrings, and type hints.",
    inputSchema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Source code to analyze"},
            "language": {
                "type": "string",
                "description": "Programming language (default: 'python')",
            },
        },
        "required": ["code"],
    },
)
def analyze_code_quality(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code quality metrics."""
    lines = code.strip().split("\n")

    data = {
        "language": language,
        "lines_of_code": len(lines),
        "empty_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "has_docstrings": '"""' in code or "'''" in code,
        "has_type_hints": ":" in code and "->" in code,
        "complexity_estimate": "low"
        if len(lines) < 50
        else "medium"
        if len(lines) < 200
        else "high",
    }

    return {
        "status": "success",
        "data": data,
        "message": f"Analyzed {len(lines)} lines of code.",
    }


@tools.register_tool(
    name="check_security_issues",
    description="Scan code for common security vulnerabilities like injection and hardcoded secrets.",
    inputSchema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Source code to scan"},
            "language": {"type": "string", "description": "Programming language"},
        },
        "required": ["code"],
    },
)
def check_security_issues(code: str, language: str = "python") -> Dict[str, Any]:
    """Check for common security vulnerabilities."""
    issues = []

    # Simple pattern matching for common issues
    if "eval(" in code:
        issues.append(
            {"severity": "high", "issue": "Use of eval() - potential code injection"}
        )
    if "exec(" in code:
        issues.append(
            {"severity": "high", "issue": "Use of exec() - potential code injection"}
        )
    if "pickle.load" in code:
        issues.append(
            {
                "severity": "medium",
                "issue": "Pickle deserialization - potential security risk",
            }
        )
    if "password" in code.lower() and "=" in code:
        issues.append(
            {"severity": "medium", "issue": "Possible hardcoded password detected"}
        )
    if "SELECT" in code.upper() and "+" in code:
        issues.append(
            {"severity": "high", "issue": "Possible SQL injection vulnerability"}
        )

    data = {
        "issues_found": len(issues),
        "issues": issues,
        "risk_level": "high"
        if any(i["severity"] == "high" for i in issues)
        else "medium"
        if issues
        else "low",
    }

    return {
        "status": "success",
        "data": data,
        "message": f"Found {len(issues)} security issues.",
    }


@tools.register_tool(
    name="check_best_practices",
    description="Check code against common best practices and style guidelines.",
    inputSchema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Source code to check"},
            "language": {"type": "string", "description": "Programming language"},
        },
        "required": ["code"],
    },
)
def check_best_practices(code: str, language: str = "python") -> Dict[str, Any]:
    """Check adherence to best practices."""
    recommendations = []

    if '"""' not in code and "'''" not in code:
        recommendations.append("Add docstrings to functions and classes")
    if "try:" not in code:
        recommendations.append("Consider adding error handling")
    if "import" in code and "*" in code:
        recommendations.append("Avoid wildcard imports (import *)")
    if "lambda" in code:
        recommendations.append(
            "Consider named functions instead of lambdas for complex logic"
        )
    if not any(hint in code for hint in ["-> ", ": str", ": int", ": List"]):
        recommendations.append("Add type hints for better code documentation")

    data = {
        "recommendations": recommendations,
        "best_practice_score": max(0, 100 - len(recommendations) * 15),
    }

    return {
        "status": "success",
        "data": data,
        "message": f"Generated {len(recommendations)} best practice recommendations.",
    }


@tools.register_tool(
    name="generate_review_summary",
    description="Generate a formatted summary of the code review findings.",
    inputSchema={
        "type": "object",
        "properties": {
            "quality": {"type": "object", "description": "Quality analysis data"},
            "security": {"type": "object", "description": "Security analysis data"},
            "best_practices": {"type": "object", "description": "Best practices data"},
        },
        "required": ["quality", "security", "best_practices"],
    },
)
def generate_review_summary(
    quality: Dict[str, Any], security: Dict[str, Any], best_practices: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a summary of the code review."""
    summary = f"""
## Code Review Summary

### Quality Metrics
- Lines of Code: {quality.get("lines_of_code", "N/A")}
- Complexity: {quality.get("complexity_estimate", "N/A")}
- Has Docstrings: {"Yes" if quality.get("has_docstrings") else "No"}
- Has Type Hints: {"Yes" if quality.get("has_type_hints") else "No"}

### Security
- Risk Level: {security.get("risk_level", "N/A").upper()}
- Issues Found: {security.get("issues_found", 0)}

### Best Practices Score: {best_practices.get("best_practice_score", "N/A")}/100
"""
    return {
        "status": "success",
        "data": {"summary_text": summary},
        "message": "Generated review summary.",
    }


# =============================================================================
# MAIN
# =============================================================================


async def main():
    """Run the Code Review Agent DeepAgent."""

    print("=" * 60)
    print("🔍 Code Review Agent - DeepAgent Demo")
    print("=" * 60)

    agent = DeepAgent(
        name="CodeReviewer",
        system_instruction="""
You are an expert code reviewer with focus on:
- Code quality and maintainability
- Security vulnerabilities
- Performance optimization
- Best practices adherence

Your review process:
1. Analyze the code structure and quality
2. Scan for security vulnerabilities
3. Check best practices adherence
4. Provide actionable recommendations

For large codebases, consider spawning subagents to analyze
different aspects (quality, security, performance) in parallel.

Be specific and constructive in your feedback.
""",
        model_config={
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
        },
        local_tools=tools,
    )

    await agent.initialize()
    print(f"\n✓ Agent initialized: {agent.name}")

    # Sample code to review
    sample_code = """
import os
import pickle

def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    # execute query
    return results

def process_config(config_file):
    with open(config_file, 'rb') as f:
        return pickle.load(f)

password = "admin123"

def run_script(script):
    eval(script)
"""

    print("\n📝 Code to review:")
    print("-" * 40)
    print(sample_code[:200] + "...")

    print("\n🔍 Running code review...\n")

    result = await agent.run(f"""
    Review the following Python code for quality, security, and best practices:

    ```python
    {sample_code}
    ```

    Use the available tools to:
    1. Analyze code quality
    2. Check for security issues
    3. Check best practices
    4. Generate a comprehensive review summary

    Provide specific, actionable recommendations.
    """)

    print("=" * 60)
    print("📋 CODE REVIEW RESULTS")
    print("=" * 60)
    print(result["response"])
    print("=" * 60)

    await agent.cleanup()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
