"""
Upwork Job Proposal Writer using DeepAgent.

This example demonstrates DeepAgent with local tools for:
- Analyzing job postings
- Finding relevant skills
- Generating tailored proposals

Run: python cookbook/deep_agent/upwork_proposal_writer.py
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any


from omnicoreagent import DeepAgent
from omnicoreagent.core.tools.local_tools_registry import ToolRegistry


# =============================================================================
# TOOLS - Domain-specific capabilities for proposal writing
# =============================================================================

tools = ToolRegistry()


@tools.register_tool(
    name="analyze_job_posting",
    description="Analyze an Upwork job posting text to extract key requirements, skills, budget type, and urgency.",
    inputSchema={
        "type": "object",
        "properties": {
            "job_description": {
                "type": "string",
                "description": "The full text of the job posting to analyze.",
            }
        },
        "required": ["job_description"],
    },
)
def analyze_job_posting(job_description: str) -> Dict[str, Any]:
    """Analyze an Upwork job posting."""
    # Simulated analysis - in production, this would use NLP
    keywords = []
    for word in [
        "Python",
        "JavaScript",
        "React",
        "API",
        "Machine Learning",
        "Data",
        "Web",
        "Mobile",
        "AI",
        "Backend",
        "Frontend",
        "Full-stack",
    ]:
        if word.lower() in job_description.lower():
            keywords.append(word)

    # Detect budget type
    budget_type = "hourly" if "hourly" in job_description.lower() else "fixed"

    # Detect urgency
    urgent = any(
        word in job_description.lower()
        for word in ["asap", "urgent", "immediately", "fast"]
    )

    return {
        "status": "success",
        "data": {
            "detected_skills": keywords,
            "budget_type": budget_type,
            "is_urgent": urgent,
            "word_count": len(job_description.split()),
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "message": f"Successfully analyzed job posting. Found {len(keywords)} skills.",
    }


@tools.register_tool(
    name="get_freelancer_profile",
    description="Retrieve the freelancer's professional profile, including skills, experience, and portfolio highlights.",
    inputSchema={
        "type": "object",
        "properties": {
            "profile_id": {
                "type": "string",
                "description": "Profile identifier (default: 'default').",
            }
        },
        "required": ["profile_id"],
    },
)
def get_freelancer_profile(profile_id: str = "default") -> Dict[str, Any]:
    """Get the freelancer's profile information."""
    # Simulated profile
    data = {
        "name": "Alex Developer",
        "title": "Senior Full-Stack Developer & AI Specialist",
        "skills": [
            "Python",
            "JavaScript",
            "TypeScript",
            "React",
            "Node.js",
            "FastAPI",
            "PostgreSQL",
            "MongoDB",
            "Docker",
            "AWS",
            "Machine Learning",
            "LLM Integration",
            "API Development",
        ],
        "years_experience": 8,
        "completed_jobs": 150,
        "success_rate": 98,
        "hourly_rate": "$75-100",
        "specialties": [
            "AI/ML Applications",
            "Full-Stack Web Development",
            "API Design & Integration",
            "Technical Architecture",
        ],
        "portfolio_highlights": [
            "Built AI-powered customer service platform (50k+ users)",
            "Developed trading analytics dashboard for hedge fund",
            "Created document processing pipeline using LLMs",
        ],
    }
    return {
        "status": "success",
        "data": data,
        "message": f"Retrieved profile for {data['name']}",
    }


@tools.register_tool(
    name="find_relevant_portfolio_items",
    description="Find portfolio items that match the required skills for the job.",
    inputSchema={
        "type": "object",
        "properties": {
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of skills required by the job (e.g. ['Python', 'React']).",
            }
        },
        "required": ["required_skills"],
    },
)
def find_relevant_portfolio_items(required_skills: List[str]) -> Dict[str, Any]:
    """Find portfolio items that match job requirements."""
    # Simulated portfolio matching
    portfolio_db = [
        {
            "title": "AI Customer Service Platform",
            "skills": ["Python", "Machine Learning", "API", "React"],
            "description": "Built an AI-powered customer service platform handling 50k+ queries/month",
            "outcome": "Reduced support tickets by 60%",
        },
        {
            "title": "E-commerce Analytics Dashboard",
            "skills": ["React", "JavaScript", "Data", "API"],
            "description": "Real-time analytics dashboard for e-commerce platform",
            "outcome": "Helped increase conversion rate by 25%",
        },
        {
            "title": "Document Processing Pipeline",
            "skills": ["Python", "AI", "Machine Learning", "Backend"],
            "description": "Automated document extraction and classification system",
            "outcome": "Processed 10k+ documents daily with 99% accuracy",
        },
        {
            "title": "Mobile Banking App",
            "skills": ["React", "Mobile", "API", "Backend"],
            "description": "Full-stack mobile banking application",
            "outcome": "Serving 100k+ active users",
        },
    ]

    relevant = []
    for item in portfolio_db:
        match_score = len(set(item["skills"]) & set(required_skills))
        if match_score > 0:
            relevant.append({**item, "relevance_score": match_score})

    # Sort by relevance
    relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_items = relevant[:3]

    return {
        "status": "success",
        "data": top_items,
        "message": f"Found {len(top_items)} relevant portfolio items.",
    }


@tools.register_tool(
    name="generate_proposal_structure",
    description="Generate optimal proposal structure and tone based on job details.",
    inputSchema={
        "type": "object",
        "properties": {
            "job_type": {
                "type": "string",
                "description": "Type of job (e.g. 'development')",
            },
            "is_urgent": {
                "type": "boolean",
                "description": "Whether the job is urgent",
            },
            "budget_type": {
                "type": "string",
                "description": "Type of budget: 'hourly' or 'fixed'",
            },
        },
        "required": ["job_type", "is_urgent", "budget_type"],
    },
)
def generate_proposal_structure(
    job_type: str, is_urgent: bool, budget_type: str
) -> Dict[str, Any]:
    """Generate proposal structure."""
    structure = {
        "opening": "Hook with relevant experience",
        "understanding": "Show you understand their needs",
        "approach": "Brief technical approach",
        "portfolio": "Relevant project examples",
        "timeline": "Realistic timeline estimate",
        "call_to_action": "Clear next step",
    }

    if is_urgent:
        structure["tone"] = "Confident, action-oriented, emphasize availability"
        structure["opening"] = "Lead with immediate availability"
    else:
        structure["tone"] = "Professional, thorough, relationship-building"

    if budget_type == "fixed":
        structure["pricing"] = "Include fixed price estimate"
    else:
        structure["pricing"] = "Mention hourly rate flexibility"

    return {
        "status": "success",
        "data": structure,
        "message": "Generated proposal structure and tone advice.",
    }


@tools.register_tool(
    name="save_proposal",
    description="Save the generated proposal to a file storage.",
    inputSchema={
        "type": "object",
        "properties": {
            "proposal_text": {
                "type": "string",
                "description": "Complete text of the proposal",
            },
            "job_title": {
                "type": "string",
                "description": "Title of the job for filing",
            },
        },
        "required": ["proposal_text", "job_title"],
    },
)
def save_proposal(proposal_text: str, job_title: str) -> Dict[str, Any]:
    """Save the proposal."""
    # In production, this would save to file/database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proposal_{timestamp}.md"

    return {
        "status": "success",
        "data": {
            "filename": filename,
            "word_count": len(proposal_text.split()),
            "character_count": len(proposal_text),
            "stored_path": f"/proposals/{filename}",
        },
        "message": f"Proposal saved as {filename}",
    }


# =============================================================================
# MAIN - Run the Upwork Proposal Writer
# =============================================================================


async def main():
    """Run the Upwork Proposal Writer DeepAgent."""

    print("=" * 60)
    print("🚀 Upwork Job Proposal Writer - DeepAgent Demo")
    print("=" * 60)

    # Create the DeepAgent with proposal writing system instruction
    agent = DeepAgent(
        name="ProposalWriter",
        system_instruction="""
You are an expert Upwork proposal writer with a 90%+ success rate in winning contracts.

Your approach:
1. Carefully analyze the job posting to understand what the client REALLY needs
2. Match the freelancer's skills and portfolio to the job requirements
3. Write proposals that are personalized, specific, and compelling
4. Keep proposals concise but impactful (150-300 words ideal)

Key principles:
- Lead with relevant experience, not generic greetings
- Show you understand their problem before offering solutions
- Include specific portfolio examples that match their needs
- End with a clear call to action

You have tools to:
- analyze_job_posting: Extract requirements from job descriptions
- get_freelancer_profile: Get your skills and experience
- find_relevant_portfolio_items: Match portfolio to job requirements
- generate_proposal_structure: Get optimal proposal structure
- save_proposal: Save the final proposal
""",
        model_config={
            "provider": "gemini",
            "model": "gemini-3-pro-preview",
        },
        local_tools=tools,
        debug=True,
    )

    await agent.initialize()
    print(f"\n✓ Agent initialized: {agent.name}")
    print(f"✓ Tools available: {len(list(tools.list_tools()))}")

    # Sample job posting
    job_posting = """
    Looking for experienced Python developer to build an AI-powered
    document processing system.

    Requirements:
    - Strong Python experience (5+ years)
    - Experience with Machine Learning and NLP
    - API development (FastAPI or Flask)
    - Database experience (PostgreSQL preferred)
    - Good communication skills

    Project: We need to process thousands of legal documents daily,
    extract key information, and classify them automatically.

    Budget: $5,000-10,000 fixed price
    Timeline: 4-6 weeks

    Please include:
    - Relevant experience with similar projects
    - Your approach to this project
    - Timeline estimate
    """

    print("\n📋 Job Posting:")
    print("-" * 40)
    print(job_posting[:300] + "...")

    print("\n🤖 Generating proposal...\n")

    result = await agent.run(f"""
    Write a winning Upwork proposal for this job posting:

    {job_posting}

    Use the available tools to:
    1. Analyze the job posting
    2. Get the freelancer profile
    3. Find relevant portfolio items
    4. Generate optimal proposal structure
    5. Write the proposal
    6. Save the proposal

    Make the proposal specific, compelling, and professional.
    """)

    print("=" * 60)
    print("📝 GENERATED PROPOSAL")
    print("=" * 60)
    print(result["response"])
    print("=" * 60)

    await agent.cleanup()
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
