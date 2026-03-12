"""
OmniRex Due Diligence Tools - CTO Pitch Quality
---------------------------------------------
Premium tools for generating financial charts, infographics, and HTML reports.
Production-grade implementation for OmniRexFlora Labs - AI-powered SME funding infrastructure.

Key Features:
- Goldman Sachs/McKinsey-style HTML reports
- Professional infographic dashboards (150 DPI)
- Multi-scenario financial projections
- Africa-specific macro risk assessment
- Competitive landscape analysis
"""

import os
import logging
import json
import ast
import re
import textwrap
from datetime import datetime
from typing import Union, List, Dict, Any
import requests

from omnicoreagent import ToolRegistry

# Configure logging
logger = logging.getLogger("OmniRexTools")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _tavily_search(query: str, max_results: int = 5) -> list:
    """Execute real-time search via Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "max_results": max_results,
            },
            timeout=15,
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return []


def ensure_string(content: Any) -> str:
    """
    Robustly ensure content is a string.
    Handles lists, stringified lists, and other types.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        clean_content = content.strip()
        if clean_content.startswith("[") and clean_content.endswith("]"):
            try:
                parsed = ast.literal_eval(clean_content)
                if isinstance(parsed, list):
                    return "\n".join(str(x) for x in parsed)
            except (ValueError, SyntaxError):
                pass
        return content

    if isinstance(content, list):
        return "\n".join(str(x) for x in content)

    return str(content)


def clean_highlight_text(text: str) -> str:
    """
    Clean highlight text for dashboard display.
    - Strip XML/HTML tags like <item>, </item>
    - Decode HTML entities like &amp; -> &
    - Remove extra whitespace
    """
    import html

    if not text:
        return ""

    # Strip XML/HTML tags (e.g., <item>, </item>, <highlight>)
    cleaned = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    cleaned = html.unescape(cleaned)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def get_output_dir() -> str:
    """Returns the absolute path to the outputs directory."""
    env_dir = os.getenv("OMNIREX_OUTPUT_DIR")
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir

    base_dir = os.getcwd()
    if "cookbook" in base_dir:
        while "cookbook" in os.path.basename(base_dir):
            base_dir = os.path.dirname(base_dir)

    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def robust_to_html(text: str) -> str:
    """
    Convert markdown to HTML with robust fallback.
    Produces clean, professional HTML output.
    """
    if text:
        text = textwrap.dedent(text)

    try:
        import markdown

        return markdown.markdown(
            text, extensions=["tables", "fenced_code", "nl2br", "sane_lists", "smarty"]
        )
    except ImportError:
        pass

    # Fallback: Custom parsing
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2).strip()
        return f'<pre><code class="language-{lang}">{code}</code></pre>'

    text = re.sub(r"```(\w*)\n(.*?)```", replace_code_block, text, flags=re.DOTALL)

    lines = text.split("\n")
    processed_lines = []
    in_list = False
    list_type = None

    for line in lines:
        stripped = line.strip()

        # Headers
        header_match = re.match(r"^(\s*)(#{1,6})\s+(.+)$", line)
        if header_match:
            if in_list:
                processed_lines.append(f"</{list_type}>")
                in_list = False
            level = len(header_match.group(2))
            content = header_match.group(3)
            processed_lines.append(f"<h{level}>{content}</h{level}>")
            continue

        # Horizontal Rules
        if re.match(r"^\s*([-*]){3,}\s*$", line):
            if in_list:
                processed_lines.append(f"</{list_type}>")
                in_list = False
            processed_lines.append("<hr>")
            continue

        # Blockquotes
        if line.startswith("> "):
            if in_list:
                processed_lines.append(f"</{list_type}>")
                in_list = False
            processed_lines.append(f"<blockquote>{line[2:].strip()}</blockquote>")
            continue

        # Lists
        ul_match = re.match(r"^\s*[\-\*]\s+(.+)$", line)
        ol_match = re.match(r"^\s*\d+\.\s+(.+)$", line)

        if ul_match or ol_match:
            current_type = "ul" if ul_match else "ol"
            content = ul_match.group(1) if ul_match else ol_match.group(1)

            if not in_list:
                processed_lines.append(f"<{current_type}>")
                in_list = True
                list_type = current_type
            elif list_type != current_type:
                processed_lines.append(f"</{list_type}>")
                processed_lines.append(f"<{current_type}>")
                list_type = current_type

            processed_lines.append(f"<li>{content}</li>")
            continue

        if in_list and stripped != "":
            processed_lines.append(f"</{list_type}>")
            in_list = False

        if stripped == "":
            processed_lines.append("")
        else:
            processed_lines.append(line)

    if in_list:
        processed_lines.append(f"</{list_type}>")

    html = "\n".join(processed_lines)

    # Inline formatting
    html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    html = re.sub(r"_(.+?)_", r"<em>\1</em>", html)

    # Links and Code
    html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

    # Paragraph wrapping
    paragraphs = html.split("\n\n")
    final_html = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if (
            p.startswith("<h")
            or p.startswith("<ul")
            or p.startswith("<ol")
            or p.startswith("<hr")
            or p.startswith("<pre")
            or p.startswith("<blockquote")
        ):
            final_html.append(p)
        else:
            p = p.replace("\n", "<br>\n")
            final_html.append(f"<p>{p}</p>")

    return "\n".join(final_html)


# =============================================================================
# MASTER TOOL REGISTRY CREATOR
# =============================================================================


def create_omnirex_tools() -> ToolRegistry:
    """
    Creates a SINGLE registry containing ALL OmniRex Due Diligence tools.

    Tools included:
    1. generate_html_report - Professional investment report
    2. generate_dashboard_infographic - Goldman Sachs-style one-pager
    3. generate_financial_chart - Revenue projections (Bear/Base/Bull)
    4. save_evaluation_memo - Investment decision memo
    5. analyze_competitor_landscape - Feature matrix vs competitors
    6. assess_macro_risk - Country-specific risk assessment
    """
    registry = ToolRegistry()

    # Check matplotlib availability
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Headless mode
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    # =========================================================================
    # TOOL 1: PREMIUM HTML REPORT
    # =========================================================================
    @registry.register_tool(
        name="generate_html_report",
        description="""
        Generate a professional HTML investment report styled like McKinsey/Goldman Sachs.

        **When to use:**
        Use this tool as the FINAL STEP to compile all due diligence findings into a
        polished, investor-ready HTML document. This tool does NOT use an LLM - it directly
        formats your content into a professional template.

        **What it produces:**
        An HTML file with:
        - Professional header with company name and report date
        - Executive summary section with gold accent styling
        - Embedded infographic dashboard (if path provided)
        - Embedded financial charts (if path provided)
        - Structured sections (Financial, Market, Team, Risks, etc.)
        - Full markdown support (headers, bold, italic, lists, tables, code)
        - Print-friendly layout for PDF export
        - OmniRex branded footer with confidentiality notice

        **How to use:**
        1. Compile your due diligence findings into markdown format
        2. Generate infographic first using generate_dashboard_infographic
        3. Generate financial chart using generate_financial_chart
        4. Call this tool with the paths to those images
        5. Provide sections dict for structured report

        **Example with sections:**
        {
            "company_name": "Paystack",
            "report_title": "Series A Investment Evaluation",
            "content": "Paystack demonstrates strong product-market fit...",
            "sections": {
                "Company Overview": "Founded in 2015, Paystack is...",
                "Market Opportunity": "African payments TAM is $50B...",
                "Financial Analysis": "Current ARR of $5.2M with 127% growth...",
                "Team Assessment": "Founders have strong execution track record...",
                "Risk Assessment": "Key risks include regulatory and FX...",
                "Recommendation": "We recommend FUND based on..."
            },
            "dashboard_path": "/path/to/infographic.png",
            "chart_path": "/path/to/chart.png"
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Name of the company being evaluated.",
                },
                "report_title": {
                    "type": "string",
                    "description": "Report title (e.g., 'Series A Investment Evaluation').",
                },
                "content": {
                    "type": "string",
                    "description": "Executive Summary content in markdown format.",
                },
                "sections": {
                    "type": "object",
                    "description": "Dict of section titles to content. Keys become section headers.",
                    "additionalProperties": {"type": "string"},
                },
                "dashboard_path": {
                    "type": "string",
                    "description": "Absolute path to the infographic PNG from generate_dashboard_infographic.",
                },
                "chart_path": {
                    "type": "string",
                    "description": "Absolute path to the financial chart PNG from generate_financial_chart.",
                },
            },
            "required": ["company_name", "report_title", "content"],
            "additionalProperties": False,
        },
    )
    def generate_html_report(
        company_name: str,
        report_title: str,
        content: Union[str, List[Any]],
        sections: Dict[str, str] = None,
        dashboard_path: str = None,
        chart_path: str = None,
    ) -> dict:
        """Generate professional HTML investment report."""

        # Clean inputs
        company_name = ensure_string(company_name)
        report_title = ensure_string(report_title)
        main_content = ensure_string(content)
        current_date = datetime.now().strftime("%B %d, %Y")

        # Sanitize paths
        if dashboard_path and (
            dashboard_path.startswith("N/A")
            or dashboard_path.lower() == "none"
            or "failed" in dashboard_path.lower()
        ):
            dashboard_path = None
        if chart_path and (
            chart_path.startswith("N/A")
            or chart_path.lower() == "none"
            or "failed" in chart_path.lower()
        ):
            chart_path = None

        # Build sections HTML
        sections_html = ""
        if sections:
            if isinstance(sections, str):
                try:
                    sections = json.loads(sections)
                except:
                    parsed_sections = {}
                    matches = re.findall(
                        r"<([a-zA-Z0-9_]+)>\s*(.*?)\s*</\1>", sections, re.DOTALL
                    )
                    if matches:
                        for tag, tag_content in matches:
                            title = tag.replace("_", " ").replace("And", "&").title()
                            parsed_sections[title] = tag_content.strip()
                        sections = parsed_sections
                    elif sections.strip():
                        sections = {"Detailed Analysis": sections}
                    else:
                        sections = {}

            if isinstance(sections, dict):
                for title, sec_content in sections.items():
                    title = ensure_string(title)
                    sec_content = ensure_string(sec_content)
                    sections_html += f"""
                    <section class="report-section">
                        <h2>{title}</h2>
                        {robust_to_html(sec_content)}
                    </section>
                    """

        # Dashboard embed HTML
        dashboard_html = ""
        if dashboard_path and os.path.exists(dashboard_path):
            rel_path = os.path.basename(dashboard_path)
            dashboard_html = f'''
            <figure class="dashboard-figure">
                <img src="{rel_path}" alt="Investment Dashboard" />
                <figcaption>Key Metrics Dashboard</figcaption>
            </figure>
            '''

        # Chart embed HTML
        chart_html = ""
        if chart_path and os.path.exists(chart_path):
            rel_path = os.path.basename(chart_path)
            chart_html = f"""
            <section class="report-section">
                <h2>Financial Projections</h2>
                <figure class="chart-figure">
                    <img src="{rel_path}" alt="Revenue Projections" />
                    <figcaption>Revenue Projection Analysis (Bear/Base/Bull Scenarios)</figcaption>
                </figure>
            </section>
            """

        # Premium HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Report - {company_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Georgia&display=swap" rel="stylesheet">
    <style>
        /* === CSS RESET === */
        *, *::before, *::after {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        /* === BASE STYLES === */
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.7;
            color: #333;
            background: #fff;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 60px;
        }}

        /* === HEADER === */
        .header {{
            border-bottom: 3px solid #d4af37;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}

        .header-meta {{
            font-family: 'Inter', sans-serif;
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }}

        h1 {{
            font-family: 'Inter', sans-serif;
            color: #1a365d;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}

        .date {{
            font-family: 'Inter', sans-serif;
            color: #888;
            font-style: italic;
            font-size: 0.9em;
        }}

        /* === SECTION HEADERS === */
        h2 {{
            font-family: 'Inter', sans-serif;
            color: #1a365d;
            font-size: 1.4em;
            margin-top: 35px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }}

        h3 {{
            font-family: 'Inter', sans-serif;
            color: #2d4a6f;
            font-size: 1.1em;
            margin-top: 25px;
            margin-bottom: 10px;
        }}

        /* === PARAGRAPHS === */
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}

        /* === EXECUTIVE SUMMARY === */
        .executive-summary {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #d4af37;
            padding: 25px 30px;
            margin: 30px 0;
            border-radius: 0 8px 8px 0;
        }}

        .executive-summary h2 {{
            margin-top: 0;
            border-bottom: none;
            color: #1a365d;
            font-size: 1.2em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* === REPORT SECTIONS === */
        .report-section {{
            margin-bottom: 30px;
        }}

        /* === TABLES === */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: 'Inter', sans-serif;
            font-size: 0.9em;
        }}

        th {{
            text-align: left;
            background: #1a365d;
            color: white;
            padding: 12px 15px;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}

        tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        /* === LISTS === */
        ul, ol {{
            margin: 15px 0 15px 25px;
        }}

        li {{
            margin-bottom: 8px;
        }}

        /* === CODE === */
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            font-family: 'Monaco', 'Consolas', monospace;
        }}

        pre {{
            background: #2d3748;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
        }}

        /* === FIGURES & IMAGES === */
        figure {{
            margin: 30px 0;
            text-align: center;
        }}

        figure img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        figcaption {{
            font-family: 'Inter', sans-serif;
            font-size: 0.85em;
            color: #666;
            margin-top: 12px;
            font-style: italic;
        }}

        .dashboard-figure {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin: 30px 0;
        }}

        /* === RECOMMENDATION BOX === */
        .recommendation {{
            background: #1a365d;
            color: white;
            padding: 25px 30px;
            margin: 30px 0;
            text-align: center;
            border-radius: 8px;
        }}

        .recommendation h2 {{
            color: #d4af37;
            border: none;
            margin: 0 0 10px 0;
        }}

        /* === FOOTER === */
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #1a365d;
            text-align: center;
            font-family: 'Inter', sans-serif;
        }}

        .footer-brand {{
            color: #1a365d;
            font-weight: 600;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}

        .footer-legal {{
            color: #888;
            font-size: 0.8em;
            font-style: italic;
        }}

        /* === PRINT STYLES === */
        @media print {{
            body {{
                padding: 20px;
            }}
            .page-break {{
                page-break-before: always;
            }}
            figure img {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-meta">OmniRexFlora Labs • Investment Due Diligence Report</div>
        <h1>{company_name}</h1>
        <div class="date">{report_title} • {current_date}</div>
    </header>

    <main>
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            {dashboard_html}
            {robust_to_html(main_content)}
        </section>

        {chart_html}

        {sections_html if sections_html else ""}
    </main>

    <footer class="footer">
        <div class="footer-brand">Generated by OmniRexFlora DeepAgent</div>
        <div class="footer-legal">
            Confidential • For Investment Purposes Only • {datetime.now().year}
        </div>
    </footer>
</body>
</html>"""

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"omnirex_report_{company_name.replace(' ', '_')}_{timestamp}.html"
        filepath = os.path.join(get_output_dir(), filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info(f"HTML report generated: {filepath}")
        return {
            "status": "success",
            "message": f"Professional report saved: {filepath}",
            "data": {"filepath": filepath},
        }

    # =========================================================================
    # TOOL 2: PROFESSIONAL INFOGRAPHIC DASHBOARD
    # =========================================================================
    @registry.register_tool(
        name="generate_dashboard_infographic",
        description="""
        Generate a professional Goldman Sachs-style investment infographic (one-pager).

        **When to use:**
        Use this tool to create a visually stunning, investor-ready summary of a company's
        key metrics. This is ideal for executive presentations, pitch decks, or investment
        committee materials. Generate this BEFORE the HTML report.

        **What it produces:**
        A high-resolution PNG image (150 DPI, 16:9 aspect) containing:
        - Dark blue header with company name and date
        - Four metric cards: Valuation, ARR, YoY Growth, Market Size (TAM)
        - Color-coded risk gauge (green to red scale, 1-10)
        - Investment recommendation badge (BUY=green, HOLD=orange, PASS=red)
        - Up to 4 key highlights with gold bullet points
        - Professional footer with CONFIDENTIAL notice

        **How to use:**
        1. Gather the key financial metrics from your research
        2. Format values with currency symbols and units (e.g., "$50M", "127%", "$8.3B")
        3. Set risk_score between 1 (low risk) and 10 (high risk)
        4. Choose recommendation: "BUY", "HOLD", or "PASS"
        5. Provide 2-4 key highlights as brief bullet points

        **Example:**
        {
            "company_name": "Flutterwave",
            "valuation": "$3B",
            "arr": "$250M",
            "growth_rate": "85%",
            "market_size": "$40B",
            "risk_score": 4.5,
            "recommendation": "BUY",
            "key_highlights": [
                "Largest payment processor in Africa",
                "Licensed in 14 countries",
                "Strong enterprise customer base",
                "Clear path to profitability"
            ]
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name displayed prominently in header.",
                },
                "valuation": {
                    "type": "string",
                    "description": "Current valuation with currency (e.g., '$50M', '$3B').",
                },
                "arr": {
                    "type": "string",
                    "description": "Annual Recurring Revenue with currency (e.g., '$5.2M').",
                },
                "growth_rate": {
                    "type": "string",
                    "description": "Year-over-year growth rate (e.g., '127%', '85%').",
                },
                "market_size": {
                    "type": "string",
                    "description": "Total Addressable Market (e.g., '$8.3B', '$50B').",
                },
                "risk_score": {
                    "type": "number",
                    "description": "Risk score from 1 (low) to 10 (high).",
                },
                "recommendation": {
                    "type": "string",
                    "enum": [
                        "BUY",
                        "HOLD",
                        "PASS",
                        "STRONG BUY",
                        "INVEST",
                        "MONITOR",
                        "FUND",
                        "CONDITIONAL",
                    ],
                    "description": "Investment recommendation.",
                },
                "key_highlights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 key highlights or bullet points.",
                },
            },
            "required": [
                "company_name",
                "valuation",
                "arr",
                "growth_rate",
                "market_size",
                "risk_score",
                "recommendation",
            ],
            "additionalProperties": False,
        },
    )
    def generate_dashboard_infographic(
        company_name: str,
        valuation: str,
        arr: str,
        growth_rate: str,
        market_size: str,
        risk_score: float,
        recommendation: str,
        key_highlights: List[str] = None,
    ) -> dict:
        """Generate Goldman Sachs-style investment infographic."""
        if not HAS_MATPLOTLIB:
            return {"status": "error", "message": "Matplotlib is not installed."}

        try:
            import numpy as np
            from matplotlib.patches import FancyBboxPatch, Circle, Wedge
            import matplotlib.patheffects as path_effects

            # Ensure clean string values
            company_name = ensure_string(company_name)
            valuation = ensure_string(valuation)
            arr = ensure_string(arr)
            growth_rate = ensure_string(growth_rate)
            market_size = ensure_string(market_size)
            recommendation = ensure_string(recommendation).upper().strip()

            # Normalize risk score
            try:
                risk_score = float(risk_score)
                risk_score = max(1, min(10, risk_score))
            except:
                risk_score = 5.0

            # Parse highlights - clean XML tags and HTML entities
            highlights = []
            if key_highlights:
                if isinstance(key_highlights, list):
                    highlights = [
                        clean_highlight_text(ensure_string(h))
                        for h in key_highlights[:4]
                    ]
                else:
                    h_str = ensure_string(key_highlights)
                    if "\n" in h_str:
                        highlights = [
                            clean_highlight_text(h.strip())
                            for h in h_str.split("\n")
                            if h.strip()
                        ][:4]
                    elif "," in h_str:
                        highlights = [
                            clean_highlight_text(h.strip())
                            for h in h_str.split(",")
                            if h.strip()
                        ][:4]
                    else:
                        highlights = [clean_highlight_text(h_str)]
            # Filter out empty highlights after cleaning
            highlights = [h for h in highlights if h]

            # Color scheme (OmniRex branding)
            DARK_BLUE = "#1a365d"
            GOLD = "#d4af37"
            LIGHT_GRAY = "#f8f9fa"
            MEDIUM_GRAY = "#6c757d"
            WHITE = "#ffffff"
            GREEN = "#198754"
            RED = "#dc3545"
            ORANGE = "#fd7e14"

            # Create figure (16:9 aspect for presentations)
            fig = plt.figure(figsize=(16, 9), facecolor=WHITE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 16)
            ax.set_ylim(0, 9)
            ax.set_aspect("equal")
            ax.axis("off")

            # === HEADER SECTION ===
            header_bg = FancyBboxPatch(
                (0, 7.5),
                16,
                1.5,
                boxstyle="square,pad=0",
                facecolor=DARK_BLUE,
                edgecolor="none",
            )
            ax.add_patch(header_bg)

            # Company name
            ax.text(
                0.5,
                8.25,
                company_name.upper(),
                fontsize=32,
                fontweight="bold",
                color=WHITE,
                va="center",
                ha="left",
                path_effects=[
                    path_effects.withStroke(linewidth=1, foreground=DARK_BLUE)
                ],
            )

            # Subtitle
            ax.text(
                0.5,
                7.75,
                "INVESTMENT DUE DILIGENCE DASHBOARD",
                fontsize=12,
                color=GOLD,
                va="center",
                ha="left",
                style="italic",
            )

            # Date on right
            ax.text(
                15.5,
                8.0,
                datetime.now().strftime("%B %d, %Y"),
                fontsize=11,
                color=WHITE,
                va="center",
                ha="right",
            )

            # === METRIC CARDS ROW ===
            def draw_metric_card(x, y, w, h, label, value, accent_color=GOLD):
                """Draw a professional metric card."""
                card = FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.03,rounding_size=0.1",
                    facecolor=WHITE,
                    edgecolor="#e0e0e0",
                    linewidth=1,
                )
                ax.add_patch(card)
                # Top accent line
                ax.plot(
                    [x + 0.1, x + w - 0.1],
                    [y + h - 0.1, y + h - 0.1],
                    color=accent_color,
                    linewidth=4,
                    solid_capstyle="round",
                )
                # Label
                ax.text(
                    x + w / 2,
                    y + h - 0.45,
                    label.upper(),
                    fontsize=10,
                    color=MEDIUM_GRAY,
                    va="center",
                    ha="center",
                    fontweight="medium",
                )
                # Value
                ax.text(
                    x + w / 2,
                    y + h / 2 - 0.15,
                    value,
                    fontsize=24,
                    color=DARK_BLUE,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )

            # Draw 4 metric cards
            card_y = 5.5
            card_h = 1.8
            card_w = 3.5
            gap = 0.4
            start_x = 0.5

            draw_metric_card(start_x, card_y, card_w, card_h, "Valuation", valuation)
            draw_metric_card(
                start_x + card_w + gap, card_y, card_w, card_h, "ARR", arr, GREEN
            )
            draw_metric_card(
                start_x + 2 * (card_w + gap),
                card_y,
                card_w,
                card_h,
                "YoY Growth",
                growth_rate,
                GREEN,
            )
            draw_metric_card(
                start_x + 3 * (card_w + gap),
                card_y,
                card_w,
                card_h,
                "Market Size (TAM)",
                market_size,
            )

            # === RISK GAUGE ===
            gauge_center_x = 3.5
            gauge_center_y = 3.2
            gauge_radius = 1.8

            # Background arc with color segments
            theta_start = 180
            for i in range(10):
                segment_color = plt.cm.RdYlGn_r(i / 9)
                wedge = Wedge(
                    (gauge_center_x, gauge_center_y),
                    gauge_radius,
                    theta_start + i * 18,
                    theta_start + (i + 1) * 18,
                    width=0.4,
                    facecolor=segment_color,
                    edgecolor=WHITE,
                    linewidth=1,
                )
                ax.add_patch(wedge)

            # Needle
            needle_angle = 180 + (risk_score - 1) * 18
            needle_rad = np.radians(needle_angle)
            needle_len = gauge_radius - 0.5
            ax.arrow(
                gauge_center_x,
                gauge_center_y,
                needle_len * np.cos(needle_rad),
                needle_len * np.sin(needle_rad),
                head_width=0.15,
                head_length=0.1,
                fc=DARK_BLUE,
                ec=DARK_BLUE,
                linewidth=2,
            )

            # Center circle
            center_circle = Circle(
                (gauge_center_x, gauge_center_y),
                0.2,
                facecolor=DARK_BLUE,
                edgecolor=WHITE,
                linewidth=2,
            )
            ax.add_patch(center_circle)

            # Risk score label
            ax.text(
                gauge_center_x,
                gauge_center_y - 1.0,
                "RISK SCORE",
                fontsize=10,
                color=MEDIUM_GRAY,
                va="center",
                ha="center",
                fontweight="medium",
            )
            ax.text(
                gauge_center_x,
                gauge_center_y - 1.4,
                f"{risk_score:.1f}/10",
                fontsize=20,
                color=DARK_BLUE,
                va="center",
                ha="center",
                fontweight="bold",
            )

            # Risk labels
            ax.text(
                gauge_center_x - gauge_radius - 0.3,
                gauge_center_y - 0.2,
                "LOW",
                fontsize=8,
                color=GREEN,
                va="center",
                ha="center",
                fontweight="bold",
            )
            ax.text(
                gauge_center_x + gauge_radius + 0.3,
                gauge_center_y - 0.2,
                "HIGH",
                fontsize=8,
                color=RED,
                va="center",
                ha="center",
                fontweight="bold",
            )

            # === RECOMMENDATION BADGE ===
            rec_x = 12.5
            rec_y = 2.5

            # Determine badge color
            rec_upper = recommendation.upper()
            if rec_upper in ["BUY", "STRONG BUY", "INVEST", "FUND"]:
                badge_color = GREEN
            elif rec_upper in ["HOLD", "NEUTRAL", "MONITOR", "CONDITIONAL"]:
                badge_color = ORANGE
            else:
                badge_color = RED

            # Badge background
            badge = FancyBboxPatch(
                (rec_x - 1.8, rec_y - 0.8),
                3.6,
                2.2,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=badge_color,
                edgecolor="none",
            )
            ax.add_patch(badge)

            # Badge text
            ax.text(
                rec_x,
                rec_y + 0.45,
                "RECOMMENDATION",
                fontsize=10,
                color=WHITE,
                va="center",
                ha="center",
                alpha=0.9,
            )
            ax.text(
                rec_x,
                rec_y - 0.15,
                recommendation.upper(),
                fontsize=28,
                color=WHITE,
                va="center",
                ha="center",
                fontweight="bold",
            )

            # === KEY HIGHLIGHTS SECTION ===
            if highlights:
                hl_x = 6.5
                hl_y = 4.0

                ax.text(
                    hl_x,
                    hl_y,
                    "KEY HIGHLIGHTS",
                    fontsize=11,
                    color=DARK_BLUE,
                    va="center",
                    ha="left",
                    fontweight="bold",
                )

                # Gold underline
                ax.plot(
                    [hl_x, hl_x + 3],
                    [hl_y - 0.15, hl_y - 0.15],
                    color=GOLD,
                    linewidth=2,
                )

                for i, highlight in enumerate(highlights):
                    bullet_y = hl_y - 0.5 - (i * 0.5)
                    # Bullet point
                    bullet = Circle(
                        (hl_x + 0.1, bullet_y), 0.08, facecolor=GOLD, edgecolor="none"
                    )
                    ax.add_patch(bullet)
                    # Text - truncate to 40 chars max to prevent overflow into recommendation badge
                    display_text = (
                        highlight[:40] + "..." if len(highlight) > 40 else highlight
                    )
                    ax.text(
                        hl_x + 0.35,
                        bullet_y,
                        display_text,
                        fontsize=10,
                        color=DARK_BLUE,
                        va="center",
                        ha="left",
                    )

            # === FOOTER ===
            ax.plot([0.5, 15.5], [0.6, 0.6], color="#e0e0e0", linewidth=1)
            ax.text(
                8,
                0.3,
                "CONFIDENTIAL • FOR INVESTMENT PURPOSES ONLY • POWERED BY OMNIREXFLORA LABS",
                fontsize=9,
                color=MEDIUM_GRAY,
                va="center",
                ha="center",
                style="italic",
            )

            # Save with high DPI
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"omnirex_dashboard_{company_name.replace(' ', '_')}_{timestamp}.png"
            )
            filepath = os.path.join(get_output_dir(), filename)
            plt.savefig(
                filepath,
                dpi=150,  # High resolution
                bbox_inches="tight",
                facecolor=WHITE,
                edgecolor="none",
                pad_inches=0.2,
            )
            plt.close()

            logger.info(f"Professional infographic generated: {filepath}")
            return {
                "status": "success",
                "message": f"Professional infographic saved: {filepath}",
                "data": {
                    "filepath": filepath,
                    "metrics": {
                        "company": company_name,
                        "valuation": valuation,
                        "arr": arr,
                        "growth": growth_rate,
                        "market_size": market_size,
                        "risk_score": risk_score,
                        "recommendation": recommendation,
                    },
                },
            }
        except Exception as e:
            logger.error(f"Error creating infographic: {e}")
            return {
                "status": "error",
                "message": f"Error creating infographic: {str(e)}",
            }

    # =========================================================================
    # TOOL 3: FINANCIAL CHART
    # =========================================================================
    @registry.register_tool(
        name="generate_financial_chart",
        description="""
        Generate a professional revenue projection chart with bear/base/bull scenarios.

        **When to use:**
        Use this tool when you need to visualize financial projections for an investment
        analysis. The chart shows three scenarios (pessimistic, expected, optimistic) to
        help investors understand the range of possible outcomes.

        **What it produces:**
        A PNG image file showing:
        - Three projection lines (Bear: red, Base: blue, Bull: green)
        - Shaded confidence region between bear and bull cases
        - Value annotations on the base case line
        - Professional styling suitable for investor presentations

        **How to use:**
        1. Provide the company name for the chart title
        2. Set the current ARR (Annual Recurring Revenue) in millions
        3. Provide growth multipliers for each scenario (e.g., 1.5 means 50% growth)
        4. The number of years projected equals the number of rates provided

        **Example:**
        For a company with $1.2M ARR projecting 5 years:
        {
            "company_name": "Paystack",
            "current_arr_m": 1.2,
            "bear_rates": "1.1,1.1,1.1,1.1,1.1",
            "base_rates": "1.5,1.4,1.3,1.25,1.2",
            "bull_rates": "2.0,1.8,1.6,1.4,1.3"
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Name of the company for chart title.",
                },
                "current_arr_m": {
                    "type": "number",
                    "description": "Current ARR in millions (e.g., 1.2 means $1.2M).",
                },
                "bear_rates": {
                    "type": "string",
                    "description": "Comma-separated YoY multipliers for bear case.",
                },
                "base_rates": {
                    "type": "string",
                    "description": "Comma-separated YoY multipliers for base case.",
                },
                "bull_rates": {
                    "type": "string",
                    "description": "Comma-separated YoY multipliers for bull case.",
                },
            },
            "required": [
                "company_name",
                "current_arr_m",
                "bear_rates",
                "base_rates",
                "bull_rates",
            ],
            "additionalProperties": False,
        },
    )
    def generate_financial_chart(
        company_name: str,
        current_arr_m: float,
        bear_rates: Union[str, List[float]],
        base_rates: Union[str, List[float]],
        bull_rates: Union[str, List[float]],
    ) -> dict:
        """Generate professional revenue projection chart."""
        if not HAS_MATPLOTLIB:
            return {"status": "error", "message": "Matplotlib is not installed."}

        try:

            def parse_rates(rates_input: Union[str, list]) -> List[float]:
                if isinstance(rates_input, list):
                    return [float(x) for x in rates_input]
                s = str(rates_input).strip()
                if "," in s:
                    return [float(x.strip()) for x in s.split(",")]
                return [float(x.strip()) for x in s.split()]

            def project_arr(start: float, rates: List[float]) -> List[float]:
                arr = [start]
                for rate in rates:
                    arr.append(arr[-1] * rate)
                return arr

            company_name = ensure_string(company_name)
            current = float(current_arr_m)

            bear = parse_rates(bear_rates)
            base = parse_rates(base_rates)
            bull = parse_rates(bull_rates)

            start_year = datetime.now().year
            years = list(range(start_year, start_year + len(base) + 1))

            bear_arr = project_arr(current, bear)
            base_arr = project_arr(current, base)
            bull_arr = project_arr(current, bull)

            # Professional chart styling
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot all three scenarios
            ax.plot(
                years,
                bear_arr,
                "o-",
                color="#dc2626",
                linewidth=2,
                markersize=8,
                label="Bear Case",
            )
            ax.plot(
                years,
                base_arr,
                "s-",
                color="#1a365d",
                linewidth=3,
                markersize=10,
                label="Base Case",
            )
            ax.plot(
                years,
                bull_arr,
                "^-",
                color="#16a34a",
                linewidth=2,
                markersize=8,
                label="Bull Case",
            )

            # Shaded confidence region
            ax.fill_between(years, bear_arr, bull_arr, alpha=0.1, color="#1a365d")

            # Styling
            ax.set_title(
                f"{company_name} - Revenue Projection Analysis",
                fontsize=16,
                fontweight="bold",
                color="#1a365d",
            )
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("ARR ($ Millions)", fontsize=12)
            ax.legend(loc="upper left", fontsize=11)

            # Value annotations for base case
            for x, y in zip(years, base_arr):
                ax.annotate(
                    f"${y:.1f}M",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    color="#1a365d",
                )

            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"omnirex_chart_{company_name.replace(' ', '_')}_{timestamp}.png"
            filepath = os.path.join(get_output_dir(), filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return {
                "status": "success",
                "message": f"Chart generated: {filepath}",
                "data": {
                    "filepath": filepath,
                    "summary": {
                        f"year_{len(base)}_bear": f"${bear_arr[-1]:.1f}M",
                        f"year_{len(base)}_base": f"${base_arr[-1]:.1f}M",
                        f"year_{len(base)}_bull": f"${bull_arr[-1]:.1f}M",
                    },
                },
            }
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return {"status": "error", "message": f"Error plotting chart: {str(e)}"}

    # =========================================================================
    # TOOL 4: EVALUATION MEMO
    # =========================================================================
    @registry.register_tool(
        name="save_evaluation_memo",
        description="""
        Save the final investment evaluation memo for investment committee review.

        **When to use:**
        Use this as the final step to document your investment recommendation.
        This creates an audit-ready memo in markdown format.

        **What it produces:**
        A markdown file with:
        - Company name and evaluation date
        - Investment recommendation
        - Confidence score
        - Full analysis content

        **Example:**
        {
            "company_name": "Paystack",
            "recommendation": "FUND",
            "confidence": 85,
            "content": "## Summary\\nPaystack demonstrates..."
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company being evaluated.",
                },
                "recommendation": {
                    "type": "string",
                    "description": "FUND, CONDITIONAL, or PASS.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-100.",
                },
                "content": {
                    "type": "string",
                    "description": "Full memo content in markdown.",
                },
            },
            "required": ["company_name", "recommendation", "confidence", "content"],
            "additionalProperties": False,
        },
    )
    def save_evaluation_memo(
        company_name: str, recommendation: str, confidence: float, content: str
    ) -> dict:
        """Save investment evaluation memo."""
        content = ensure_string(content)
        company_name = ensure_string(company_name)

        memo = f"""# Investment Evaluation: {company_name}

**Date:** {datetime.now().strftime("%B %d, %Y")}
**Recommendation:** {recommendation.upper()}
**Confidence:** {confidence}%

---

{content}

---

*Generated by OmniRexFlora DeepAgent • Confidential*
"""

        filename = f"omnirex_memo_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(get_output_dir(), filename)

        with open(filepath, "w") as f:
            f.write(memo)

        return {
            "status": "success",
            "message": f"Memo saved: {filepath}",
            "data": {"filepath": filepath},
        }

    # =========================================================================
    # TOOL 5: COMPETITOR ANALYSIS
    # =========================================================================
    @registry.register_tool(
        name="analyze_competitor_landscape",
        description="""
        Generate a competitive landscape analysis with feature matrix.

        **When to use:**
        Use this to compare the target company against local and global competitors.
        This tool performs real-time research using Tavily to gather competitor data.

        **What it produces:**
        - Research findings from live search
        - Template feature matrix for comparison
        - Strategic differentiation insights

        **Example:**
        {
            "company_name": "Paystack",
            "sector": "Fintech/Payments",
            "competitors": ["Flutterwave", "Interswitch", "Stripe"],
            "key_features": ["API Quality", "Country Coverage", "Pricing", "Support"]
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Target company."},
                "sector": {"type": "string", "description": "Industry sector."},
                "competitors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 3-5 competitors.",
                },
                "key_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Features to compare.",
                },
            },
            "required": ["company_name", "sector", "competitors", "key_features"],
            "additionalProperties": False,
        },
    )
    def analyze_competitor_landscape(
        company_name: str, sector: str, competitors: List[str], key_features: List[str]
    ) -> dict:
        """Generate competitive landscape analysis."""
        findings = []
        for comp in competitors[:3]:
            q = f"{comp} vs {company_name} features {sector} Africa"
            results = _tavily_search(q, max_results=3)
            if results:
                findings.append(
                    f"### {comp} Analysis (Live Data):\n"
                    + "\n".join(
                        f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}..."
                        for r in results
                    )
                )
            else:
                findings.append(f"### {comp} Analysis:\n(No live data found)")

        research_text = "\n\n".join(findings)

        # Generate Template Table
        md_table = f"### Competitive Landscape: {sector}\n\n"
        md_table += f"| Feature | {company_name} | " + " | ".join(competitors) + " |\n"
        md_table += "|---|---|" + "---|" * len(competitors) + "\n"
        for feature in key_features:
            md_table += (
                f"| {feature} | TBD | "
                + " | ".join(["TBD"] * len(competitors))
                + " |\n"
            )

        return {
            "status": "success",
            "data": f"## RESEARCH FINDINGS\n\n{research_text}\n\n## TEMPLATE\n\n{md_table}",
            "message": "Use findings to fill the template table.",
        }

    # =========================================================================
    # TOOL 6: MACRO RISK ASSESSMENT
    # =========================================================================
    @registry.register_tool(
        name="assess_macro_risk",
        description="""
        Evaluate country-specific macro risks for African markets.

        **When to use:**
        Use this for any company operating in African markets to assess:
        - Currency/FX risk
        - Regulatory stability
        - Infrastructure reliability

        **What it produces:**
        - Real-time risk profile based on current news
        - Risk ratings (High/Medium/Low) per category
        - Context for investment decision

        **Example:**
        {
            "country": "Nigeria",
            "sector": "Fintech"
        }
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Target country (e.g., Kenya, Nigeria).",
                },
                "sector": {"type": "string", "description": "Industry sector."},
            },
            "required": ["country", "sector"],
            "additionalProperties": False,
        },
    )
    def assess_macro_risk(country: str, sector: str) -> dict:
        """Assess country-specific macro risks."""
        year = datetime.now().year
        q = f"{country} macroeconomic risk {sector} {year} currency regulatory infrastructure"
        results = _tavily_search(q, max_results=5)

        if results:
            context = "\n".join(
                f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}..."
                for r in results
            )
        else:
            context = "No live data available."

        # Simple keyword heuristics
        c = context.lower()
        fx = (
            "High"
            if any(
                x in c
                for x in [
                    "depreciation",
                    "volatility",
                    "crash",
                    "weakening",
                    "shortage",
                ]
            )
            else "Medium"
        )
        reg = (
            "High"
            if any(x in c for x in ["uncertainty", "ban", "crackdown", "policy shift"])
            else "Low"
        )
        infra = (
            "High"
            if any(
                x in c for x in ["outage", "blackout", "load shedding", "grid failure"]
            )
            else "Medium"
        )

        report = f"""## Macro Risk Assessment: {country} ({year})

### Live News Analysis
{context}

### Risk Ratings

| Category | Level | Notes |
|----------|-------|-------|
| FX/Currency | {fx} | Based on current news |
| Regulatory | {reg} | Policy stability analysis |
| Infrastructure | {infra} | Power/Internet reliability |

### Investment Implications
{"⚠️ **HIGH RISK ENVIRONMENT** - Proceed with caution" if fx == "High" or reg == "High" else "✅ **MODERATE RISK** - Standard due diligence recommended"}
"""

        return {
            "status": "success",
            "risk_profile": report,
            "risk_data": {
                "FX": {"level": fx, "context": "Live Search Analysis"},
                "Regulatory": {"level": reg, "context": "Live Search Analysis"},
                "Infrastructure": {"level": infra, "context": "Live Search Analysis"},
            },
        }

    return registry
