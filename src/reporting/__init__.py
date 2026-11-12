"""
QuantMacro AI Reporting and Document Generation Package.
Provides institutional PDF tearsheet and quantitative briefing export engines.
"""
from src.reporting.tearsheet_generator import ExecutiveTearsheetGenerator, generate_executive_briefing_pdf

__all__ = ["ExecutiveTearsheetGenerator", "generate_executive_briefing_pdf"]

