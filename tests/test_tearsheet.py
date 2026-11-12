import pytest
from src.reporting.tearsheet_generator import ExecutiveTearsheetGenerator, generate_executive_briefing_pdf

def test_tearsheet_generator_instantiation():
    gen = ExecutiveTearsheetGenerator()
    assert gen.sim_engine is not None
    assert gen.fred_loader is not None
    assert gen.hyp_engine is not None

def test_generate_pdf_structure_and_header():
    """
    Verifies that the PDF generator outputs a valid PDF binary stream
    with the standard '%PDF-' magic bytes and a file size appropriate for a 2-page tearsheet.
    """
    pdf_bytes = generate_executive_briefing_pdf(
        spot_price=5000.0,
        expected_drift=0.08,
        annual_volatility=0.18,
        target_horizon_days=100,
        asset_label="S&P 500 Test"
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 20000  # Should be substantial (including embedded chart)
    assert pdf_bytes.startswith(b"%PDF-")

def test_custom_parameters_in_tearsheet():
    """
    Verifies execution across custom parameters without raising exceptions.
    """
    gen = ExecutiveTearsheetGenerator()
    pdf_bytes = gen.generate_tearsheet(
        spot_price=2500.0,
        expected_drift=-0.02,
        annual_volatility=0.25,
        target_horizon_days=63,
        asset_label="Gold Futures (GC=F)"
    )
    assert len(pdf_bytes) > 20000
    assert pdf_bytes.startswith(b"%PDF-")

def test_currency_formatting_and_math_sanitizer():
    """
    Tests formatting of negative currency values and sanitization of LaTeX math delimiters.
    Ensures:
    - Never inverted dollar signs (e.g. 1,102.66-$)
    - Comma separator for thousands, period for cents (e.g. -$3,368.55, never -$3.368.55)
    - No raw LaTeX delimiters ($(d=0)$, $(p<0.05)$)
    """
    from src.reporting.tearsheet_generator import format_loss_currency, format_gain_currency, sanitize_math_text

    # Negative currency formatting
    assert format_loss_currency(1102.66) == "-$1,102.66"
    assert format_loss_currency(-1091.79) == "-$1,091.79"
    assert format_loss_currency(1982.21) == "-$1,982.21"
    assert format_loss_currency(3368.55) == "-$3,368.55"
    assert not format_loss_currency(1102.66).endswith("-$")
    assert not format_loss_currency(3368.55).startswith("-$3.368")

    # Positive currency formatting
    assert format_gain_currency(5815.0) == "$5,815.00"
    assert format_gain_currency(70.85) == "$70.85"

    # LaTeX math sanitizer
    assert sanitize_math_text("Levels ( $(d=0)$") == "Levels (d = 0)"
    assert sanitize_math_text("STATISTICALLY SIGNIFICANT ( $(p<0.05)$") == "STATISTICALLY SIGNIFICANT (p < 0.05)"
    assert sanitize_math_text("$d=1$") == "d = 1"
    assert "$" not in sanitize_math_text("$(d=0)$")


