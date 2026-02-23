"""Tests for debater sub-agent with independent tools."""

from unittest.mock import MagicMock

from src.agent.subagents.debater import (
    TERMINATION_SIGNAL,
    create_debater_subagent,
)


class TestDebaterSubagent:
    """Test debater uses only independent tools."""

    def test_debater_has_independent_tools_only(self) -> None:
        """Debater must NOT use Alpha Vantage tools."""
        mock_model = MagicMock()
        mock_context = MagicMock()
        mock_context.to_context_header.return_value = "test context"

        subagent = create_debater_subagent(
            model=mock_model,
            context=mock_context,
            exa_api_key="test-key",
        )

        tool_names = subagent.get_tool_names()
        assert "fetch_yfinance_news" in tool_names
        assert "search_web_exa" in tool_names
        # Must NOT have Alpha Vantage tools
        assert "get_company_overview" not in tool_names
        assert "get_news_sentiment" not in tool_names
        assert "get_financial_statements" not in tool_names

    def test_debater_without_exa_key(self) -> None:
        """Debater works with only yfinance when no exa key provided."""
        mock_model = MagicMock()
        mock_context = MagicMock()
        mock_context.to_context_header.return_value = "test context"

        subagent = create_debater_subagent(
            model=mock_model,
            context=mock_context,
            exa_api_key="",
        )

        tool_names = subagent.get_tool_names()
        assert "fetch_yfinance_news" in tool_names
        assert "search_web_exa" not in tool_names

    def test_termination_signal_unchanged(self) -> None:
        assert TERMINATION_SIGNAL == "NO FURTHER CONCERNS"
