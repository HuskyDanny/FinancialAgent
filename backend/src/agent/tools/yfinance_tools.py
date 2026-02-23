"""
LangChain tool for fetching financial news and stats from Yahoo Finance.

Independent data source for the debater agent — NOT the same
Alpha Vantage API used by research sub-agents. This ensures
genuine cross-verification in the debate.
"""

import json

import structlog
import yfinance as yf
from langchain_core.tools import tool

logger = structlog.get_logger()


def create_yfinance_tools() -> list:
    """Create Yahoo Finance tools for independent data verification."""

    @tool
    async def fetch_yfinance_news(symbol: str) -> str:
        """Fetch financial news and key stats from Yahoo Finance.

        Use this to cross-check investment thesis claims against an
        independent data source. Returns recent news headlines and
        key financial statistics from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")

        Returns:
            JSON string with news headlines and key financial stats
        """
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news or []
            info = ticker.info or {}

            news = [
                {
                    "title": n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link": n.get("link", ""),
                }
                for n in raw_news[:10]
            ]

            key_stats = {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": info.get("marketCap"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
            }

            return json.dumps(
                {"source": "yahoo_finance", "news": news, "key_stats": key_stats}
            )
        except Exception as e:
            logger.warning("yfinance fetch failed", symbol=symbol, error=str(e))
            return json.dumps({"source": "yahoo_finance", "error": str(e)})

    return [fetch_yfinance_news]
