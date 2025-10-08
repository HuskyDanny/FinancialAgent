/**
 * Formatting functions for analysis responses.
 * Converts API responses into markdown for chat display.
 */

import type {
  FibonacciAnalysisResponse,
  MacroSentimentResponse,
  StockFundamentalsResponse,
  StochasticAnalysisResponse,
} from "../../services/analysis";

export function formatFibonacciResponse(
  result: FibonacciAnalysisResponse,
): string {
  // Get trend emoji
  const trendEmoji = result.market_structure.trend_direction
    .toLowerCase()
    .includes("up")
    ? "📈"
    : result.market_structure.trend_direction.toLowerCase().includes("down")
      ? "📉"
      : "➡️";

  // Build trends section
  let trendsSection = "";
  if (result.raw_data?.top_trends && result.raw_data.top_trends.length > 0) {
    const trends = result.raw_data.top_trends.slice(0, 3); // Top 3 trends

    trendsSection = `

### 📊 Key Trends Identified

${trends
  .map((trend: any, index: number) => {
    const trendEmoji = trend.type.includes("Uptrend") ? "📈" : "📉";
    const isMainTrend = index === 0;

    // Build Fibonacci levels collapsible section
    const fibLevels = trend.fibonacci_levels || [];
    const fibSection =
      fibLevels.length > 0
        ? `

<details>
<summary><strong>📐 Fibonacci Levels</strong> (click to expand)</summary>

| Level | Price |
|-------|-------|
${fibLevels.map((level: any) => `| ${level.percentage} | $${level.price.toFixed(2)} |`).join("\n")}

</details>`
        : "";

    return `**${index + 1}. ${trendEmoji} ${trend.type.toUpperCase()}**
• Period: ${trend.period}
• Magnitude: $${(trend.magnitude || 0).toFixed(2)} move
• Range: $${trend.low?.toFixed(2)} → $${trend.high?.toFixed(2)}${
      isMainTrend && result.pressure_zone
        ? `\n• Golden Zone: $${result.pressure_zone.lower_bound.toFixed(2)} - $${result.pressure_zone.upper_bound.toFixed(2)}`
        : ""
    }${fibSection}`;
  })
  .join("\n\n")}
`;
  }

  return `## 📊 Fibonacci Analysis - ${result.symbol}

### ${trendEmoji} Bottom Line

• **Trend**: ${result.market_structure.trend_direction.toUpperCase()}
• **Current Price**: $${result.current_price.toFixed(2)}
• **Confidence**: ${(result.confidence_score * 100).toFixed(1)}%
• **Period**: ${result.start_date || "Dynamic"} to ${result.end_date || "Current"}
${trendsSection}
`;
}

export function formatMacroResponse(result: MacroSentimentResponse): string {
  const sentimentEmoji =
    result.market_sentiment.toLowerCase() === "bullish"
      ? "📈"
      : result.market_sentiment.toLowerCase() === "bearish"
        ? "📉"
        : "➡️";

  const allSectors = Object.entries(result.sector_performance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6);

  return `## 🌍 Macro Market Sentiment

### ${sentimentEmoji} Bottom Line

| Metric | Value |
|--------|-------|
| Market Sentiment | ${result.market_sentiment.toUpperCase()} |
| VIX Level | ${result.vix_level.toFixed(2)} (${result.vix_interpretation}) |
| Fear/Greed Score | ${result.fear_greed_score}/100 |

### 📝 Market Outlook

${result.market_outlook}

### 📊 Sector Performance

| Sector | Performance |
|--------|-------------|
${allSectors.map(([sector, perf]) => `| ${sector} | ${perf > 0 ? "+" : ""}${perf.toFixed(2)}% |`).join("\n")}

### 🔑 Key Factors
${result.key_factors.map((factor) => `• ${factor}`).join("\n")}
`;
}

export function formatFundamentalsResponse(
  result: StockFundamentalsResponse,
): string {
  const priceChangeEmoji = result.price_change >= 0 ? "📈" : "📉";
  const analysisDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return `## 💼 Fundamentals - ${result.symbol}
*${result.company_name} • ${analysisDate}*

### ${priceChangeEmoji} Bottom Line

| Metric | Value |
|--------|-------|
| Current Price | $${result.current_price.toFixed(2)} (${result.price_change >= 0 ? "+" : ""}${result.price_change_percent.toFixed(2)}%) |
| Market Cap | $${(result.market_cap / 1e9).toFixed(2)}B |
${result.pe_ratio ? `| P/E Ratio | ${result.pe_ratio.toFixed(2)} |` : ""}
${result.pb_ratio ? `| P/B Ratio | ${result.pb_ratio.toFixed(2)} |` : ""}

### 📊 Valuation & Trading

| Metric | Value |
|--------|-------|
${result.dividend_yield ? `| Dividend Yield | ${result.dividend_yield.toFixed(2)}% |` : ""}
${result.beta ? `| Beta | ${result.beta.toFixed(2)} |` : ""}
| Volume | ${result.volume.toLocaleString()} |
| Avg Volume | ${result.avg_volume.toLocaleString()} |
| 52-Week High | $${result.fifty_two_week_high.toFixed(2)} |
| 52-Week Low | $${result.fifty_two_week_low.toFixed(2)} |
`;
}

export function formatStochasticResponse(
  result: StochasticAnalysisResponse,
): string {
  // Signal interpretation with dynamic color intensity
  let signalEmoji = "";
  let signalMeaning = "";
  let signalColor = "";

  const kValue = result.current_k;

  if (result.current_signal === "overbought") {
    // Red: darker as %K approaches 100
    const intensity = Math.min(((kValue - 80) / 20) * 100, 100); // 0-100%
    const red = Math.round(139 + (intensity / 100) * 116); // 139 (dark) → 255 (bright)
    signalColor = `rgb(${red}, 0, 0)`;
    signalEmoji = "🔴";
    signalMeaning = "OVERBOUGHT (Potential Sell Zone)";
  } else if (result.current_signal === "oversold") {
    // Green: darker as %K approaches 0
    const intensity = Math.min(((20 - kValue) / 20) * 100, 100); // 0-100%
    const green = Math.round(100 + (intensity / 100) * 155); // 100 (dark) → 255 (bright)
    signalColor = `rgb(0, ${green}, 0)`;
    signalEmoji = "🟢";
    signalMeaning = "OVERSOLD (Potential Buy Zone)";
  } else {
    // Yellow for neutral
    signalColor = "rgb(255, 215, 0)"; // Gold yellow
    signalEmoji = "🟡";
    signalMeaning = "NEUTRAL (No Clear Signal)";
  }

  const analysisDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const recentSignals = result.signal_changes.slice(-3);

  return `## 📊 Stochastic Oscillator - ${result.symbol}
*${analysisDate} • ${result.timeframe} timeframe*

### ${signalEmoji} Bottom Line

<table style="width: 100%; border-collapse: collapse; border: 1px solid #d1d5db; margin-bottom: 1rem;">
  <tbody>
    <tr style="border-bottom: 1px solid #d1d5db;">
      <td style="padding: 0.5rem 1rem; font-weight: 600; border-right: 1px solid #d1d5db;">Signal</td>
      <td style="padding: 0.5rem 1rem; background-color: ${signalColor}; color: white; font-weight: 700; border-right: 1px solid #d1d5db;">${signalMeaning}</td>
    </tr>
    <tr style="border-bottom: 1px solid #d1d5db;">
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">Current Price</td>
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">$${result.current_price.toFixed(2)}</td>
    </tr>
    <tr style="border-bottom: 1px solid #d1d5db;">
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">%K Line</td>
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">${result.current_k.toFixed(1)}%</td>
    </tr>
    <tr style="border-bottom: 1px solid #d1d5db;">
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">%D Line</td>
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">${result.current_d.toFixed(1)}%</td>
    </tr>
    <tr>
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">Parameters</td>
      <td style="padding: 0.5rem 1rem; border-right: 1px solid #d1d5db;">%K(${result.k_period}) %D(${result.d_period})</td>
    </tr>
  </tbody>
</table>

### 💡 Key Insights

${result.key_insights.map((insight) => `• ${insight}`).join("\n\n")}
${
  recentSignals.length > 0
    ? `

### 🔔 Recent Signals

${recentSignals
  .map((signal) => {
    const emoji =
      signal.type.toLowerCase() === "buy"
        ? "🟢"
        : signal.type.toLowerCase() === "sell"
          ? "🔴"
          : "🟡";
    return `${emoji} **${signal.type.toUpperCase()}**: ${signal.description}`;
  })
  .join("\n\n")}`
    : ""
}
`;
}
