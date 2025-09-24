/**
 * useAnalysis Hook
 *
 * This hook manages the logic for making analysis requests to the backend.
 * It uses react-query's useMutation to handle the asynchronous nature of
 * API calls, including loading, success, and error states.
 */

import { useMutation } from '@tanstack/react-query'
import { analysisService } from '../../services/analysis'
import type { FibonacciAnalysisResponse, MacroSentimentResponse, StockFundamentalsResponse } from '../../services/analysis'

// Formatting functions
function formatFibonacciResponse(result: FibonacciAnalysisResponse): string {
    const keyLevels = result.fibonacci_levels.filter(level => level.is_key_level)

    // Format Big 3 trends with Fibonacci levels
    let bigThreeSection = '';
    if (result.raw_data?.top_trends && result.raw_data.top_trends.length > 0) {
        bigThreeSection = `
### 🎯 Big 3 Trends with Fibonacci Levels:

${result.raw_data.top_trends.slice(0, 3).map((trend: any, index: number) => {
    const trendType = trend.type.includes('Uptrend') ? '📈' : '📉';
    const magnitude = (trend.magnitude || 0).toFixed(1);
    const isMainTrend = index === 0;

    // Format Fibonacci levels for this trend
    const fibLevels = trend.fibonacci_levels || [];

    // Debug: log what we're receiving
    console.log(`Trend ${index + 1}:`, trend);
    console.log(`Fibonacci levels:`, fibLevels);

    const keyFibLevels = fibLevels.filter((level: any) => level.is_key_level);

    let fibSection = '';
    if (keyFibLevels.length > 0) {
        fibSection = keyFibLevels.map((level: any) =>
            `   • **${level.percentage}** - $${level.price.toFixed(2)}`
        ).join('\n');
    } else {
        // Debug: show all levels if no key levels found
        if (fibLevels.length > 0) {
            fibSection = fibLevels.map((level: any) =>
                `   • **${level.percentage}** - $${level.price.toFixed(2)}${level.is_key_level ? ' 🌟' : ''}`
            ).join('\n');
        } else {
            fibSection = '   • No levels calculated at all - Debug: ' + JSON.stringify(Object.keys(trend));
        }
    }

    // Add pressure zone info for the biggest trend only
    let pressureInfo = '';
    if (isMainTrend && result.pressure_zone) {
        pressureInfo = `\n   • **Golden Zone:** $${result.pressure_zone.lower_bound.toFixed(2)} - $${result.pressure_zone.upper_bound.toFixed(2)}`;
    }

    return `**${index + 1}. ${trendType} ${trend.type.toUpperCase()}** (${trend.period})
   • **Magnitude:** $${magnitude} move
   • **Range:** $${trend.low?.toFixed(2)} → $${trend.high?.toFixed(2)}
   • **Fibonacci Levels:**
${fibSection}${pressureInfo}`;
}).join('\n\n')}
`;
    }

    return `## 📊 Fibonacci Analysis - ${result.symbol}

**Analysis Period:** ${result.start_date} to ${result.end_date} (${result.timeframe === '1h' ? 'Hourly' : result.timeframe === '1d' ? 'Daily' : result.timeframe === '1w' ? 'Weekly' : result.timeframe === '1M' ? 'Monthly' : result.timeframe} timeframe)
**Current Price:** $${result.current_price.toFixed(2)}
**Trend Direction:** ${result.market_structure.trend_direction}
**Confidence Score:** ${(result.confidence_score * 100).toFixed(1)}%
${bigThreeSection}
`
}

function formatMacroResponse(result: MacroSentimentResponse): string {
    const topSectors = Object.entries(result.sector_performance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3)

    const bottomSectors = Object.entries(result.sector_performance)
        .sort(([, a], [, b]) => a - b)
        .slice(0, 3)

    return `## Macro Market Sentiment Analysis

**Overall Sentiment:** ${result.market_sentiment.toUpperCase()}
**VIX Level:** ${result.vix_level.toFixed(2)} (${result.vix_interpretation})
**Fear/Greed Score:** ${result.fear_greed_score}/100

### Top Performing Sectors:
${topSectors.map(([sector, perf]) => `• **${sector}**: ${perf > 0 ? '+' : ''}${perf.toFixed(2)}%`).join('\n')}

### Underperforming Sectors:
${bottomSectors.map(([sector, perf]) => `• **${sector}**: ${perf.toFixed(2)}%`).join('\n')}

### Market Outlook:
${result.market_outlook}

### Key Factors:
${result.key_factors.map(factor => `• ${factor}`).join('\n')}
`
}

function formatFundamentalsResponse(result: StockFundamentalsResponse): string {
    const analysisDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    return `## Fundamental Analysis - ${result.symbol}
*Analysis Date: ${analysisDate}*

**Company:** ${result.company_name}
**Current Price:** $${result.current_price.toFixed(2)} (${result.price_change >= 0 ? '+' : ''}${result.price_change_percent.toFixed(2)}%)

### Valuation Metrics:
${result.pe_ratio ? `• **P/E Ratio:** ${result.pe_ratio.toFixed(2)}` : ''}
${result.pb_ratio ? `• **P/B Ratio:** ${result.pb_ratio.toFixed(2)}` : ''}
${result.dividend_yield ? `• **Dividend Yield:** ${result.dividend_yield.toFixed(2)}%` : ''}

### Financial Health:
• **Market Cap:** $${(result.market_cap / 1e9).toFixed(2)}B
• **Volume:** ${result.volume.toLocaleString()} (Avg: ${result.avg_volume.toLocaleString()})
${result.beta ? `• **Beta:** ${result.beta.toFixed(2)} (volatility vs market)` : ''}

### Price Range:
• **52-Week High:** $${result.fifty_two_week_high.toFixed(2)}
• **52-Week Low:** $${result.fifty_two_week_low.toFixed(2)}
`
}


export const useAnalysis = (
    currentSymbol: string | null,
    selectedDateRange: { start: string; end: string },
    setMessages: (updater: (prevMessages: any[]) => any[]) => void,
    setSelectedDateRange: (range: { start: string; end: string }) => void,
    selectedInterval?: string
) => {
    return useMutation({
        mutationKey: ['analysis', currentSymbol, selectedInterval, selectedDateRange.start, selectedDateRange.end],
        mutationFn: async (userMessage: string) => {
            // DEBUG: Log the user message to see what we're parsing
            console.log('🔍 MUTATION DEBUG - User message:', userMessage);

            const intent = analysisService.parseAnalysisIntent(userMessage)
            const analysisSymbol = intent.symbol || currentSymbol

            switch (intent.type) {
                case 'fibonacci':
                    if (!analysisSymbol) throw new Error('Please select a stock symbol first.');

                    // Parse timeframe and dates from the user message to avoid closure issues
                    let timeframe = "1d";
                    let startDate = intent.start_date;
                    let endDate = intent.end_date;

                    console.log('🔍 MUTATION DEBUG - Intent object:', intent);
                    console.log('🔍 MUTATION DEBUG - Initial dates from intent:', startDate, 'to', endDate);

                    // Extract timeframe from user message patterns
                    if (userMessage.includes('1h analysis') || userMessage.includes('Hourly analysis')) {
                        timeframe = "1h";
                    } else if (userMessage.includes('1wk analysis') || userMessage.includes('Weekly analysis')) {
                        timeframe = "1w";
                    } else if (userMessage.includes('1mo analysis') || userMessage.includes('Monthly analysis')) {
                        timeframe = "1M";
                    } else if (userMessage.includes('Daily analysis')) {
                        timeframe = "1d";
                    }

                    // DEBUG: Log parsed values
                    console.log('🔍 MUTATION DEBUG - Parsed timeframe:', timeframe);
                    console.log('🔍 MUTATION DEBUG - Parsed dates:', startDate, 'to', endDate);

                    // Extract dates from user message if present (flexible pattern)
                    // Handles formats like: "2025-08-24 to 2025-09-23" anywhere in the message
                    const dateMatch = userMessage.match(/(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})/);
                    if (dateMatch) {
                        startDate = dateMatch[1];
                        endDate = dateMatch[2];
                        console.log('🔍 MUTATION DEBUG - Regex matched dates:', startDate, 'to', endDate);
                    } else {
                        console.log('🔍 MUTATION DEBUG - No date regex match for message:', userMessage);
                        // Try a more flexible approach
                        const flexibleMatch = userMessage.match(/(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})/);
                        if (flexibleMatch && flexibleMatch[1] !== flexibleMatch[2]) {
                            startDate = flexibleMatch[1];
                            endDate = flexibleMatch[2];
                            console.log('🔍 MUTATION DEBUG - Flexible regex matched dates:', startDate, 'to', endDate);
                        }
                    }

                    // If no dates found, calculate based on timeframe
                    if (!startDate || !endDate) {
                        const today = new Date();
                        let periodsBack;

                        switch (timeframe) {
                            case '1h':
                                periodsBack = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000); // 1 month
                                break;
                            case '1d':
                                periodsBack = new Date(today.getTime() - 6 * 30 * 24 * 60 * 60 * 1000); // 6 months
                                break;
                            case '1w':
                                periodsBack = new Date(today.getTime() - 365 * 24 * 60 * 60 * 1000); // 1 year
                                break;
                            case '1M':
                                periodsBack = new Date(today.getTime() - 2 * 365 * 24 * 60 * 60 * 1000); // 2 years
                                break;
                            default:
                                periodsBack = new Date(today.getTime() - 6 * 30 * 24 * 60 * 60 * 1000); // 6 months
                        }

                        startDate = periodsBack.toISOString().split('T')[0];
                        endDate = today.toISOString().split('T')[0];
                    }

                    const fibResult = await analysisService.fibonacciAnalysis({
                        symbol: analysisSymbol,
                        start_date: startDate,
                        end_date: endDate,
                        timeframe: timeframe
                    });
                    return { type: 'fibonacci', content: formatFibonacciResponse(fibResult), analysis_data: fibResult };

                case 'macro':
                    const macroResult = await analysisService.macroSentimentAnalysis({});
                    return { type: 'macro', content: formatMacroResponse(macroResult), analysis_data: macroResult };

                case 'fundamentals':
                    if (!analysisSymbol) throw new Error('Please select a stock symbol first.');
                    const fundResult = await analysisService.stockFundamentals({ symbol: analysisSymbol });
                    return { type: 'fundamentals', content: formatFundamentalsResponse(fundResult), analysis_data: fundResult };

                default:
                    throw new Error("I can help with Fibonacci, Macro, and Fundamentals analysis.");
            }
        },
        onMutate: (newMessage) => {
            setMessages(prev => [...prev, { role: 'user', content: newMessage, timestamp: new Date().toISOString() }]);
        },
        onSuccess: (response) => {
            setMessages(prev => [...prev, { role: 'assistant', content: response.content, timestamp: new Date().toISOString(), analysis_data: response.analysis_data }]);
        },
        onError: (error: any) => {
            const errorContent = error?.response?.data?.detail || error.message || 'Unknown error';
            setMessages(prev => [...prev, { role: 'assistant', content: `❌ **Error**: ${errorContent}`, timestamp: new Date().toISOString() }]);
        },
    });
};