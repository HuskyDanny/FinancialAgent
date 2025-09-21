/**
 * Symbol Search Autocomplete Component
 *
 * Features:
 * - Real-time symbol search as you type
 * - Debounced API calls for performance
 * - Company name to symbol mapping (e.g., "Apple" → "AAPL")
 * - Keyboard navigation support
 * - Clean, accessible UI
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Building2, TrendingUp } from 'lucide-react'
import { marketService, SymbolSearchResult } from '../services/market'

interface SymbolSearchProps {
  onSymbolSelect: (symbol: string, name: string) => void
  placeholder?: string
  className?: string
  autoFocus?: boolean
}

export const SymbolSearch: React.FC<SymbolSearchProps> = ({
  onSymbolSelect,
  placeholder = "Search stocks (e.g., Apple, AAPL, Microsoft...)",
  className = '',
  autoFocus = false
}) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SymbolSearchResult[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)

  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const debounceTimeoutRef = useRef<NodeJS.Timeout>()

  // Debounced search function
  const debouncedSearch = useCallback(
    marketService.createDebouncedSearch(300),
    []
  )

  // Search function
  const performSearch = useCallback(async (searchQuery: string) => {
    if (searchQuery.trim().length < 1) {
      setResults([])
      setIsOpen(false)
      return
    }

    setIsLoading(true)

    try {
      debouncedSearch(searchQuery, (searchResults) => {
        setResults(searchResults.results)
        setIsOpen(searchResults.results.length > 0)
        setSelectedIndex(-1)
        setIsLoading(false)
      })
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
      setIsOpen(false)
      setIsLoading(false)
    }
  }, [debouncedSearch])

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = e.target.value
    setQuery(newQuery)
    performSearch(newQuery)
  }

  // Handle result selection
  const handleResultSelect = (result: SymbolSearchResult) => {
    setQuery(`${result.symbol} - ${result.name}`)
    setIsOpen(false)
    setResults([])
    onSymbolSelect(result.symbol, result.name)
  }

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen || results.length === 0) return

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(prev =>
          prev < results.length - 1 ? prev + 1 : 0
        )
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(prev =>
          prev > 0 ? prev - 1 : results.length - 1
        )
        break
      case 'Enter':
        e.preventDefault()
        if (selectedIndex >= 0 && selectedIndex < results.length) {
          handleResultSelect(results[selectedIndex])
        } else if (results.length > 0) {
          const top = results[0]
          if (top.confidence && top.confidence >= 0.85) {
            handleResultSelect(top)
          }
        }
        break
      case 'Escape':
        setIsOpen(false)
        setSelectedIndex(-1)
        inputRef.current?.blur()
        break
    }
  }

  // Handle click outside to close
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        resultsRef.current &&
        !resultsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
        setSelectedIndex(-1)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Auto focus
  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus()
    }
  }, [autoFocus])

  // Clear timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current)
      }
    }
  }, [])

  return (
    <div className={`relative ${className}`}>
      {/* Search Input */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (results.length > 0) {
              setIsOpen(true)
            }
          }}
          placeholder={placeholder}
          className="block w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg text-sm placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
          autoComplete="off"
        />
        {isLoading && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
          </div>
        )}
      </div>

      {/* Search Results */}
      {isOpen && results.length > 0 && (
        <div
          ref={resultsRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-auto"
        >
          {results.map((result, index) => (
            <button
              key={`${result.symbol}-${index}`}
              onClick={() => handleResultSelect(result)}
              className={`w-full px-4 py-3 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none border-b border-gray-100 last:border-b-0 transition-colors ${
                index === selectedIndex ? 'bg-blue-50' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-gray-900">
                      {result.symbol}
                    </span>
                    {result.match_type && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700">
                        {result.match_type.replace('_', ' ')}{result.confidence !== undefined ? ` ${(result.confidence*100).toFixed(0)}%` : ''}
                      </span>
                    )}
                    {result.exchange && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                        {result.exchange}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 truncate mt-1">
                    {result.name}
                  </p>
                </div>
                <div className="flex items-center text-gray-400">
                  {result.type === 'EQUITY' ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <Building2 className="h-4 w-4" />
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* No Results */}
      {isOpen && !isLoading && query.length > 0 && results.length === 0 && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4">
          <div className="text-center text-gray-500">
            <Search className="h-8 w-8 mx-auto mb-2 text-gray-400" />
            <p className="text-sm">No stocks found for "{query}"</p>
            <p className="text-xs text-gray-400 mt-1">
              Try searching by company name or stock symbol
            </p>
          </div>
        </div>
      )}

      {/* Search Tips */}
      {query.length === 0 && (
        <div className="mt-2 text-xs text-gray-500">
          <p>💡 Try searching: "Apple", "AAPL", "Microsoft", "Tesla", etc.</p>
        </div>
      )}
    </div>
  )
}