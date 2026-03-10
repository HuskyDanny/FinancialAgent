/**
 * Unit tests for messageParser utility.
 *
 * Tests parseBackendMessage conversion and replayDeepEvents dispatch logic.
 */

import { describe, it, expect, vi } from "vitest";
import { parseBackendMessage, replayDeepEvents } from "../messageParser";
import type { ChatMessage, DeepStreamEvent } from "../../types/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeBackendMsg(overrides: Record<string, unknown> = {}) {
  return {
    role: "assistant",
    content: "hello",
    timestamp: "2026-03-10T00:00:00Z",
    ...overrides,
  };
}

function makeChatMsg(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    role: "assistant",
    content: "hello",
    timestamp: "2026-03-10T00:00:00Z",
    ...overrides,
  };
}

const sampleDeepEvent: DeepStreamEvent = {
  type: "deep_start",
  seq: 1,
  timestamp: "2026-03-10T00:00:00Z",
  symbol: "AAPL",
  subagent_names: ["technical"],
  enable_debate: false,
};

// ---------------------------------------------------------------------------
// parseBackendMessage
// ---------------------------------------------------------------------------

describe("parseBackendMessage", () => {
  it("maps role, content, and timestamp", () => {
    const result = parseBackendMessage(makeBackendMsg());

    expect(result.role).toBe("assistant");
    expect(result.content).toBe("hello");
    expect(result.timestamp).toBe("2026-03-10T00:00:00Z");
  });

  it("passes through tool_call", () => {
    const toolCall = { tool_name: "get_price", input: {}, output: "100" };
    const result = parseBackendMessage(makeBackendMsg({ tool_call: toolCall }));

    expect(result.tool_call).toEqual(toolCall);
  });

  it("extracts deep_events from metadata.raw_data", () => {
    const msg = makeBackendMsg({
      metadata: {
        raw_data: {
          deep_events: [sampleDeepEvent],
          other_key: "keep",
        },
      },
    });

    const result = parseBackendMessage(msg);

    expect(result.deep_events).toEqual([sampleDeepEvent]);
    expect(result.analysis_data).toEqual({ other_key: "keep" });
  });

  it("filters deep_events out of analysis_data", () => {
    const msg = makeBackendMsg({
      metadata: {
        raw_data: {
          deep_events: [sampleDeepEvent],
        },
      },
    });

    const result = parseBackendMessage(msg);

    expect(result.deep_events).toEqual([sampleDeepEvent]);
    // Only deep_events in raw_data → analysis_data should be undefined
    expect(result.analysis_data).toBeUndefined();
  });

  it("falls back to metadata as analysis_data when no raw_data", () => {
    const msg = makeBackendMsg({
      metadata: { summary: "bullish" },
    });

    const result = parseBackendMessage(msg);

    expect(result.analysis_data).toEqual({ summary: "bullish" });
    expect(result.deep_events).toBeUndefined();
  });

  it("returns undefined analysis_data when metadata is empty", () => {
    const result = parseBackendMessage(makeBackendMsg({ metadata: {} }));

    expect(result.analysis_data).toBeUndefined();
  });

  it("returns undefined analysis_data when no metadata", () => {
    const result = parseBackendMessage(makeBackendMsg());

    expect(result.analysis_data).toBeUndefined();
    expect(result.deep_events).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// replayDeepEvents
// ---------------------------------------------------------------------------

describe("replayDeepEvents", () => {
  it("dispatches actions for the most recent message with deep_events", () => {
    const dispatch = vi.fn();
    const mapper = (event: DeepStreamEvent) => ({ type: "ACTION", event });

    const messages: ChatMessage[] = [
      makeChatMsg({ deep_events: [sampleDeepEvent] }),
      makeChatMsg({ content: "later message" }),
    ];

    const result = replayDeepEvents(messages, mapper, dispatch);

    // Should find the FIRST message (iterating backward, the last with events)
    // Actually it iterates backward and picks the first one with deep_events
    // messages[1] has no events, messages[0] has events
    expect(result).toBe(true);
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith({
      type: "ACTION",
      event: sampleDeepEvent,
    });
  });

  it("picks the most recent message with deep_events (backward iteration)", () => {
    const dispatch = vi.fn();
    const oldEvent: DeepStreamEvent = {
      ...sampleDeepEvent,
      symbol: "OLD",
    };
    const newEvent: DeepStreamEvent = {
      ...sampleDeepEvent,
      symbol: "NEW",
    };
    const mapper = (event: DeepStreamEvent) => ({ type: "ACTION", event });

    const messages: ChatMessage[] = [
      makeChatMsg({ deep_events: [oldEvent] }),
      makeChatMsg({ content: "middle" }),
      makeChatMsg({ deep_events: [newEvent] }),
    ];

    replayDeepEvents(messages, mapper, dispatch);

    // Should dispatch newEvent (index 2), not oldEvent (index 0)
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith({
      type: "ACTION",
      event: newEvent,
    });
  });

  it("returns false when no messages have deep_events", () => {
    const dispatch = vi.fn();
    const mapper = () => null;

    const messages: ChatMessage[] = [makeChatMsg(), makeChatMsg()];

    const result = replayDeepEvents(messages, mapper, dispatch);

    expect(result).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("returns false when mapper returns null for all events", () => {
    const dispatch = vi.fn();
    const mapper = () => null;

    const messages: ChatMessage[] = [
      makeChatMsg({ deep_events: [sampleDeepEvent] }),
    ];

    const result = replayDeepEvents(messages, mapper, dispatch);

    expect(result).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("returns false for empty message array", () => {
    const dispatch = vi.fn();
    const mapper = () => ({ type: "ACTION" });

    const result = replayDeepEvents([], mapper, dispatch);

    expect(result).toBe(false);
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("dispatches multiple events from a single message", () => {
    const dispatch = vi.fn();
    const event2: DeepStreamEvent = {
      ...sampleDeepEvent,
      seq: 2,
      type: "deep_subagent_start",
      subagent_name: "technical",
      display_name: "Technical",
      icon: "chart",
      tool_names: ["get_price"],
    };
    const mapper = (event: DeepStreamEvent) => ({ type: "ACTION", event });

    const messages: ChatMessage[] = [
      makeChatMsg({ deep_events: [sampleDeepEvent, event2] }),
    ];

    const result = replayDeepEvents(messages, mapper, dispatch);

    expect(result).toBe(true);
    expect(dispatch).toHaveBeenCalledTimes(2);
  });

  it("preserves generic type safety (compile-time check)", () => {
    // This test verifies the generic <A> works — if types don't match,
    // TypeScript would fail to compile this file
    type MyAction = { kind: "custom"; payload: string };
    const dispatch = vi.fn<[MyAction], void>();
    const mapper = (): MyAction | null => ({
      kind: "custom",
      payload: "test",
    });

    const messages: ChatMessage[] = [
      makeChatMsg({ deep_events: [sampleDeepEvent] }),
    ];

    replayDeepEvents<MyAction>(messages, mapper, dispatch);

    expect(dispatch).toHaveBeenCalledWith({
      kind: "custom",
      payload: "test",
    });
  });
});
