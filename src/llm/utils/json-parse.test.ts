import {
  parseCompletedToolCallJson,
  parseJsonWithRepair,
  parseStreamingJson,
  repairJson,
} from "@openclaw/ai/internal/runtime";
// JSON parse tests cover tolerant parsing of partial model JSON output.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("json-parse repairJson invalid \\u escapes", () => {
  it("repairs a \\u not followed by four hex digits so the result parses", () => {
    // JS string is: {"path":"C:\users"} — a model emitting an unescaped Windows path.
    const broken = '{"path":"C:\\users"}';
    expect(() => JSON.parse(repairJson(broken))).not.toThrow();
    expect(parseJsonWithRepair(broken)).toEqual({ path: "C:\\users" });
  });

  it("preserves valid \\uXXXX escapes", () => {
    expect(parseJsonWithRepair('{"e":"\\u0041"}')).toEqual({ e: "A" });
  });

  it.each([
    ['{"path":"C:\\bin\\app.exe"}', "C:\\bin\\app.exe"],
    ['{"path":"C:\\temp\\x"}', "C:\\temp\\x"],
    ['{"path":"C:\\new\\file"}', "C:\\new\\file"],
    ['{"path":"D:\\reports\\q"}', "D:\\reports\\q"],
    ['{"path":"C:\\users\\bob"}', "C:\\users\\bob"],
  ])("preserves unescaped Windows path control-letter segments: %s", (input, expected) => {
    expect(parseStreamingJson(input)).toEqual({ path: expected });
    expect(parseJsonWithRepair(input)).toEqual({ path: expected });
  });

  it("preserves legitimate JSON control escapes outside Windows paths", () => {
    expect(parseJsonWithRepair('{"message":"line\\nnext\\ttabbed"}')).toEqual({
      message: "line\nnext\ttabbed",
    });
  });

  it("recovers streaming tool-call arguments instead of dropping them to {}", () => {
    // LaTeX-style \u (\underline) is a valid string value the model may emit in args.
    const args = '{"cmd":"\\underline{x}"}';
    expect(parseStreamingJson(args)).toEqual({ cmd: "\\underline{x}" });
  });

  it.each(["null", "[]", '"text"', "1", "true"])(
    "returns an empty object for non-object streaming JSON: %s",
    (input) => {
      expect(parseStreamingJson(input)).toEqual({});
    },
  );
});

// ──────────────────────────────────────────────────────────────────────────────
// Regression: Hermes contract 20260709-buffer-mcp-tool-call-json-payload
// Root-cause falsification: the provider transport already accumulates deltas in
// partialArgs/partialJson and only executes tools at toolcall_end, so frame-
// complete buffering is satisfied by existing code.  These tests guard the
// specific acceptance criteria that were previously untested.
// ──────────────────────────────────────────────────────────────────────────────
describe("parseStreamingJson delta-accumulation contract", () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it("reconstructs a payload split at byte 15 exactly once on full accumulation", () => {
    // Spec acceptance: a tool-call JSON object split across at least two deltas
    // reconstructs and parses exactly once; partial deltas never throw.
    const full = '{"tool":"memory","args":{"q":"test"}}';
    const delta1 = full.slice(0, 15); // '{"tool":"memor'
    const delta2 = full.slice(15); // 'y","args":{"q":"test"}}'

    // Intermediate delta: partial-json recovers a partial value — no throw, no warn.
    const partialResult = parseStreamingJson(delta1);
    expect(partialResult).toMatchObject({ tool: expect.any(String) });
    expect(warnSpy).not.toHaveBeenCalled();

    // Final accumulated payload: fully parsed exactly once.
    const accumulated = delta1 + delta2;
    expect(accumulated).toBe(full);
    const finalResult = parseStreamingJson(accumulated);
    expect(finalResult).toEqual({ tool: "memory", args: { q: "test" } });
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("parses a multi-KB unified-memory-like nested payload correctly", () => {
    // Spec acceptance: unified-memory-like multi-KB nested payloads must parse.
    const longValue = "This is a memory entry about a Q2 planning meeting. ".repeat(40);
    const payload = JSON.stringify({
      key: "meeting_notes_2026_q2",
      value: longValue,
      category: "work",
      importance: 4,
      tags: ["meeting", "q2", "planning", "strategy", "visiting_media"],
      metadata: {
        source: "conversation",
        participants: ["alice", "bob", "charlie"],
        created: "2026-07-28T18:00:00Z",
        project: "visiting_media",
        nested: {
          flags: { compressed: false, indexed: true },
          ids: [1, 2, 3, 42],
        },
      },
    });
    expect(payload.length).toBeGreaterThan(2_000);

    const result = parseStreamingJson(payload);
    expect(result.key).toBe("meeting_notes_2026_q2");
    expect(result.category).toBe("work");
    expect(result.importance).toBe(4);
    expect(Array.isArray(result.tags)).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("parseCompletedToolCallJson: returns {} and warns exactly once for an unrecoverable object payload", () => {
    // Spec acceptance: malformed final data fails once with a clear error.
    // parseCompletedToolCallJson is called at toolcall_end / content_block_stop
    // (frame-complete) when all strategies produced nothing from a non-trivial payload.
    // partial-json never throws for object-like inputs: it returns {} silently.
    // parseCompletedToolCallJson detects this case and logs once.
    const corrupt = '{"key": GARBAGE_VALUE_NO_PARSE}';

    const result = parseCompletedToolCallJson(corrupt, "unified_memory_remember");
    expect(result).toEqual({});
    // Warn fires exactly once, not silently swallowed.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const warnMsg = warnSpy.mock.calls[0][0] as string;
    expect(warnMsg).toContain("[json-parse]");
    expect(warnMsg).toContain("unrecoverable");
    expect(warnMsg).toContain("unified_memory_remember");
  });

  it("parseCompletedToolCallJson: does not warn for a valid empty-object payload {}", () => {
    // A tool that genuinely takes no arguments emits {}; no warn should fire.
    const result = parseCompletedToolCallJson("{}");
    expect(result).toEqual({});
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("parseCompletedToolCallJson: does not warn when keys are successfully recovered", () => {
    const result = parseCompletedToolCallJson('{"q":"test"}');
    expect(result).toEqual({ q: "test" });
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("does not warn for empty or whitespace-only inputs", () => {
    expect(parseStreamingJson("")).toEqual({});
    expect(parseStreamingJson("   ")).toEqual({});
    expect(parseStreamingJson(undefined)).toEqual({});
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("does not warn for non-object JSON primitives (not object-like payloads)", () => {
    // These are non-object inputs that fall through asStreamingJsonRecord to {}.
    // They should not trigger the unrecoverable-object warning.
    expect(parseStreamingJson("null")).toEqual({});
    expect(parseStreamingJson("[1,2,3]")).toEqual({});
    expect(parseStreamingJson('"text"')).toEqual({});
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
