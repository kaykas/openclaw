/**
 * Deterministic unit tests for the Sol session-cache probe (phase 1).
 *
 * Required invariants verified here:
 *  1. Timestamp / current time does NOT change the stable-prefix hash.
 *  2. A dynamic suffix changes the full-instructions hash but NOT the
 *     stable-prefix hash.
 *  3. The same Telegram session key always produces the same cache-key
 *     fingerprint (stable across calls).
 *  4. No raw prompt, session ID, session key, or cache-key content is
 *     present in any sol:turn-probe JSONL event.
 */
import crypto from "node:crypto";
import { describe, expect, it, afterEach } from "vitest";
import { SYSTEM_PROMPT_CACHE_BOUNDARY } from "@openclaw/ai/internal/shared";
import {
  computeInstructionHashes,
  fingerprintSessionKey,
  isSolTransport,
  recordActiveProcessState,
  resetActiveProcessTrackerForTest,
  buildSolContextProbeData,
  buildSolTurnTokenResult,
  SOL_TRANSPORT_API,
} from "./sol-cache-probe.js";
import { createCacheTrace } from "./cache-trace.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMemoryTrace() {
  const lines: string[] = [];
  const trace = createCacheTrace({
    cfg: { diagnostics: { cacheTrace: { enabled: true } } },
    env: {},
    sessionId: "test-session-id",
    sessionKey: "telegram:direct:12345678",
    provider: "openai",
    modelId: "gpt-5.6-sol",
    modelApi: SOL_TRANSPORT_API,
    writer: {
      filePath: "memory",
      write: (line) => lines.push(line),
      flush: async () => undefined,
    },
  });
  return { lines, trace };
}

function parseLine(line: string): Record<string, unknown> {
  return JSON.parse(line.trim()) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// 1. Timestamp does not change the stable-prefix hash
// ---------------------------------------------------------------------------
describe("computeInstructionHashes", () => {
  it("stable-prefix hash is unchanged when only the dynamic suffix (timestamp) changes", () => {
    const stableContent = "# SOUL.md\nYou are Mira.\n\n## Tools\n- exec\n- read";

    // Two prompts that differ only in the dynamic suffix (simulate time changing).
    const promptA = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}Current time: 2026-08-02T10:00:00Z`;
    const promptB = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}Current time: 2026-08-02T11:00:00Z`;

    const hashesA = computeInstructionHashes(promptA);
    const hashesB = computeInstructionHashes(promptB);

    // Stable-prefix hash must be identical despite the time change.
    expect(hashesA.stablePrefixHash).toBe(hashesB.stablePrefixHash);

    // Full-instructions hash must differ because the suffix changed.
    expect(hashesA.fullInstructionsHash).not.toBe(hashesB.fullInstructionsHash);
  });

  it("stable-prefix hash is unchanged when activeProcessSessions count changes in suffix", () => {
    const stableContent = "# Core context\nDo not expose secrets.";

    const promptA = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}Active processes: 0`;
    const promptB = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}Active processes: 3`;

    const hashesA = computeInstructionHashes(promptA);
    const hashesB = computeInstructionHashes(promptB);

    expect(hashesA.stablePrefixHash).toBe(hashesB.stablePrefixHash);
    expect(hashesA.fullInstructionsHash).not.toBe(hashesB.fullInstructionsHash);
  });

  // 2. Dynamic suffix changes full hash but not stable-prefix hash
  it("dynamic suffix change updates full-instructions hash; stable-prefix hash is invariant", () => {
    const stableContent = "# System context v42";

    const promptBase = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}turn=1`;
    const promptNext = `${stableContent}${SYSTEM_PROMPT_CACHE_BOUNDARY}turn=2`;

    const baseHashes = computeInstructionHashes(promptBase);
    const nextHashes = computeInstructionHashes(promptNext);

    expect(baseHashes.stablePrefixHash).toBe(nextHashes.stablePrefixHash);
    expect(baseHashes.fullInstructionsHash).not.toBe(nextHashes.fullInstructionsHash);
  });

  it("changing stable content changes both hashes", () => {
    const promptA = `ContentA${SYSTEM_PROMPT_CACHE_BOUNDARY}suffix`;
    const promptB = `ContentB${SYSTEM_PROMPT_CACHE_BOUNDARY}suffix`;

    const hashesA = computeInstructionHashes(promptA);
    const hashesB = computeInstructionHashes(promptB);

    expect(hashesA.stablePrefixHash).not.toBe(hashesB.stablePrefixHash);
    expect(hashesA.fullInstructionsHash).not.toBe(hashesB.fullInstructionsHash);
  });

  it("when no cache boundary is present the whole prompt is treated as stable", () => {
    const prompt = "Static prompt with no boundary marker.";

    const { stablePrefixHash, fullInstructionsHash } = computeInstructionHashes(prompt);

    // Both hashes must equal SHA-256 of the raw prompt.
    const expected = crypto.createHash("sha256").update(prompt, "utf8").digest("hex");
    expect(stablePrefixHash).toBe(expected);
    expect(fullInstructionsHash).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// 3. Same Telegram session produces a stable cache-key fingerprint
// ---------------------------------------------------------------------------
describe("fingerprintSessionKey", () => {
  it("same Telegram session key always produces the same fingerprint", () => {
    const sessionKey = "telegram:direct:8456174966";

    const fp1 = fingerprintSessionKey(sessionKey);
    const fp2 = fingerprintSessionKey(sessionKey);

    expect(fp1).toBe(fp2);
    expect(fp1).toHaveLength(16);
  });

  it("different session keys produce different fingerprints", () => {
    const fp1 = fingerprintSessionKey("telegram:direct:111");
    const fp2 = fingerprintSessionKey("telegram:direct:222");

    expect(fp1).not.toBe(fp2);
  });

  it("fingerprint does not contain the raw session key", () => {
    const rawKey = "telegram:direct:8456174966";
    const fp = fingerprintSessionKey(rawKey);

    expect(fp).not.toContain(rawKey);
    expect(fp).not.toContain("telegram");
    expect(fp).not.toContain("direct");
    expect(fp).not.toContain("8456174966");
  });

  it("fingerprint is a hex string of expected length", () => {
    const fp = fingerprintSessionKey("any-key");

    // 16 hex chars = 64 bits (SHA-256 prefix slice)
    expect(fp).toMatch(/^[0-9a-f]{16}$/);
  });
});

// ---------------------------------------------------------------------------
// 4. No raw content in the sol:turn-probe JSONL event
// ---------------------------------------------------------------------------
describe("sol:turn-probe JSONL event redaction", () => {
  afterEach(() => {
    resetActiveProcessTrackerForTest();
  });

  it("sol:turn-probe event contains no raw prompt, session key, or cache-key content", () => {
    const { lines, trace } = makeMemoryTrace();
    if (!trace) throw new Error("trace must be created");

    const rawSessionKey = "telegram:direct:8456174966";
    const secretContent = "MYSECRETPROMPT_DO_NOT_LOG";
    const stablePrompt = secretContent;
    const fullPrompt = `${stablePrompt}${SYSTEM_PROMPT_CACHE_BOUNDARY}ts=2026-08-02T10:00:00Z`;

    // Simulate a Sol stream call to capture context data.
    trace.wrapStreamFn(
      ((_model: unknown, _context: unknown, _options: unknown) => {
        return { text: "ignored" } as never;
      }) as never,
    )(
      { id: "gpt-5.6-sol", provider: "openai", api: SOL_TRANSPORT_API } as never,
      { systemPrompt: fullPrompt } as never,
      {},
    );

    // Record the post-turn result.
    trace.recordSolTurnProbeIfActive({
      usage: { input: 10000, cacheRead: 3000, cacheWrite: undefined },
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: rawSessionKey,
      sessionId: "test-session-id",
      fallbackReason: null,
    });

    // Find the sol:turn-probe event.
    const probeLines = lines.filter((l) => {
      try {
        const ev = parseLine(l);
        return ev.stage === "sol:turn-probe";
      } catch {
        return false;
      }
    });

    expect(probeLines.length).toBe(1);
    const probeEvent = probeLines[0]!;
    const serialized = probeEvent;

    // No raw prompt content.
    expect(serialized).not.toContain(secretContent);
    // No raw session key.
    expect(serialized).not.toContain(rawSessionKey);
    expect(serialized).not.toContain("8456174966");
    // No raw cache boundary marker text (not needed in diagnostic output).
    // sessionKey field must be absent from the event body.
    const ev = parseLine(probeEvent);
    expect(ev).not.toHaveProperty("sessionKey");
    expect(ev).not.toHaveProperty("workspaceDir");
    // System/prompt/messages fields must be absent.
    expect(ev).not.toHaveProperty("system");
    expect(ev).not.toHaveProperty("prompt");
    expect(ev).not.toHaveProperty("messages");
  });

  it("sol:turn-probe event contains expected hashes, fingerprint, and numeric fields", () => {
    const { lines, trace } = makeMemoryTrace();
    if (!trace) throw new Error("trace must be created");

    const stable = "## Context";
    const dynamic = "turn=1";
    const fullPrompt = `${stable}${SYSTEM_PROMPT_CACHE_BOUNDARY}${dynamic}`;

    trace.wrapStreamFn(
      ((_model: unknown, _ctx: unknown, _opts: unknown) => ({ text: "" }) as never) as never,
    )(
      { id: "gpt-5.6-sol", provider: "openai", api: SOL_TRANSPORT_API } as never,
      { systemPrompt: fullPrompt } as never,
      {},
    );

    trace.recordSolTurnProbeIfActive({
      usage: { input: 50000, cacheRead: 20000, cacheWrite: undefined },
      compactionOccurredThisAttempt: true,
      compactionCount: 1,
      activeProcessSessions: [{ id: "p1" }, { id: "p2" }],
      sessionKey: "telegram:direct:12345678",
      sessionId: "test-session-id",
      fallbackReason: null,
    });

    const probeEvent = lines.find((l) => {
      try {
        return parseLine(l).stage === "sol:turn-probe";
      } catch {
        return false;
      }
    });
    expect(probeEvent).toBeDefined();
    const ev = parseLine(probeEvent!);

    // Stage
    expect(ev.stage).toBe("sol:turn-probe");

    // Hashes — must be 64-char hex strings.
    expect(ev.stablePrefixHash).toMatch(/^[0-9a-f]{64}$/);
    expect(ev.fullInstructionsHash).toMatch(/^[0-9a-f]{64}$/);

    // Fingerprint — 16-char hex.
    expect(ev.cacheKeyFingerprint).toMatch(/^[0-9a-f]{16}$/);

    // Numeric token fields.
    expect(ev.inputTokens).toBe(50000);
    expect(ev.cacheReadTokens).toBe(20000);
    // openai-chatgpt-responses does not report writes.
    expect(ev.cacheWriteAvailable).toBe(false);
    expect(ev.cacheWriteTokens).toBeUndefined();

    // Compaction.
    expect(ev.compactionMarker).toBe(true);
    expect(ev.compactionCount).toBe(1);

    // Active-process state.
    expect(ev.activeProcessStateCount).toBe(2);

    // No raw content.
    const s = JSON.stringify(ev);
    expect(s).not.toContain("## Context");
    expect(s).not.toContain("turn=1");
    expect(s).not.toContain("telegram:direct");
  });

  it("recordSolTurnProbeIfActive is a no-op when no Sol stream was wrapped", () => {
    const { lines, trace } = makeMemoryTrace();
    if (!trace) throw new Error("trace must be created");

    // Do NOT call wrapStreamFn with a Sol model — simulate a non-Sol turn.
    trace.wrapStreamFn(
      ((_model: unknown, _ctx: unknown, _opts: unknown) => ({ text: "" }) as never) as never,
    )(
      { id: "claude-sonnet-5", provider: "anthropic", api: "anthropic" } as never,
      { systemPrompt: "stable prefix only" } as never,
      {},
    );

    const countBefore = lines.length;
    trace.recordSolTurnProbeIfActive({
      usage: { input: 1000, cacheRead: 200, cacheWrite: undefined },
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: "telegram:direct:999",
      sessionId: "test-session-id",
      fallbackReason: null,
    });

    // No sol:turn-probe event should appear.
    const afterLines = lines.slice(countBefore);
    const hasProbe = afterLines.some((l) => {
      try {
        return parseLine(l).stage === "sol:turn-probe";
      } catch {
        return false;
      }
    });
    expect(hasProbe).toBe(false);
  });

  it("pending Sol context data is cleared after one probe so stale data is not reused", () => {
    const { lines, trace } = makeMemoryTrace();
    if (!trace) throw new Error("trace must be created");

    const doSolStream = (promptText: string) => {
      trace.wrapStreamFn(
        ((_model: unknown, _ctx: unknown, _opts: unknown) => ({ text: "" }) as never) as never,
      )(
        { id: "gpt-5.6-sol", provider: "openai", api: SOL_TRANSPORT_API } as never,
        { systemPrompt: promptText } as never,
        {},
      );
    };

    const doProbe = (turnLabel: string) => {
      trace.recordSolTurnProbeIfActive({
        usage: { input: 1000, cacheRead: 100, cacheWrite: undefined },
        compactionOccurredThisAttempt: false,
        compactionCount: 0,
        activeProcessSessions: [],
        sessionKey: "telegram:direct:12345678",
        sessionId: "test-session-id",
        fallbackReason: null,
      });
      void turnLabel; // used only in the description
    };

    // Turn 1: Sol stream then probe.
    const stable = "## Context";
    doSolStream(`${stable}${SYSTEM_PROMPT_CACHE_BOUNDARY}turn=1`);
    doProbe("turn-1");

    // Turn 2: non-Sol stream then probe attempt — should produce NO probe.
    trace.wrapStreamFn(
      ((_model: unknown, _ctx: unknown, _opts: unknown) => ({ text: "" }) as never) as never,
    )(
      { id: "claude-sonnet-5", provider: "anthropic", api: "anthropic" } as never,
      { systemPrompt: "not sol" } as never,
      {},
    );
    const probeCountBefore = lines.filter((l) => {
      try {
        return parseLine(l).stage === "sol:turn-probe";
      } catch {
        return false;
      }
    }).length;
    doProbe("turn-2-no-sol");
    const probeCountAfter = lines.filter((l) => {
      try {
        return parseLine(l).stage === "sol:turn-probe";
      } catch {
        return false;
      }
    }).length;

    expect(probeCountAfter).toBe(probeCountBefore); // no new probe for non-Sol turn

    // Turn 3: Sol stream again — should produce exactly one new probe.
    doSolStream(`${stable}${SYSTEM_PROMPT_CACHE_BOUNDARY}turn=3`);
    doProbe("turn-3");

    const allProbeEvents = lines.filter((l) => {
      try {
        return parseLine(l).stage === "sol:turn-probe";
      } catch {
        return false;
      }
    });
    expect(allProbeEvents.length).toBe(2); // turn-1 and turn-3 only
  });
});

// ---------------------------------------------------------------------------
// isSolTransport helper
// ---------------------------------------------------------------------------
describe("isSolTransport", () => {
  it("returns true for openai-chatgpt-responses (exact)", () => {
    expect(isSolTransport(SOL_TRANSPORT_API)).toBe(true);
  });

  it("returns true for mixed-case variants (case-insensitive)", () => {
    expect(isSolTransport("OpenAI-ChatGPT-Responses")).toBe(true);
    expect(isSolTransport("OPENAI-CHATGPT-RESPONSES")).toBe(true);
  });

  it("returns false for other transports", () => {
    expect(isSolTransport("openai-responses")).toBe(false);
    expect(isSolTransport("anthropic")).toBe(false);
    expect(isSolTransport("openai-completions")).toBe(false);
    expect(isSolTransport(null)).toBe(false);
    expect(isSolTransport(undefined)).toBe(false);
    expect(isSolTransport("")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// recordActiveProcessState helper
// ---------------------------------------------------------------------------
describe("recordActiveProcessState", () => {
  afterEach(() => {
    resetActiveProcessTrackerForTest();
  });

  it("returns null delta on first call for a session", () => {
    const result = recordActiveProcessState({
      sessionKey: "telegram:direct:1",
      sessionId: "sid-1",
      count: 3,
    });
    expect(result.count).toBe(3);
    expect(result.delta).toBeNull();
  });

  it("returns signed delta on subsequent calls", () => {
    recordActiveProcessState({ sessionKey: "telegram:direct:1", sessionId: "sid-1", count: 2 });
    const result = recordActiveProcessState({
      sessionKey: "telegram:direct:1",
      sessionId: "sid-1",
      count: 5,
    });
    expect(result.count).toBe(5);
    expect(result.delta).toBe(3);
  });

  it("returns negative delta when processes finish", () => {
    recordActiveProcessState({ sessionKey: "key-x", sessionId: "sid-x", count: 4 });
    const result = recordActiveProcessState({ sessionKey: "key-x", sessionId: "sid-x", count: 1 });
    expect(result.delta).toBe(-3);
  });

  it("independent sessions have independent baselines", () => {
    recordActiveProcessState({ sessionKey: "key-a", sessionId: "sid-a", count: 10 });
    const resultB = recordActiveProcessState({
      sessionKey: "key-b",
      sessionId: "sid-b",
      count: 2,
    });
    expect(resultB.delta).toBeNull(); // key-b has no prior baseline
  });
});

// ---------------------------------------------------------------------------
// buildSolTurnTokenResult
// ---------------------------------------------------------------------------
describe("buildSolTurnTokenResult", () => {
  afterEach(() => {
    resetActiveProcessTrackerForTest();
  });

  it("cacheWriteAvailable is false for openai-chatgpt-responses (API does not report writes)", () => {
    const result = buildSolTurnTokenResult({
      usage: { input: 10000, cacheRead: 5000, cacheWrite: undefined },
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: "telegram:direct:99",
      sessionId: "s1",
      fallbackReason: undefined,
    });
    expect(result.cacheWriteAvailable).toBe(false);
    expect(result.cacheWriteTokens).toBeUndefined();
  });

  it("records input and cacheRead from normalized usage", () => {
    const result = buildSolTurnTokenResult({
      usage: { input: 20000, cacheRead: 8000, cacheWrite: undefined },
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: "k1",
      sessionId: "s1",
      fallbackReason: null,
    });
    expect(result.inputTokens).toBe(20000);
    expect(result.cacheReadTokens).toBe(8000);
  });

  it("handles undefined usage without throwing", () => {
    const result = buildSolTurnTokenResult({
      usage: undefined,
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: "k1",
      sessionId: "s1",
      fallbackReason: null,
    });
    expect(result.inputTokens).toBeUndefined();
    expect(result.cacheReadTokens).toBeUndefined();
  });

  it("records compaction marker and count", () => {
    const result = buildSolTurnTokenResult({
      usage: undefined,
      compactionOccurredThisAttempt: true,
      compactionCount: 2,
      activeProcessSessions: [],
      sessionKey: "k2",
      sessionId: "s2",
      fallbackReason: null,
    });
    expect(result.compactionMarker).toBe(true);
    expect(result.compactionCount).toBe(2);
  });

  it("records fallback reason as modelTransition marker", () => {
    const result = buildSolTurnTokenResult({
      usage: undefined,
      compactionOccurredThisAttempt: false,
      compactionCount: 0,
      activeProcessSessions: [],
      sessionKey: "k3",
      sessionId: "s3",
      fallbackReason: "context-window-exceeded",
    });
    expect(result.fallbackReason).toBe("context-window-exceeded");
  });
});

// ---------------------------------------------------------------------------
// buildSolContextProbeData
// ---------------------------------------------------------------------------
describe("buildSolContextProbeData", () => {
  it("produces deterministic hashes and fingerprint", () => {
    const prompt = `stable${SYSTEM_PROMPT_CACHE_BOUNDARY}dynamic`;
    const d1 = buildSolContextProbeData({
      systemPrompt: prompt,
      sessionKey: "telegram:direct:12345678",
      sessionId: "sid",
      modelApi: SOL_TRANSPORT_API,
      provider: "openai",
      modelId: "gpt-5.6-sol",
    });
    const d2 = buildSolContextProbeData({
      systemPrompt: prompt,
      sessionKey: "telegram:direct:12345678",
      sessionId: "sid",
      modelApi: SOL_TRANSPORT_API,
      provider: "openai",
      modelId: "gpt-5.6-sol",
    });
    expect(d1.stablePrefixHash).toBe(d2.stablePrefixHash);
    expect(d1.fullInstructionsHash).toBe(d2.fullInstructionsHash);
    expect(d1.cacheKeyFingerprint).toBe(d2.cacheKeyFingerprint);
  });

  it("cacheKeyFingerprint does not expose the raw session key", () => {
    const rawKey = "telegram:direct:8456174966";
    const d = buildSolContextProbeData({
      systemPrompt: "ctx",
      sessionKey: rawKey,
      sessionId: "sid",
      modelApi: SOL_TRANSPORT_API,
      provider: "openai",
      modelId: "gpt-5.6-sol",
    });
    expect(d.cacheKeyFingerprint).not.toContain(rawKey);
    expect(d.cacheKeyFingerprint).not.toContain("telegram");
    expect(d.cacheKeyFingerprint).not.toContain("8456174966");
  });
});
