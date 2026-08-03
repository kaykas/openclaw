/**
 * Sol session-cache probe for openai-chatgpt-responses transport.
 *
 * Records only hashes/fingerprints and numeric metadata — no raw prompt,
 * session key, cache key, or secret content is written to any log.
 *
 * The stable-prefix boundary follows the existing `SYSTEM_PROMPT_CACHE_BOUNDARY`
 * marker (`\n<!-- OPENCLAW_CACHE_BOUNDARY -->\n`) which OpenClaw inserts between
 * the invariant bootstrap context and the per-turn dynamic suffix.
 */
import crypto from "node:crypto";
import { splitSystemPromptCacheBoundary } from "@openclaw/ai/internal/shared";
import type { NormalizedUsage } from "./usage.js";

/** The API transport string for Sol (openai-chatgpt-responses). */
export const SOL_TRANSPORT_API = "openai-chatgpt-responses";

// Real Sol traffic resolves model.api to "openai-responses", a distinct
// value the rest of the codebase already treats as a sibling of
// "openai-chatgpt-responses" everywhere else. isSolTransport previously
// matched SOL_TRANSPORT_API only and missed real Sol traffic entirely
// (found during the 2026-08-02 cache probe). Match both.
const SOL_TRANSPORT_APIS = new Set(["openai-chatgpt-responses", "openai-responses"]);

/**
 * Returns true when the model API is the Sol transport.
 * Matches case-insensitively to survive provider-object normalization.
 */
export function isSolTransport(modelApi: string | null | undefined): boolean {
  return typeof modelApi === "string" && SOL_TRANSPORT_APIS.has(modelApi.toLowerCase());
}

function sha256Hex(value: string): string {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

/**
 * Compute stable-prefix hash and full-instructions hash from a raw system
 * prompt string (before the cache boundary is stripped for the API call).
 *
 * - `stablePrefixHash`: SHA-256 of everything before `SYSTEM_PROMPT_CACHE_BOUNDARY`.
 *   When no boundary is present, the entire prompt is treated as stable.
 *   This value is invariant across turns if the stable context hasn't changed
 *   — timestamps and per-turn additions placed after the boundary do not affect it.
 *
 * - `fullInstructionsHash`: SHA-256 of the complete raw prompt including the
 *   boundary marker and any dynamic suffix. Changes every turn if the suffix
 *   changes, even when the stable prefix is unchanged.
 *
 * Neither hash contains raw text.
 */
export function computeInstructionHashes(systemPrompt: string): {
  stablePrefixHash: string;
  fullInstructionsHash: string;
} {
  const split = splitSystemPromptCacheBoundary(systemPrompt);
  // When there is no boundary the whole prompt is the stable prefix; hash it once.
  const stablePrefixHash = sha256Hex(split?.stablePrefix ?? systemPrompt);
  const fullInstructionsHash = sha256Hex(systemPrompt);
  return { stablePrefixHash, fullInstructionsHash };
}

/**
 * Produce a short one-way fingerprint of a raw session/cache key.
 *
 * The fingerprint is collision-resistant for correlation purposes but is NOT
 * reversible and does not expose the raw key. Callers must never log the input.
 * A constant prefix prevents pre-computed rainbow-table reuse.
 */
export function fingerprintSessionKey(rawKey: string): string {
  // 16 hex chars = 64 bits. Sufficient for correlation; not reversible.
  return sha256Hex(`sol-cache-key-v1:${rawKey}`).slice(0, 16);
}

// ---------------------------------------------------------------------------
// Active-process-state tracker (stateful, in-process, per logical session)
// ---------------------------------------------------------------------------

type ActiveProcessEntry = { count: number };
const activeProcessRegistry = new Map<string, ActiveProcessEntry>();
const MAX_PROCESS_REGISTRY_ENTRIES = 256;

function resolveActiveProcessRegistryKey(params: {
  sessionKey: string | undefined;
  sessionId: string;
}): string {
  return (params.sessionKey?.trim() || params.sessionId).trim();
}

/**
 * Record the current active-process count for a session and return the count
 * plus the delta since the last call for the same session key.
 *
 * - `delta` is `null` on the first call for a session (no prior baseline).
 * - Subsequent calls return the signed difference (positive = more processes,
 *   negative = fewer, zero = no change).
 */
export function recordActiveProcessState(params: {
  sessionKey: string | undefined;
  sessionId: string;
  count: number;
}): { count: number; delta: number | null } {
  const key = resolveActiveProcessRegistryKey(params);
  const existing = activeProcessRegistry.get(key);

  if (existing === undefined && activeProcessRegistry.size >= MAX_PROCESS_REGISTRY_ENTRIES) {
    const oldest = activeProcessRegistry.keys().next().value;
    if (typeof oldest === "string") {
      activeProcessRegistry.delete(oldest);
    }
  }

  const delta = existing !== undefined ? params.count - existing.count : null;
  if (existing !== undefined) {
    existing.count = params.count;
  } else {
    activeProcessRegistry.set(key, { count: params.count });
  }

  return { count: params.count, delta };
}

/** Clear the active-process tracker. Call only from test setup/teardown. */
export function resetActiveProcessTrackerForTest(): void {
  activeProcessRegistry.clear();
}

// ---------------------------------------------------------------------------
// Probe data assembly
// ---------------------------------------------------------------------------

/**
 * Fields recorded in a `sol:turn-probe` cache-trace event.
 * All values are hashes, fingerprints, or numeric metadata — no raw content.
 */
export type SolTurnProbeData = {
  /** SHA-256 of the stable prefix (before the OPENCLAW_CACHE_BOUNDARY marker). */
  stablePrefixHash: string;
  /** SHA-256 of the full raw system prompt including any dynamic suffix. */
  fullInstructionsHash: string;
  /** Short fingerprint of the session/cache key — not reversible to the raw key. */
  cacheKeyFingerprint: string;
  /** Model/API transport identifier for the probe source. */
  modelApi: string;
  /** Provider identifier for the probe source. */
  provider: string | undefined;
  /** Model identifier for the probe source. */
  modelId: string | undefined;
};

/**
 * Build the instruction-level fields of a Sol turn probe from stream-context data.
 * Call this inside `wrapStreamFn` when `modelApi === SOL_TRANSPORT_API`.
 */
export function buildSolContextProbeData(params: {
  systemPrompt: string;
  sessionKey: string | undefined;
  sessionId: string;
  modelApi: string;
  provider: string | undefined;
  modelId: string | undefined;
}): SolTurnProbeData {
  const { stablePrefixHash, fullInstructionsHash } = computeInstructionHashes(params.systemPrompt);
  const cacheKeyFingerprint = fingerprintSessionKey(params.sessionKey ?? params.sessionId);
  return {
    stablePrefixHash,
    fullInstructionsHash,
    cacheKeyFingerprint,
    modelApi: params.modelApi,
    provider: params.provider,
    modelId: params.modelId,
  };
}

/**
 * Post-turn token result for a Sol turn probe.
 * `cacheWriteAvailable = false` means the provider response did not include a
 * cache-write count; this must be recorded as unavailable rather than zero.
 */
export type SolTurnTokenResult = {
  inputTokens: number | undefined;
  cacheReadTokens: number | undefined;
  cacheWriteTokens: number | undefined;
  /** False when the provider API response did not include cache-write data. */
  cacheWriteAvailable: boolean;
  compactionMarker: boolean;
  compactionCount: number;
  activeProcessStateCount: number;
  activeProcessStateDelta: number | null;
  /** Non-null when a model fallback/transition occurred this turn. */
  fallbackReason: string | null;
};

/**
 * Extract the Sol post-turn token result from normalized turn data.
 * For openai-chatgpt-responses, cache writes are not reported by the API
 * (OpenAI uses automatic prefix caching with no explicit write signal) so
 * `cacheWriteAvailable` is always false for this transport.
 */
export function buildSolTurnTokenResult(params: {
  usage: NormalizedUsage | undefined;
  compactionOccurredThisAttempt: boolean;
  compactionCount: number;
  activeProcessSessions: readonly unknown[];
  sessionKey: string | undefined;
  sessionId: string;
  fallbackReason: string | null | undefined;
}): SolTurnTokenResult {
  const usage = params.usage;
  const activeProcessState = recordActiveProcessState({
    sessionKey: params.sessionKey,
    sessionId: params.sessionId,
    count: params.activeProcessSessions.length,
  });
  return {
    inputTokens: usage?.input,
    cacheReadTokens: usage?.cacheRead,
    // openai-chatgpt-responses does not report cache writes — the API response
    // only surfaces cached reads via input_tokens_details.cached_tokens.
    cacheWriteTokens: undefined,
    cacheWriteAvailable: false,
    compactionMarker: params.compactionOccurredThisAttempt,
    compactionCount: params.compactionCount,
    activeProcessStateCount: activeProcessState.count,
    activeProcessStateDelta: activeProcessState.delta,
    fallbackReason: params.fallbackReason ?? null,
  };
}
