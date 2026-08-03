/**
 * Optional JSONL diagnostics for agent cache/session/prompt tracing.
 */
import crypto from "node:crypto";
import path from "node:path";
import { sanitizeSurrogates } from "@openclaw/ai/internal/shared";
import { resolveStateDir } from "../config/paths.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { resolveUserPath } from "../utils.js";
import { parseBooleanValue } from "../utils/boolean.js";
import { safeJsonStringify } from "../utils/safe-json.js";
import { redactAgentDiagnosticPayload } from "./diagnostic-redaction.js";
import { getQueuedFileWriter, type QueuedFileWriter } from "./queued-file-writer.js";
import type { AgentMessage, StreamFn } from "./runtime/index.js";
import {
  buildSolContextProbeData,
  buildSolTurnTokenResult,
  isSolTransport,
  type SolTurnProbeData,
} from "./sol-cache-probe.js";
import type { NormalizedUsage } from "./usage.js";
import { stableStringify } from "./stable-stringify.js";
import { buildAgentTraceBase } from "./trace-base.js";

// Payloads are redacted before JSONL output while stable digests preserve
// correlation across prompt/session/cache stages.
type CacheTraceStage =
  | "cache:result"
  | "cache:state"
  | "session:loaded"
  | "session:raw-model-run"
  | "session:sanitized"
  | "session:limited"
  | "prompt:before"
  | "prompt:images"
  | "sol:turn-probe"
  | "stream:context"
  | "session:after";

type CacheTraceEvent = {
  ts: string;
  seq: number;
  stage: CacheTraceStage;
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  provider?: string;
  modelId?: string;
  modelApi?: string | null;
  workspaceDir?: string;
  prompt?: string;
  system?: unknown;
  options?: Record<string, unknown>;
  model?: Record<string, unknown>;
  messages?: AgentMessage[];
  messageCount?: number;
  messageRoles?: Array<string | undefined>;
  messageFingerprints?: string[];
  messagesDigest?: string;
  systemDigest?: string;
  note?: string;
  error?: string;
  // Sol-specific probe fields — only set on sol:turn-probe stage events.
  // All values are hashes/fingerprints/numeric metadata; no raw content.
  stablePrefixHash?: string;
  fullInstructionsHash?: string;
  /**
   * Short fingerprint derived from the session/cache key via SHA-256.
   * Not reversible to the raw key; safe for diagnostic correlation.
   */
  cacheKeyFingerprint?: string;
  inputTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  /**
   * False when the provider API did not include cache-write data in the
   * response. Recorded as unavailable rather than zero.
   */
  cacheWriteAvailable?: boolean;
  compactionMarker?: boolean;
  compactionCount?: number;
  activeProcessStateCount?: number;
  /** Signed change vs the previous turn; null on the first turn for a session. */
  activeProcessStateDelta?: number | null;
  /** Non-null when a model fallback or transition occurred this turn. */
  modelTransition?: string | null;
};

/** Parameters for `recordSolTurnProbeIfActive`. Callers supply raw attempt data. */
export type SolTurnProbeAttemptParams = {
  usage: NormalizedUsage | undefined;
  compactionOccurredThisAttempt: boolean;
  compactionCount: number;
  activeProcessSessions: readonly unknown[];
  sessionKey: string | undefined;
  sessionId: string;
  fallbackReason: string | null | undefined;
};

type CacheTrace = {
  enabled: true;
  filePath: string;
  recordStage: (stage: CacheTraceStage, payload?: Partial<CacheTraceEvent>) => void;
  wrapStreamFn: (streamFn: StreamFn) => StreamFn;
  /**
   * Emit a `sol:turn-probe` event after a Sol turn completes.
   *
   * No-ops when the last wrapped stream was not openai-chatgpt-responses.
   * The context probe data captured in `wrapStreamFn` is combined with the
   * supplied post-turn data. Clears the pending context after recording so
   * stale data is never carried into the next turn.
   *
   * Call after the existing `cache:result` stage recording.
   */
  recordSolTurnProbeIfActive: (params: SolTurnProbeAttemptParams) => void;
};

type CacheTraceInit = {
  cfg?: OpenClawConfig;
  env?: NodeJS.ProcessEnv;
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  provider?: string;
  modelId?: string;
  modelApi?: string | null;
  workspaceDir?: string;
  writer?: CacheTraceWriter;
};

type CacheTraceConfig = {
  enabled: boolean;
  filePath: string;
  includeMessages: boolean;
  includePrompt: boolean;
  includeSystem: boolean;
};

type CacheTraceWriter = QueuedFileWriter;

const writers = new Map<string, CacheTraceWriter>();

function resolveCacheTraceConfig(params: CacheTraceInit): CacheTraceConfig {
  const env = params.env ?? process.env;
  const config = params.cfg?.diagnostics?.cacheTrace;
  const envEnabled = parseBooleanValue(env.OPENCLAW_CACHE_TRACE);
  const enabled = envEnabled ?? config?.enabled ?? false;
  const fileOverride = config?.filePath?.trim() || env.OPENCLAW_CACHE_TRACE_FILE?.trim();
  const filePath = fileOverride
    ? resolveUserPath(fileOverride)
    : path.join(resolveStateDir(env), "logs", "cache-trace.jsonl");

  const includeMessages =
    parseBooleanValue(env.OPENCLAW_CACHE_TRACE_MESSAGES) ?? config?.includeMessages;
  const includePrompt = parseBooleanValue(env.OPENCLAW_CACHE_TRACE_PROMPT) ?? config?.includePrompt;
  const includeSystem = parseBooleanValue(env.OPENCLAW_CACHE_TRACE_SYSTEM) ?? config?.includeSystem;

  return {
    enabled,
    filePath,
    includeMessages: includeMessages ?? true,
    includePrompt: includePrompt ?? true,
    includeSystem: includeSystem ?? true,
  };
}

function getWriter(filePath: string): CacheTraceWriter {
  return getQueuedFileWriter(writers, filePath);
}

function digest(value: unknown): string {
  const serialized = stableStringify(value, sanitizeSurrogates);
  return crypto.createHash("sha256").update(serialized).digest("hex");
}

export function summarizeMessages(messages: AgentMessage[]): {
  messageCount: number;
  messageRoles: Array<string | undefined>;
  messageFingerprints: string[];
  messagesDigest: string;
} {
  // Hash each message and then the ordered fingerprint list so traces can detect
  // prompt drift without writing full messages when disabled.
  const messageFingerprints = messages.map((msg) => digest(msg));
  return {
    messageCount: messages.length,
    messageRoles: messages.map((msg) => (msg as { role?: string }).role),
    messageFingerprints,
    messagesDigest: digest(messageFingerprints.join("|")),
  };
}

/** Create a cache trace recorder when diagnostics config/env enables it. */
export function createCacheTrace(params: CacheTraceInit): CacheTrace | null {
  const cfg = resolveCacheTraceConfig(params);
  if (!cfg.enabled) {
    return null;
  }

  const writer = params.writer ?? getWriter(cfg.filePath);
  let seq = 0;

  const base: Omit<CacheTraceEvent, "ts" | "seq" | "stage"> = buildAgentTraceBase(params);

  // Stores the Sol instruction-level probe data captured in wrapStreamFn
  // so it can be combined with post-turn token data in recordSolTurnProbeIfActive.
  let pendingSolContextData: SolTurnProbeData | null = null;

  const recordStage: CacheTrace["recordStage"] = (stage, payload = {}) => {
    const event: CacheTraceEvent = {
      ...base,
      ts: new Date().toISOString(),
      seq: (seq += 1),
      stage,
    };

    if (payload.prompt !== undefined && cfg.includePrompt) {
      event.prompt = redactAgentDiagnosticPayload(payload.prompt);
    }
    if (payload.system !== undefined && cfg.includeSystem) {
      event.system = redactAgentDiagnosticPayload(payload.system);
      event.systemDigest = digest(payload.system);
    }
    if (payload.options) {
      event.options = redactAgentDiagnosticPayload(payload.options);
    }
    if (payload.model) {
      event.model = redactAgentDiagnosticPayload(payload.model);
    }

    const messages = payload.messages;
    if (Array.isArray(messages)) {
      const summary = summarizeMessages(messages);
      event.messageCount = summary.messageCount;
      event.messageRoles = summary.messageRoles;
      event.messageFingerprints = summary.messageFingerprints;
      event.messagesDigest = summary.messagesDigest;
      if (cfg.includeMessages) {
        // Full messages are optional; summaries/digests are always recorded when
        // message payloads are supplied.
        event.messages = redactAgentDiagnosticPayload(messages);
      }
    }

    if (payload.note) {
      event.note = redactAgentDiagnosticPayload(payload.note);
    }
    if (payload.error) {
      event.error = redactAgentDiagnosticPayload(payload.error);
    }

    const line = safeJsonStringify(event);
    if (!line) {
      return;
    }
    writer.write(`${line}\n`);
  };

  const wrapStreamFn: CacheTrace["wrapStreamFn"] = (streamFn) => {
    const wrapped: StreamFn = (model, context, options) => {
      const traceContext = context as {
        messages?: AgentMessage[];
        system?: unknown;
        systemPrompt?: unknown;
      };
      recordStage("stream:context", {
        model: {
          id: model?.id,
          provider: model?.provider,
          api: model?.api,
        },
        system: traceContext.systemPrompt ?? traceContext.system,
        messages: traceContext.messages ?? [],
        options: (options ?? {}) as Record<string, unknown>,
      });

      // Sol-specific: capture instruction hashes and cache-key fingerprint when
      // the transport is openai-chatgpt-responses. Stored for combination with
      // post-turn token data in recordSolTurnProbeIfActive.
      if (isSolTransport(model?.api as string | null | undefined)) {
        const rawSystemPrompt =
          typeof traceContext.systemPrompt === "string"
            ? traceContext.systemPrompt
            : typeof traceContext.system === "string"
              ? traceContext.system
              : "";
        pendingSolContextData = buildSolContextProbeData({
          systemPrompt: rawSystemPrompt,
          sessionKey: params.sessionKey,
          sessionId: params.sessionId ?? "",
          modelApi: model?.api as string,
          provider: model?.provider as string | undefined,
          modelId: model?.id as string | undefined,
        });
      }

      return streamFn(model, context, options);
    };
    return wrapped;
  };

  const recordSolTurnProbeIfActive: CacheTrace["recordSolTurnProbeIfActive"] = (probeParams) => {
    const contextData = pendingSolContextData;
    if (!contextData) {
      // No Sol stream was captured this turn; no-op.
      return;
    }
    // Clear pending state so stale data is never carried into the next turn.
    pendingSolContextData = null;

    const tokenResult = buildSolTurnTokenResult({
      usage: probeParams.usage,
      compactionOccurredThisAttempt: probeParams.compactionOccurredThisAttempt,
      compactionCount: probeParams.compactionCount,
      activeProcessSessions: probeParams.activeProcessSessions,
      sessionKey: probeParams.sessionKey,
      sessionId: probeParams.sessionId,
      fallbackReason: probeParams.fallbackReason,
    });

    // Build the probe event. Raw session key, prompt text, and message contents
    // are deliberately excluded — only hashes, fingerprints, and numeric metadata.
    const event: CacheTraceEvent = {
      // Omit sessionKey and workspaceDir from base; cacheKeyFingerprint replaces them.
      ts: new Date().toISOString(),
      seq: (seq += 1),
      stage: "sol:turn-probe",
      runId: base.runId,
      sessionId: base.sessionId,
      provider: contextData.provider ?? base.provider,
      modelId: contextData.modelId ?? base.modelId,
      modelApi: contextData.modelApi,
      // Instruction hashes — no raw text.
      stablePrefixHash: contextData.stablePrefixHash,
      fullInstructionsHash: contextData.fullInstructionsHash,
      // Fingerprint — not reversible to the raw session/cache key.
      cacheKeyFingerprint: contextData.cacheKeyFingerprint,
      // Token counts — numeric only.
      inputTokens: tokenResult.inputTokens,
      cacheReadTokens: tokenResult.cacheReadTokens,
      cacheWriteTokens: tokenResult.cacheWriteTokens,
      cacheWriteAvailable: tokenResult.cacheWriteAvailable,
      // Compaction state.
      compactionMarker: tokenResult.compactionMarker,
      compactionCount: tokenResult.compactionCount,
      // Active-process state.
      activeProcessStateCount: tokenResult.activeProcessStateCount,
      activeProcessStateDelta: tokenResult.activeProcessStateDelta,
      // Model fallback/transition — null means no transition this turn.
      modelTransition: tokenResult.fallbackReason ?? null,
    };

    const line = safeJsonStringify(event);
    if (!line) {
      return;
    }
    writer.write(`${line}\n`);
  };

  return {
    enabled: true,
    filePath: cfg.filePath,
    recordStage,
    wrapStreamFn,
    recordSolTurnProbeIfActive,
  };
}
