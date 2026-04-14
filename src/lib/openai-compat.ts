/**
 * OpenAI compatibility module. Ported from Python openai_compat.py.
 */

import { z } from "zod";

// --- Zod schemas (with passthrough for extra fields) ---

const ChatMessageRoleSchema = z.enum(["system", "user", "assistant", "tool", "developer"]);

export const ChatMessageSchema = z
  .object({
    role: ChatMessageRoleSchema,
    content: z.unknown(),
  })
  .passthrough();

export const ChatCompletionRequestSchema = z
  .object({
    model: z.string().nullable().optional(),
    messages: z.array(ChatMessageSchema),
    stream: z.boolean().optional().default(false),
    max_tokens: z.number().nullable().optional(),
  })
  .passthrough();

export const ErrorResponseSchema = z
  .object({
    error: z.record(z.unknown()).optional().default({}),
  })
  .passthrough();

export const ChatCompletionRequestCompatSchema = z
  .object({
    model: z.string().nullable().optional(),
    messages: z.array(ChatMessageSchema).nullable().optional(),
    input: z.unknown().optional(),
    instructions: z.string().nullable().optional(),
    stream: z.boolean().optional().default(false),
    max_tokens: z.number().nullable().optional(),
    max_output_tokens: z.number().nullable().optional(),
  })
  .passthrough();

export const ResponsesRequestSchema = z
  .object({
    model: z.string().nullable().optional(),
    input: z.unknown().optional(),
    stream: z.boolean().optional().default(false),
    max_output_tokens: z.number().nullable().optional(),
    instructions: z.string().nullable().optional(),
  })
  .passthrough();

// --- Inferred types ---
export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;
export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;
export type ChatCompletionRequestCompat = z.infer<typeof ChatCompletionRequestCompatSchema>;
export type ResponsesRequest = z.infer<typeof ResponsesRequestSchema>;

type LooseRecord = Record<string, unknown>;

function copyMissingToolingFields(target: LooseRecord, source: LooseRecord, keys: string[]): boolean {
  let changed = false;
  for (const key of keys) {
    if (source[key] !== undefined && target[key] === undefined) {
      target[key] = source[key];
      changed = true;
    }
  }
  return changed;
}

function normalizeToolingMessage(message: ChatMessage): ChatMessage {
  const raw = message as LooseRecord;
  const extraRaw = raw.model_extra;
  const extra: LooseRecord =
    typeof extraRaw === "object" && extraRaw !== null && !Array.isArray(extraRaw)
      ? { ...(extraRaw as LooseRecord) }
      : {};

  let changed = false;

  if (message.role === "assistant") {
    changed =
      copyMissingToolingFields(extra, raw, ["tool_calls", "function_call"]) || changed;
  } else if (message.role === "tool") {
    changed = copyMissingToolingFields(extra, raw, ["tool_call_id", "call_id"]) || changed;
  }

  if (!changed) return message;
  return { ...raw, model_extra: extra } as unknown as ChatMessage;
}

/**
 * Mirror top-level OpenAI tool-call fields into model_extra for providers that
 * currently read internal tooling config from model_extra while preserving
 * existing model_extra callers as the source of truth.
 */
export function normalizeToolingRequest(req: ChatCompletionRequest): ChatCompletionRequest {
  const raw = req as LooseRecord;
  const extraRaw = raw.model_extra;
  const extra: LooseRecord =
    typeof extraRaw === "object" && extraRaw !== null && !Array.isArray(extraRaw)
      ? { ...(extraRaw as LooseRecord) }
      : {};

  const requestExtraChanged = copyMissingToolingFields(extra, raw, [
    "tools",
    "tool_choice",
    "parallel_tool_calls",
  ]);

  let messageChanged = false;
  const messages = req.messages.map((message) => {
    const normalized = normalizeToolingMessage(message);
    if (normalized !== message) messageChanged = true;
    return normalized;
  });

  if (!requestExtraChanged && !messageChanged) return req;

  const next: LooseRecord = { ...raw, messages };
  if (requestExtraChanged || raw.model_extra !== undefined) {
    next.model_extra = extra;
  }
  return next as ChatCompletionRequest;
}

// --- Helper: coerce Responses API part ---
function _coerceResponsesPart(part: Record<string, unknown>): Record<string, unknown> | null {
  const partType = part.type;
  if (
    (partType === "input_text" || partType === "output_text" || partType === "text") &&
    typeof part.text === "string"
  ) {
    return { type: "text", text: part.text };
  }
  if (partType === "image_url" || partType === "input_image") {
    return part;
  }
  return null;
}

function _coerceResponsesContent(content: unknown): string | Array<Record<string, unknown>> {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (typeof content === "object" && !Array.isArray(content) && content !== null) {
    const coerced = _coerceResponsesPart(content as Record<string, unknown>);
    if (coerced !== null) {
      const t = coerced.type;
      if (t === "image_url" || t === "input_image") {
        return [coerced];
      }
      return (coerced.text as string) ?? "";
    }
    return String(content);
  }
  if (Array.isArray(content)) {
    const parts: Array<Record<string, unknown>> = [];
    for (const part of content) {
      if (typeof part !== "object" || part === null || Array.isArray(part)) continue;
      const coerced = _coerceResponsesPart(part as Record<string, unknown>);
      if (coerced !== null) parts.push(coerced);
    }
    if (parts.length > 0) return parts;
    const texts = content
      .filter((p): p is Record<string, unknown> => typeof p === "object" && p !== null && !Array.isArray(p))
      .map((p) => p.text)
      .filter((t): t is string => typeof t === "string");
    if (texts.length > 0) return texts.join("");
    return "";
  }
  return String(content);
}

/** Convert Responses API input to ChatMessage[]. */
export function responsesInputToMessages(inputObj: unknown): ChatMessage[] {
  const messages: ChatMessage[] = [];

  function add(role: string, content: unknown): void {
    messages.push({ role: role as ChatMessage["role"], content });
  }

  function coerceItem(item: unknown): void {
    if (item === null || item === undefined) return;
    if (typeof item === "string") {
      add("user", item);
      return;
    }
    if (typeof item !== "object" || Array.isArray(item)) {
      add("user", String(item));
      return;
    }
    const obj = item as Record<string, unknown>;
    const role = obj.role;
    const itemType = obj.type;
    if (itemType === "message" || typeof role === "string") {
      const r = typeof role === "string" ? role : "user";
      add(r, _coerceResponsesContent(obj.content));
      return;
    }
    if (
      (itemType === "input_text" || itemType === "output_text" || itemType === "text") &&
      typeof obj.text === "string"
    ) {
      add("user", obj.text);
      return;
    }
    if (itemType === "image_url" || itemType === "input_image") {
      add("user", _coerceResponsesContent(obj));
      return;
    }
  }

  if (inputObj === null || inputObj === undefined) return messages;
  if (Array.isArray(inputObj)) {
    for (const item of inputObj) coerceItem(item);
    return messages;
  }
  coerceItem(inputObj);
  return messages;
}

/** Convert ResponsesRequest to ChatCompletionRequest. */
export function responsesRequestToChatRequest(req: ResponsesRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = [];
  if (typeof req.instructions === "string" && req.instructions.trim()) {
    messages.push({ role: "system", content: req.instructions });
  }
  messages.push(...responsesInputToMessages(req.input));

  const extra = { ...req };
  delete (extra as Record<string, unknown>).model;
  delete (extra as Record<string, unknown>).input;
  delete (extra as Record<string, unknown>).stream;
  delete (extra as Record<string, unknown>).max_output_tokens;
  delete (extra as Record<string, unknown>).max_tokens;
  delete (extra as Record<string, unknown>).instructions;

  let maxTokens = req.max_output_tokens;
  if (maxTokens === undefined && typeof (extra as Record<string, unknown>).max_tokens === "number") {
    maxTokens = (extra as Record<string, unknown>).max_tokens as number;
  }

  return normalizeToolingRequest({
    ...extra,
    model: req.model ?? null,
    messages,
    stream: req.stream ?? false,
    max_tokens: maxTokens ?? null,
  } as ChatCompletionRequest);
}

/** Normalize compat request to standard ChatCompletionRequest. */
export function compatChatRequestToChatRequest(
  req: ChatCompletionRequest | ChatCompletionRequestCompat
): ChatCompletionRequest {
  if ("messages" in req && Array.isArray(req.messages) && req.messages.length > 0) {
    return normalizeToolingRequest(req as ChatCompletionRequest);
  }

  const compat = req as ChatCompletionRequestCompat;
  let messages: ChatMessage[] = [];
  if (typeof compat.instructions === "string" && compat.instructions.trim()) {
    messages.push({ role: "system", content: compat.instructions });
  }
  messages = messages.concat(responsesInputToMessages(compat.input));

  if (messages.length === 0) {
    throw new Error("Missing messages or input");
  }

  const extra = { ...compat };
  delete (extra as Record<string, unknown>).model;
  delete (extra as Record<string, unknown>).messages;
  delete (extra as Record<string, unknown>).input;
  delete (extra as Record<string, unknown>).instructions;
  delete (extra as Record<string, unknown>).stream;
  delete (extra as Record<string, unknown>).max_tokens;
  delete (extra as Record<string, unknown>).max_output_tokens;

  let maxTokens = compat.max_tokens;
  if (maxTokens === undefined && compat.max_output_tokens !== undefined) {
    maxTokens = compat.max_output_tokens;
  }
  if (maxTokens === undefined && typeof (extra as Record<string, unknown>).max_tokens === "number") {
    maxTokens = (extra as Record<string, unknown>).max_tokens as number;
  }

  return normalizeToolingRequest({
    ...extra,
    model: compat.model ?? null,
    messages,
    stream: compat.stream ?? false,
    max_tokens: maxTokens ?? null,
  } as ChatCompletionRequest);
}

/** Extract text from various content formats. */
export function normalizeMessageContent(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (typeof part !== "object" || part === null || Array.isArray(part)) continue;
      const p = part as Record<string, unknown>;
      if (p.type === "text" && typeof p.text === "string") parts.push(p.text);
    }
    return parts.join("");
  }
  if (typeof content === "object" && content !== null && !Array.isArray(content)) {
    const obj = content as Record<string, unknown>;
    if (obj.type === "text" && typeof obj.text === "string") return obj.text;
  }
  return String(content);
}

/** Convert messages array to "ROLE: text" format. */
export function messagesToPrompt(messages: ChatMessage[]): string {
  const parts = messages.map((m) => {
    const role = m.role.toUpperCase();
    const text = normalizeMessageContent(m.content);
    return `${role}: ${text}`;
  });
  return parts.join("\n\n").trim();
}

/** Extract image URLs from a single content value. */
export function extractImageUrlsFromContent(content: unknown): string[] {
  const urls: string[] = [];
  if (content === null || content === undefined) return urls;

  if (typeof content === "object" && !Array.isArray(content) && content !== null) {
    const obj = content as Record<string, unknown>;
    const partType = obj.type;
    if (partType === "image_url" || partType === "input_image") {
      const image = obj.image_url;
      if (typeof image === "object" && image !== null && !Array.isArray(image)) {
        const url = (image as Record<string, unknown>).url;
        if (typeof url === "string" && url) urls.push(url);
      } else if (typeof image === "string" && image) {
        urls.push(image);
      }
    }
    return urls;
  }

  if (!Array.isArray(content)) return urls;

  for (const part of content) {
    if (typeof part !== "object" || part === null || Array.isArray(part)) continue;
    const p = part as Record<string, unknown>;
    const partType = p.type;
    if (partType !== "image_url" && partType !== "input_image") continue;
    const image = p.image_url;
    if (typeof image === "object" && image !== null && !Array.isArray(image)) {
      const url = (image as Record<string, unknown>).url;
      if (typeof url === "string" && url) urls.push(url);
    } else if (typeof image === "string" && image) {
      urls.push(image);
    }
  }
  return urls;
}

/** Extract image URLs from all messages. */
export function extractImageUrls(messages: ChatMessage[]): string[] {
  const urls: string[] = [];
  for (const message of messages) {
    urls.push(...extractImageUrlsFromContent(message.content));
  }
  return urls;
}

// ─── Tool Call support via prompt engineering ────────────────────────────────

interface ToolFunction {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}
interface ToolDef {
  type: string;
  function: ToolFunction;
}

const TOOL_CALL_MARKER = "___TOOL_CALL___";

/**
 * Extract balanced JSON objects between TOOL_CALL_MARKER pairs.
 * Uses brace-depth counting to correctly handle nested objects.
 */
function extractToolCallBlocks(text: string): { json: string; fullMatch: string }[] {
  const results: { json: string; fullMatch: string }[] = [];
  let searchFrom = 0;
  while (true) {
    const startIdx = text.indexOf(TOOL_CALL_MARKER, searchFrom);
    if (startIdx === -1) break;
    const afterStart = startIdx + TOOL_CALL_MARKER.length;

    const braceStart = text.indexOf("{", afterStart);
    if (braceStart === -1) break;

    let depth = 0;
    let inString = false;
    let escape = false;
    let braceEnd = -1;
    for (let i = braceStart; i < text.length; i++) {
      const ch = text[i];
      if (escape) { escape = false; continue; }
      if (ch === "\\") { escape = true; continue; }
      if (ch === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (ch === "{") depth++;
      else if (ch === "}") { depth--; if (depth === 0) { braceEnd = i; break; } }
    }
    if (braceEnd === -1) break;

    const endMarkerIdx = text.indexOf(TOOL_CALL_MARKER, braceEnd + 1);
    if (endMarkerIdx === -1) break;

    const json = text.slice(braceStart, braceEnd + 1);
    const fullMatch = text.slice(startIdx, endMarkerIdx + TOOL_CALL_MARKER.length);
    results.push({ json, fullMatch });
    searchFrom = endMarkerIdx + TOOL_CALL_MARKER.length;
  }
  return results;
}

/**
 * Build a system instruction that teaches the model how to invoke tools.
 * Returns empty string if no tools are provided.
 */
export function buildToolCallSystemPrompt(tools: unknown[] | undefined): string {
  if (!tools || !Array.isArray(tools) || tools.length === 0) return "";

  const defs = tools
    .filter((t): t is ToolDef => t !== null && typeof t === "object" && "function" in t)
    .map((t) => t.function);
  if (defs.length === 0) return "";

  const schema = defs.map((f) => {
    const params = f.parameters ? JSON.stringify(f.parameters) : "{}";
    return `- ${f.name}: ${f.description || "(no description)"}\n  parameters: ${params}`;
  }).join("\n");

  return [
    `You have access to the following tools. When you need to call a tool, respond EXACTLY in this format (no other text around the markers):`,
    ``,
    `${TOOL_CALL_MARKER}`,
    `{"name": "<function_name>", "arguments": {<argument_object>}}`,
    `${TOOL_CALL_MARKER}`,
    ``,
    `Available tools:`,
    schema,
    ``,
    `IMPORTANT RULES:`,
    `- If you decide to call a tool, output ONLY the marker block above, nothing else.`,
    `- You may call only ONE tool at a time.`,
    `- If you do NOT need a tool, respond normally without any markers.`,
  ].join("\n");
}

/**
 * Format tool-role messages into the prompt text.
 */
export function formatToolResultMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => {
    if (m.role === "tool") {
      const toolCallId = (m as Record<string, unknown>).tool_call_id ?? "unknown";
      const content = normalizeMessageContent(m.content);
      return {
        role: "user" as const,
        content: `[Tool Result for call ${toolCallId}]:\n${content}`,
      };
    }
    if (m.role === "assistant" && (m as Record<string, unknown>).tool_calls) {
      const tc = (m as Record<string, unknown>).tool_calls as Array<Record<string, unknown>>;
      const text = tc.map((call) => {
        const fn = call.function as Record<string, unknown> | undefined;
        const name = fn?.name ?? "";
        let args: string;
        if (typeof fn?.arguments === "string") {
          args = fn.arguments;
        } else {
          args = JSON.stringify(fn?.arguments ?? {});
        }
        const obj = JSON.stringify({ name, arguments: JSON.parse(args) });
        return `${TOOL_CALL_MARKER}\n${obj}\n${TOOL_CALL_MARKER}`;
      }).join("\n");
      return { role: "assistant" as const, content: text };
    }
    return m;
  });
}

export interface ParsedToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/**
 * Parse the model's text response and extract tool calls if present.
 * Returns { text, toolCalls } where toolCalls is null if no calls found.
 */
export function parseToolCallResponse(
  text: string
): { text: string; toolCalls: ParsedToolCall[] | null } {
  const blocks = extractToolCallBlocks(text);
  if (blocks.length === 0) {
    return { text, toolCalls: null };
  }

  const toolCalls: ParsedToolCall[] = [];
  let remaining = text;

  for (const block of blocks) {
    try {
      const parsed = JSON.parse(block.json) as { name?: string; arguments?: unknown };
      if (parsed.name) {
        toolCalls.push({
          id: `call_${randomCallId()}`,
          type: "function",
          function: {
            name: parsed.name,
            arguments: typeof parsed.arguments === "string"
              ? parsed.arguments
              : JSON.stringify(parsed.arguments ?? {}),
          },
        });
      }
    } catch {
      // not valid JSON, skip
    }
    remaining = remaining.replace(block.fullMatch, "").trim();
  }

  if (toolCalls.length === 0) return { text, toolCalls: null };
  return { text: remaining, toolCalls };
}

function randomCallId(): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let id = "";
  for (let i = 0; i < 24; i++) id += chars[Math.floor(Math.random() * chars.length)];
  return id;
}
