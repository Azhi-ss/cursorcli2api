/**
 * Claude API via OAuth or CLI config. Ported from Python claude_oauth.py.
 */

import * as fs from "node:fs";
import { dirname, join } from "node:path";
import { fetch } from "undici";
import { settings } from "../config.js";
import { getAgent, requestJsonWithRetries } from "../lib/http-client.js";
import type { ChatCompletionRequest, ChatMessage } from "../lib/openai-compat.js";
import { normalizeMessageContent } from "../lib/openai-compat.js";

const ANTHROPIC_VERSION = "2023-06-01";
const DEFAULT_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

export interface ClaudeOAuthCreds {
  accessToken: string | null;
  refreshToken: string | null;
  expiresAtS: number | null;
  tokenType: string | null;
}

export interface ClaudeCliConfig {
  baseUrl: string | null;
  authToken: string | null;
  defaultModel: string | null;
}

function expandUser(path: string): string {
  if (path.startsWith("~")) {
    return join(process.env.HOME ?? "", path.slice(1));
  }
  return path;
}

/**
 * Load Claude CLI settings from ~/.claude/settings.json.
 */
export function loadClaudeCliSettings(): ClaudeCliConfig {
  const settingsPath = join(process.env.HOME ?? "", ".claude", "settings.json");
  try {
    const raw = JSON.parse(fs.readFileSync(settingsPath, "utf-8")) as unknown;
    if (typeof raw !== "object" || raw === null) {
      return { baseUrl: null, authToken: null, defaultModel: null };
    }
    const obj = raw as Record<string, unknown>;
    const env = (obj.env as Record<string, unknown>) ?? {};
    if (typeof env !== "object") {
      return { baseUrl: null, authToken: null, defaultModel: null };
    }
    const baseUrl = env.ANTHROPIC_BASE_URL;
    const authToken = env.ANTHROPIC_AUTH_TOKEN;
    const defaultModel = env.ANTHROPIC_DEFAULT_SONNET_MODEL;
    return {
      baseUrl: typeof baseUrl === "string" ? baseUrl : null,
      authToken: typeof authToken === "string" ? authToken : null,
      defaultModel: typeof defaultModel === "string" ? defaultModel : null,
    };
  } catch {
    return { baseUrl: null, authToken: null, defaultModel: null };
  }
}

let _cliConfig: ClaudeCliConfig | null = null;

/**
 * Get cached Claude CLI configuration.
 */
export function getClaudeCliConfig(): ClaudeCliConfig {
  if (_cliConfig === null) {
    _cliConfig = loadClaudeCliSettings();
  }
  return _cliConfig;
}

/**
 * Read OAuth creds from JSON file.
 */
export function loadCreds(path: string): ClaudeOAuthCreds {
  const p = expandUser(path);
  try {
    const raw = JSON.parse(fs.readFileSync(p, "utf-8")) as Record<string, unknown>;
    if (typeof raw !== "object" || raw === null) {
      return { accessToken: null, refreshToken: null, expiresAtS: null, tokenType: null };
    }
    const accessToken = raw.access_token;
    const refreshToken = raw.refresh_token;
    const expiresAtS = raw.expires_at_s;
    const tokenType = raw.token_type;
    return {
      accessToken: typeof accessToken === "string" ? accessToken : null,
      refreshToken: typeof refreshToken === "string" ? refreshToken : null,
      expiresAtS: typeof expiresAtS === "number" ? Math.floor(expiresAtS) : null,
      tokenType: typeof tokenType === "string" ? tokenType : null,
    };
  } catch {
    return { accessToken: null, refreshToken: null, expiresAtS: null, tokenType: null };
  }
}

/**
 * Write OAuth creds to JSON file.
 */
export function saveCreds(path: string, creds: ClaudeOAuthCreds): void {
  const p = expandUser(path);
  const dir = dirname(p);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const payload: Record<string, unknown> = {};
  if (creds.accessToken) payload.access_token = creds.accessToken;
  if (creds.refreshToken) payload.refresh_token = creds.refreshToken;
  if (creds.expiresAtS != null) payload.expires_at_s = Math.floor(creds.expiresAtS);
  if (creds.tokenType) payload.token_type = creds.tokenType;
  fs.writeFileSync(p, JSON.stringify(payload, null, 2) + "\n", "utf-8");
  try {
    fs.chmodSync(p, 0o600);
  } catch {
    /* ignore */
  }
}

function isExpired(expiresAtS: number | null, skewS = 90): boolean {
  if (!expiresAtS) return true;
  return expiresAtS <= Math.floor(Date.now() / 1000) + skewS;
}

async function refreshAccessToken(
  refreshToken: string,
  oauthClientId: string,
  baseUrl: string,
  timeoutS: number
): Promise<ClaudeOAuthCreds> {
  const url = `${baseUrl.replace(/\/$/, "")}/v1/oauth/token`;
  const payload = {
    client_id: oauthClientId,
    grant_type: "refresh_token",
    refresh_token: refreshToken,
  };
  const agent = getAgent("claude-oauth");
  const resp = await requestJsonWithRetries({
    url,
    method: "POST",
    timeoutMs: timeoutS * 1000,
    body: payload,
    headers: { Accept: "application/json" },
    agent,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Claude OAuth refresh failed: ${resp.status} ${text}`);
  }
  const data = (await resp.json()) as Record<string, unknown>;
  if (typeof data !== "object" || data === null) {
    throw new Error("Claude OAuth refresh: invalid JSON response");
  }
  const accessToken = data.access_token;
  const newRefresh = data.refresh_token ?? refreshToken;
  const expiresIn = data.expires_in;
  const tokenType = (data.token_type as string) ?? "Bearer";
  if (typeof accessToken !== "string" || !accessToken) {
    throw new Error("Claude OAuth refresh: missing access_token");
  }
  let expiresAtS: number | null = null;
  if (typeof expiresIn === "number" && expiresIn > 0) {
    expiresAtS = Math.floor(Date.now() / 1000) + Math.floor(expiresIn);
  }
  return {
    accessToken,
    refreshToken: String(newRefresh),
    expiresAtS,
    tokenType: String(tokenType),
  };
}

/**
 * Refresh OAuth token if expired. Returns updated creds.
 */
export async function maybeRefreshClaudeOauth(credsPath: string): Promise<ClaudeOAuthCreds> {
  const creds = loadCreds(credsPath);
  if (creds.accessToken && !isExpired(creds.expiresAtS)) {
    return creds;
  }
  if (!creds.refreshToken) {
    return creds;
  }
  const oauthClientId = settings.claude_oauth_client_id || DEFAULT_OAUTH_CLIENT_ID;
  const baseUrl = settings.claude_oauth_base_url;
  const refreshed = await refreshAccessToken(
    creds.refreshToken,
    oauthClientId,
    baseUrl,
    settings.timeout_seconds
  );
  saveCreds(credsPath, refreshed);
  return refreshed;
}

function parseDataUrl(dataUrl: string): [string, string] | null {
  if (!dataUrl.startsWith("data:")) return null;
  const commaIdx = dataUrl.indexOf(",");
  if (commaIdx < 0) return null;
  const header = dataUrl.slice(0, commaIdx);
  const b64 = dataUrl.slice(commaIdx + 1);
  if (!b64 || !header.includes(";base64")) return null;
  const mime = header.slice(5).split(";")[0]?.trim() || "application/octet-stream";
  return [mime, b64];
}

function contentToAnthropicBlocks(content: unknown): Array<Record<string, unknown>> {
  if (typeof content === "string") {
    const text = content.trim();
    return text ? [{ type: "text", text }] : [];
  }
  if (!Array.isArray(content)) return [];
  const blocks: Array<Record<string, unknown>> = [];
  for (const item of content) {
    if (typeof item !== "object" || item === null) continue;
    const obj = item as Record<string, unknown>;
    const t = obj.type;
    if (t === "text") {
      const text = obj.text;
      if (typeof text === "string" && text.trim()) {
        blocks.push({ type: "text", text });
      }
    } else if (t === "image_url") {
      const imageUrl = obj.image_url;
      let url: string | undefined;
      if (typeof imageUrl === "object" && imageUrl !== null && !Array.isArray(imageUrl)) {
        url = (imageUrl as Record<string, unknown>).url as string;
      }
      if (typeof url !== "string") continue;
      const parsed = parseDataUrl(url);
      if (!parsed) continue;
      const [mime, b64] = parsed;
      if (b64.length > settings.max_image_bytes * 2) continue;
      blocks.push({
        type: "image",
        source: { type: "base64", media_type: mime, data: b64 },
      });
    }
  }
  return blocks;
}

/**
 * Convert OpenAI messages to Anthropic format. Returns [system, messages].
 */
export function openaiMessagesToAnthropic(
  req: ChatCompletionRequest
): [string | null, Array<Record<string, unknown>>] {
  const systemParts: string[] = [];
  const out: Array<Record<string, unknown>> = [];
  const toolCallIdMap: Record<string, string> = {};

  for (const msg of req.messages) {
    const role = ((msg as ChatMessage).role ?? "").trim();
    const blocks = contentToAnthropicBlocks((msg as ChatMessage).content);
    const extra = (msg as Record<string, unknown>).model_extra as Record<string, unknown> | undefined;
    const toolCalls = extra?.tool_calls;

    if (role === "system") {
      for (const b of blocks) {
        if (b.type === "text") systemParts.push(String(b.text ?? ""));
      }
      continue;
    }

    if (role === "tool") {
      let toolCallId = extra?.tool_call_id ?? (msg as Record<string, unknown>).tool_call_id;
      if (typeof toolCallId !== "string" || !toolCallId) continue;
      const content = normalizeMessageContent((msg as ChatMessage).content);
      out.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: toolCallId, content }],
      });
      continue;
    }

    if (role !== "user" && role !== "assistant") continue;

    if (role === "assistant" && Array.isArray(toolCalls)) {
      const toolBlocks: Array<Record<string, unknown>> = [];
      for (const call of toolCalls) {
        if (typeof call !== "object" || call === null) continue;
        const c = call as Record<string, unknown>;
        let callId = c.id ?? c.tool_call_id;
        if (typeof callId !== "string" || !callId) {
          callId = `toolu_${toolBlocks.length + 1}`;
        }
        const func = c.function as Record<string, unknown> | undefined;
        let name = func?.name;
        let args = func?.arguments;
        if (typeof name !== "string" || !name) {
          name = (typeof c.name === "string" ? c.name : null) ?? "tool";
        }
        let parsedArgs: Record<string, unknown> = {};
        if (typeof args === "string") {
          try {
            const parsed = JSON.parse(args) as unknown;
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
              parsedArgs = parsed as Record<string, unknown>;
            }
          } catch {
            parsedArgs = {};
          }
        } else if (typeof args === "object" && args !== null) {
          parsedArgs = args as Record<string, unknown>;
        }
        toolBlocks.push({ type: "tool_use", id: callId, name, input: parsedArgs });
        toolCallIdMap[String(callId)] = String(name);
      }
      blocks.push(...toolBlocks);
    }

    if (blocks.length === 0) continue;
    out.push({ role, content: blocks });
  }

  const system = systemParts
    .map((s) => s.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim() || null;
  return [system || null, out];
}

/**
 * Convert OpenAI function tools to Anthropic format.
 */
export function openaiToolsToAnthropic(
  tools: Array<Record<string, unknown>>
): Array<Record<string, unknown>> {
  const out: Array<Record<string, unknown>> = [];
  for (const tool of tools) {
    if (typeof tool !== "object" || tool === null) continue;
    if (tool.type !== "function") continue;
    const func = tool.function as Record<string, unknown> | undefined;
    if (typeof func !== "object" || func === null) continue;
    const name = func.name;
    if (typeof name !== "string" || !name) continue;
    const entry: Record<string, unknown> = { name };
    const desc = func.description;
    if (typeof desc === "string" && desc) entry.description = desc;
    const params = func.parameters;
    if (typeof params === "object" && params !== null) entry.input_schema = params;
    out.push(entry);
  }
  return out;
}

/**
 * Convert OpenAI tool_choice to Anthropic format.
 */
export function openaiToolChoiceToAnthropic(
  choice: unknown
): Record<string, unknown> | null {
  if (choice === null || choice === undefined) return null;
  if (typeof choice === "string") {
    const lowered = choice.trim().toLowerCase();
    if (lowered === "auto" || lowered === "") return null;
    if (lowered === "required" || lowered === "any") return { type: "any" };
    if (lowered === "none") return null;
    return null;
  }
  if (typeof choice === "object" && choice !== null) {
    const obj = choice as Record<string, unknown>;
    if (obj.type === "function") {
      const fn = obj.function as Record<string, unknown> | undefined;
      if (typeof fn === "object" && fn !== null && typeof fn.name === "string" && fn.name) {
        return { type: "tool", name: fn.name };
      }
    }
  }
  return null;
}

/**
 * Add tools and tool_choice from request to Anthropic payload.
 */
export function applyOpenaiTools(
  payload: Record<string, unknown>,
  req: ChatCompletionRequest
): void {
  const extra = (req as Record<string, unknown>).model_extra as Record<string, unknown> | undefined;
  if (typeof extra !== "object" || extra === null) return;
  const tools = extra.tools;
  const toolChoice = extra.tool_choice;
  if (toolChoice === "none") return;
  if (Array.isArray(tools) && tools.length > 0) {
    const converted = openaiToolsToAnthropic(tools as Array<Record<string, unknown>>);
    if (converted.length > 0) {
      (payload as Record<string, unknown>).tools = converted;
    }
  }
  const mappedChoice = openaiToolChoiceToAnthropic(toolChoice);
  if (mappedChoice !== null) {
    (payload as Record<string, unknown>).tool_choice = mappedChoice;
  }
}

function extractTextFromAnthropicResponse(data: unknown): string {
  if (typeof data !== "object" || data === null) return "";
  const obj = data as Record<string, unknown>;
  const content = obj.content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const item of content) {
      if (typeof item === "object" && item !== null && (item as Record<string, unknown>).type === "text") {
        const t = (item as Record<string, unknown>).text;
        if (typeof t === "string") parts.push(t);
      }
    }
    return parts.join("");
  }
  return "";
}

function normalizeAnthropicToolArgs(input: unknown): string {
  if (typeof input === "string") return input;
  try {
    return JSON.stringify(input ?? {});
  } catch {
    return "{}";
  }
}

export function extractToolCallsFromAnthropicResponse(data: unknown): Record<string, unknown>[] | null {
  if (typeof data !== "object" || data === null) return null;
  const obj = data as Record<string, unknown>;
  const content = obj.content;
  if (!Array.isArray(content)) return null;

  const toolCalls: Record<string, unknown>[] = [];
  for (const item of content) {
    if (typeof item !== "object" || item === null) continue;
    const block = item as Record<string, unknown>;
    if (block.type !== "tool_use") continue;
    const id = typeof block.id === "string" && block.id ? block.id : `call_${toolCalls.length + 1}`;
    const name = typeof block.name === "string" && block.name ? block.name : "tool";
    toolCalls.push({
      id,
      type: "function",
      function: {
        name,
        arguments: normalizeAnthropicToolArgs(block.input),
      },
    });
  }

  return toolCalls.length > 0 ? toolCalls : null;
}

function extractUsageFromAnthropicResponse(data: unknown): Record<string, number> | null {
  if (typeof data !== "object" || data === null) return null;
  const obj = data as Record<string, unknown>;
  const usage = obj.usage;
  if (typeof usage !== "object" || usage === null) return null;
  const u = usage as Record<string, unknown>;
  const inTokens = Math.floor(Number(u.input_tokens) || 0);
  const outTokens = Math.floor(Number(u.output_tokens) || 0);
  return {
    prompt_tokens: inTokens,
    completion_tokens: outTokens,
    total_tokens: inTokens + outTokens,
  };
}

function extractDeltaText(obj: unknown): string {
  if (typeof obj !== "object" || obj === null) return "";
  const o = obj as Record<string, unknown>;
  const delta = o.delta;
  if (typeof delta === "object" && delta !== null) {
    const d = delta as Record<string, unknown>;
    const t = d.text;
    if (typeof t === "string" && t) return t;
  }
  const t2 = o.text;
  if (typeof t2 === "string" && t2) return t2;
  const contentBlock = o.content_block;
  if (typeof contentBlock === "object" && contentBlock !== null) {
    const cb = contentBlock as Record<string, unknown>;
    const t3 = cb.text;
    if (typeof t3 === "string" && t3) return t3;
  }
  const message = o.message;
  if (typeof message === "object" && message !== null) {
    return extractTextFromAnthropicResponse(message);
  }
  return "";
}

function extractStreamUsage(obj: unknown): Record<string, number> | null {
  if (typeof obj !== "object" || obj === null) return null;
  const o = obj as Record<string, unknown>;
  if ("usage" in o) return extractUsageFromAnthropicResponse(o);
  const msg = o.message;
  if (typeof msg === "object" && msg !== null) return extractUsageFromAnthropicResponse(msg);
  return null;
}

/**
 * Non-streaming Claude API call via OAuth or CLI config.
 */
export async function generateOauth(
  req: ChatCompletionRequest,
  modelName: string
): Promise<[string, Record<string, number> | null, Record<string, unknown>[] | null]> {
  const cliConfig = getClaudeCliConfig();
  let authToken: string;
  let baseUrl: string;

  if (cliConfig.authToken && cliConfig.baseUrl) {
    authToken = cliConfig.authToken;
    baseUrl = cliConfig.baseUrl;
    if (["sonnet", "opus", "haiku"].includes(modelName) && cliConfig.defaultModel) {
      modelName = cliConfig.defaultModel;
    }
  } else {
    const creds = await maybeRefreshClaudeOauth(settings.claude_oauth_creds_path);
    if (!creds.accessToken) {
      throw new Error(
        "Claude OAuth: missing access_token (set up OAuth credentials first)"
      );
    }
    authToken = creds.accessToken;
    baseUrl = settings.claude_api_base_url;
  }

  const [system, messages] = openaiMessagesToAnthropic(req);
  const maxTokens = req.max_tokens ?? 8192;
  const payload: Record<string, unknown> = {
    model: modelName,
    max_tokens: maxTokens,
    messages,
  };
  if (system) payload.system = system;
  applyOpenaiTools(payload, req);

  const url = `${baseUrl.replace(/\/$/, "")}/v1/messages`;
  const agent = getAgent("claude");
  const resp = await requestJsonWithRetries({
    url,
    method: "POST",
    timeoutMs: settings.timeout_seconds * 1000,
    body: payload,
    headers: {
      Authorization: `Bearer ${authToken}`,
      "anthropic-version": ANTHROPIC_VERSION,
      Accept: "application/json",
    },
    agent,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Claude API error: ${resp.status} ${text}`);
  }

  const data = (await resp.json()) as unknown;
  return [
    extractTextFromAnthropicResponse(data),
    extractUsageFromAnthropicResponse(data),
    extractToolCallsFromAnthropicResponse(data),
  ];
}

async function* iterSseEvents(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<[string | null, string]> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event: string | null = null;
  let dataLines: string[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith(":")) continue;
        if (!line.trim()) {
          if (dataLines.length > 0) {
            yield [event, dataLines.join("\n")];
          }
          event = null;
          dataLines = [];
          continue;
        }
        if (line.startsWith("event:")) {
          event = line.slice(6).trim() || null;
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
          continue;
        }
      }
    }
    if (dataLines.length > 0) {
      yield [event, dataLines.join("\n")];
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Streaming Claude API call. Yields {type:"assistant", message:{role,content}} and {type:"result", usage}.
 */
export async function* iterOauthStreamEvents(
  req: ChatCompletionRequest,
  modelName: string
): AsyncGenerator<Record<string, unknown>> {
  const cliConfig = getClaudeCliConfig();
  let authToken: string;
  let baseUrl: string;

  if (cliConfig.authToken && cliConfig.baseUrl) {
    authToken = cliConfig.authToken;
    baseUrl = cliConfig.baseUrl;
    if (["sonnet", "opus", "haiku"].includes(modelName) && cliConfig.defaultModel) {
      modelName = cliConfig.defaultModel;
    }
  } else {
    const creds = await maybeRefreshClaudeOauth(settings.claude_oauth_creds_path);
    if (!creds.accessToken) {
      throw new Error(
        "Claude OAuth: missing access_token (set up OAuth credentials first)"
      );
    }
    authToken = creds.accessToken;
    baseUrl = settings.claude_api_base_url;
  }

  const [system, messages] = openaiMessagesToAnthropic(req);
  const maxTokens = req.max_tokens ?? 8192;
  const payload: Record<string, unknown> = {
    model: modelName,
    max_tokens: maxTokens,
    messages,
    stream: true,
  };
  if (system) payload.system = system;
  applyOpenaiTools(payload, req);

  const url = `${baseUrl.replace(/\/$/, "")}/v1/messages`;
  const agent = getAgent("claude-stream");
  const resp = await fetch(url, {
    method: "POST",
    body: JSON.stringify(payload),
    headers: {
      Authorization: `Bearer ${authToken}`,
      "anthropic-version": ANTHROPIC_VERSION,
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    dispatcher: agent,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Claude API error: ${resp.status} ${text}`);
  }

  const body = resp.body;
  if (!body) throw new Error("No response body");

  let usage: Record<string, number> | null = null;
  const toolCallsByIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for await (const [, data] of iterSseEvents(body)) {
    if (!data || data.trim() === "[DONE]") continue;
    try {
      const obj = JSON.parse(data) as unknown;
      if (typeof obj === "object" && obj !== null) {
        const eventObj = obj as Record<string, unknown>;
        const eventType = eventObj.type;
        const index = typeof eventObj.index === "number" ? eventObj.index : null;
        if (eventType === "content_block_start" && index !== null) {
          const block = eventObj.content_block;
          if (typeof block === "object" && block !== null) {
            const contentBlock = block as Record<string, unknown>;
            if (contentBlock.type === "tool_use") {
              const id =
                typeof contentBlock.id === "string" && contentBlock.id
                  ? contentBlock.id
                  : `call_${toolCallsByIndex.size + 1}`;
              const name =
                typeof contentBlock.name === "string" && contentBlock.name
                  ? contentBlock.name
                  : "tool";
              toolCallsByIndex.set(index, { id, name, arguments: normalizeAnthropicToolArgs(contentBlock.input) });
            }
          }
        } else if (eventType === "content_block_delta" && index !== null) {
          const deltaObj = eventObj.delta;
          const current = toolCallsByIndex.get(index);
          if (current && typeof deltaObj === "object" && deltaObj !== null) {
            const delta = deltaObj as Record<string, unknown>;
            if (delta.type === "input_json_delta" && typeof delta.partial_json === "string") {
              current.arguments += delta.partial_json;
            }
          }
        }
      }
      const delta = extractDeltaText(obj);
      if (delta) {
        yield {
          type: "assistant",
          message: { role: "assistant", content: [{ type: "text", text: delta }] },
        };
      }
      const maybeUsage = extractStreamUsage(obj);
      if (maybeUsage) usage = maybeUsage;
    } catch {
      /* ignore parse errors */
    }
  }

  const toolCalls = Array.from(toolCallsByIndex.values()).map((call) => ({
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: call.arguments || "{}",
    },
  }));
  if (usage || toolCalls.length > 0) {
    yield {
      type: "result",
      usage: {
        input_tokens: usage?.prompt_tokens ?? 0,
        output_tokens: usage?.completion_tokens ?? 0,
      },
      ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
    };
  }
}
