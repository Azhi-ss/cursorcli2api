/**
 * Main Hono server. Ported from Python FastAPI server.py.
 */

import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { randomUUID } from "crypto";

/**
 * Build cursor-agent command. For long prompts (>100KB), passes prompt
 * via stdin to avoid OS argument length limits (E2BIG).
 */
function buildCursorAgentCmd(baseArgs: string[], prompt: string): { cmd: string[]; stdinData: string | null } {
  const MAX_ARG_LEN = 100 * 1024;
  if (prompt.length < MAX_ARG_LEN) {
    return { cmd: [...baseArgs, prompt], stdinData: null };
  }
  return { cmd: baseArgs, stdinData: prompt };
}
import pino from "pino";
import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Context } from "hono";

import { settings } from "./config.js";
import type { ChatCompletionRequest, ChatMessage } from "./lib/openai-compat.js";
import {
  compatChatRequestToChatRequest,
  extractImageUrls,
  messagesToPrompt,
  normalizeMessageContent,
  responsesRequestToChatRequest,
  buildToolCallSystemPrompt,
  formatToolResultMessages,
  parseToolCallResponse,
} from "./lib/openai-compat.js";
import { ResponsesRequestSchema } from "./lib/openai-compat.js";
import { ChatCompletionRequestCompatSchema } from "./lib/openai-compat.js";
import {
  AnthropicMessagesRequestSchema,
  anthropicRequestToChatRequest,
  chatCompletionToAnthropicResponse,
  anthropicStreamMessageStart,
  anthropicStreamContentBlockStart,
  anthropicStreamContentBlockDelta,
  anthropicStreamContentBlockStop,
  anthropicStreamMessageDelta,
  anthropicStreamMessageStop,
  anthropicStreamPing,
  anthropicErrorResponse,
} from "./lib/anthropic-compat.js";
import { closeAll } from "./lib/http-client.js";
import { iterCodexEvents, collectCodexTextAndUsageFromEvents } from "./providers/codex-cli.js";
import {
  loadCodexAuth,
  maybeRefreshCodexAuth,
  warmupCodexAuth,
  buildCodexHeaders,
  extractCodexUsageHeaders,
  extractCodexToolCalls,
  convertChatCompletionsToCodexResponses,
  iterCodexResponsesEvents,
  collectCodexResponsesTextAndUsage,
} from "./providers/codex-responses.js";
import { generateOauth as claudeOauthGenerate, iterOauthStreamEvents as iterClaudeOauthEvents } from "./providers/claude-oauth.js";
import {
  generateCloudcode as geminiCloudcodeGenerate,
  iterCloudcodeStreamEvents as iterGeminiCloudcodeEvents,
  warmupGeminiCaches,
} from "./providers/gemini-cloudcode.js";
import {
  TextAssembler,
  iterStreamJsonEvents,
  extractCursorAgentDelta,
  extractClaudeDelta,
  extractGeminiDelta,
  extractUsageFromClaudeResult,
  extractUsageFromGeminiResult,
} from "./providers/stream-json-cli.js";

// ─────────────────────────────────────────────────────────────────────────────
// Logger
// ─────────────────────────────────────────────────────────────────────────────

const logger = pino({ level: process.env.LOG_LEVEL ?? "info" });

// ─────────────────────────────────────────────────────────────────────────────
// Hono app
// ─────────────────────────────────────────────────────────────────────────────

export const app = new Hono();

if (settings.cors_origins.trim()) {
  const origins = settings.cors_origins.split(",").map((o) => o.trim()).filter(Boolean);
  if (origins.length > 0) {
    app.use("/*", cors({ origin: origins, credentials: true, allowMethods: ["*"], allowHeaders: ["*"] }));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrency
// ─────────────────────────────────────────────────────────────────────────────

let activeRequests = 0;
const semaphore = { current: 0, max: settings.max_concurrency, waiters: [] as (() => void)[] };

async function acquireSemaphore(): Promise<void> {
  if (semaphore.current < semaphore.max) {
    semaphore.current++;
    return;
  }
  await new Promise<void>((resolve) => {
    semaphore.waiters.push(resolve);
  });
  semaphore.current++;
}

function releaseSemaphore(): void {
  semaphore.current--;
  if (semaphore.waiters.length > 0) {
    const next = semaphore.waiters.shift();
    if (next) next();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function splitOnFirstColon(s: string): [string, string] {
  const idx = s.indexOf(":");
  if (idx === -1) return [s, ""];
  return [s.slice(0, idx), s.slice(idx + 1)];
}

function parseProviderModel(model: string): [string, string | null] {
  const raw = (model ?? "").trim();
  if (!raw) return ["codex", null];

  for (const prefix of ["cursor-agent:", "cursor:"]) {
    if (raw.startsWith(prefix)) {
      const inner = splitOnFirstColon(raw)[1].trim();
      return ["cursor-agent", inner || null];
    }
  }
  if (raw === "cursor-agent" || raw === "cursor") return ["cursor-agent", null];

  for (const prefix of ["claude-code:", "claude:"]) {
    if (raw.startsWith(prefix)) {
      const inner = splitOnFirstColon(raw)[1].trim();
      return ["claude", inner || null];
    }
  }
  if (raw === "claude-code" || raw === "claude") return ["claude", null];

  if (raw.startsWith("gemini:")) {
    const inner = splitOnFirstColon(raw)[1].trim();
    return ["gemini", inner || null];
  }
  if (raw === "gemini") return ["gemini", null];

  return ["codex", raw];
}

function normalizeProvider(raw: string | null): string {
  const p = (raw ?? "").trim().toLowerCase();
  if (!p) return "auto";
  if (["auto", "codex", "cursor-agent", "claude", "gemini"].includes(p)) return p;
  if (["cursor", "cursor_agent", "cursoragent"].includes(p)) return "cursor-agent";
  if (["claude-code", "claude_code", "claudecode"].includes(p)) return "claude";
  return "auto";
}

function providerDefaultModel(provider: string): string | null {
  if (provider === "codex") return settings.default_model;
  if (provider === "cursor-agent") return settings.cursor_agent_model ?? "auto";
  if (provider === "claude") return settings.claude_model ?? "sonnet";
  if (provider === "gemini") return settings.gemini_model ?? "gemini-3-flash-preview";
  return null;
}

function maybeStripAnswerTags(text: string): string {
  if (!settings.strip_answer_tags || !text) return text;
  for (const tag of ["<think>", "</think>", "<answer>", "</answer>"]) {
    text = text.replaceAll(tag, "");
  }
  return text;
}

function ensureWorkspaceDir(dirPath: string | null | undefined): void {
  const resolved = (dirPath ?? "").trim();
  if (!resolved) return;
  if (existsSync(resolved)) {
    const stat = statSync(resolved);
    if (!stat.isDirectory()) {
      throw new Error(`Workspace path is not a directory: ${resolved}`);
    }
    return;
  }
  mkdirSync(resolved, { recursive: true });
}

function openaiError(message: string, statusCode = 500): Response {
  return new Response(
    JSON.stringify({
      error: {
        message,
        type: "codex_gateway_error",
        param: null,
        code: null,
      },
    }),
    { status: statusCode, headers: { "Content-Type": "application/json" } }
  );
}

const UPSTREAM_STATUS_RE = /(?:\bAPI Error:\s*|\bfailed:\s*)(\d{3})\b/;
const HTTPX_STATUS_RE = /\b(?:Client|Server) error '(\d{3})\b/;
const GENERIC_STATUS_RE = /\bstatus\s*[=:]\s*(\d{3})\b/;

function extractUpstreamStatusCode(err: unknown): number | null {
  const msg = String(err ?? "").trim();
  if (!msg) return null;
  for (const rx of [UPSTREAM_STATUS_RE, HTTPX_STATUS_RE, GENERIC_STATUS_RE]) {
    const m = rx.exec(msg);
    if (m) {
      const code = parseInt(m[1], 10);
      if (code >= 400 && code <= 599) return code;
    }
  }
  return null;
}

function truncateForLog(text: string): string {
  const limit = settings.log_max_chars;
  if (limit <= 0) return "";
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}\n... (truncated, ${text.length} chars total)`;
}

function checkAuth(authorization: string | null | undefined): void {
  const token = settings.bearer_token;
  if (!token) return;
  if (!authorization || !authorization.startsWith("Bearer ")) {
    throw new Error("Missing Authorization: Bearer <token>");
  }
  const provided = authorization.replace(/^Bearer\s+/, "").trim();
  if (provided !== token) {
    throw new Error("Invalid token");
  }
}

function chatCompletionToResponses(chat: Record<string, unknown>): Record<string, unknown> {
  const created = Number(chat.created) || Math.floor(Date.now() / 1000);
  const model = chat.model;
  let text = "";
  const choices = (chat.choices as unknown[]) ?? [];
  if (choices.length > 0) {
    const first = choices[0] as Record<string, unknown>;
    const message = (first.message as Record<string, unknown>) ?? {};
    text = normalizeMessageContent(message.content);
  }

  let usageOut: Record<string, number> | undefined;
  const usage = chat.usage as Record<string, unknown> | undefined;
  if (usage && typeof usage === "object") {
    const promptTokens = Math.floor(Number(usage.prompt_tokens) || 0);
    const completionTokens = Math.floor(Number(usage.completion_tokens) || 0);
    usageOut = {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      total_tokens: Math.floor(Number(usage.total_tokens) || promptTokens + completionTokens),
    };
  }

  const outputMsg = {
    id: `msg_${randomUUID().slice(0, 24)}`,
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text }],
  };
  const resp: Record<string, unknown> = {
    id: `resp_${randomUUID().slice(0, 24)}`,
    object: "response",
    created,
    model,
    output: [outputMsg],
  };
  if (usageOut) resp.usage = usageOut;
  return resp;
}

function looksLikeUnsupportedModelError(message: string): boolean {
  const msg = (message ?? "").trim();
  if (!msg) return false;
  let detail = msg;
  try {
    const obj = JSON.parse(msg) as Record<string, unknown>;
    if (typeof obj?.detail === "string") detail = obj.detail;
  } catch {
    /* ignore */
  }
  const lowered = detail.toLowerCase();
  return lowered.includes("model is not supported") || lowered.includes("not supported when using codex");
}

function mimeToExt(mime: string): string {
  const m = (mime ?? "").trim().toLowerCase();
  if (["image/png", "png"].includes(m)) return "png";
  if (["image/jpeg", "image/jpg", "jpeg", "jpg"].includes(m)) return "jpg";
  if (["image/webp", "webp"].includes(m)) return "webp";
  return "bin";
}

function decodeDataUrl(dataUrl: string): [Buffer, string] {
  if (!dataUrl.startsWith("data:")) throw new Error("Unsupported image_url (expected data: URL)");
  const commaIdx = dataUrl.indexOf(",");
  if (commaIdx < 0) throw new Error("Invalid data: URL");
  const header = dataUrl.slice(0, commaIdx);
  const payload = dataUrl.slice(commaIdx + 1).replace(/\s/g, "");
  if (!header.includes(";base64")) throw new Error("Unsupported data: URL encoding (expected base64)");
  const mime = header.replace(/^data:/, "").split(";")[0]?.trim() || "application/octet-stream";
  const data = Buffer.from(payload, "base64");
  return [data, mimeToExt(mime)];
}

function materializeRequestImages(
  messages: ChatMessage[],
  respId: string
): { tmpdir: string; paths: string[] } | null {
  if (!settings.enable_image_input) return null;
  const urls = extractImageUrls(messages);
  if (urls.length === 0) return null;
  const maxCount = Math.max(settings.max_image_count, 0);
  if (maxCount === 0) return null;
  const limited = urls.slice(-maxCount);
  const dir = mkdtempSync(join(tmpdir(), "codex-gateway-images-"));
  const paths: string[] = [];
  for (let i = 0; i < limited.length; i++) {
    const [data, ext] = decodeDataUrl(limited[i]);
    if (settings.max_image_bytes > 0 && data.length > settings.max_image_bytes) {
      throw new Error(`Image too large (${data.length} bytes > ${settings.max_image_bytes})`);
    }
    const filePath = join(dir, `${respId}-${i}.${ext}`);
    writeFileSync(filePath, data);
    paths.push(filePath);
  }
  return { tmpdir: dir, paths };
}

function extractCodexSessionId(req: Record<string, unknown>, headers: Headers): string | null {
  const headerNames = ["x-codex-session-id", "x-session-id", "session-id", "session_id"];
  for (const name of headerNames) {
    const value = headers.get(name);
    if (typeof value === "string" && value.trim()) return value.trim().slice(0, 128);
  }
  const extra = (req.model_extra as Record<string, unknown>) ?? {};
  for (const key of ["session_id", "sessionId", "codex_session_id", "codexSessionId"]) {
    const value = extra[key];
    if (typeof value === "string" && value.trim()) return value.trim().slice(0, 128);
  }
  return null;
}

function extractReasoningEffort(req: ChatCompletionRequest): string | null {
  const extra = (req as Record<string, unknown>).model_extra as Record<string, unknown> | undefined;
  if (!extra || typeof extra !== "object") return null;
  const direct = extra.reasoning_effort;
  if (typeof direct === "string" && direct.trim()) return direct.trim();
  const reasoning = extra.reasoning;
  if (reasoning && typeof reasoning === "object") {
    const effort = (reasoning as Record<string, unknown>).effort;
    if (typeof effort === "string" && effort.trim()) return effort.trim();
  }
  return null;
}

const AUTOMATION_GUARD = `SYSTEM: IMPORTANT (Open-AutoGLM action mode)
- This is an Android UI automation loop. You will be given a screenshot each turn.
- The screenshot is already attached as an image in the user's message; use it directly (native vision).
- Output must be machine-parseable and contain exactly ONE action.
- Follow the required format exactly:
  <think>short reason</think>
  <answer>do(...)</answer> OR <answer>finish(...)</answer>
- The <answer> must contain ONLY the action expression (no extra text).
- Do NOT call MCP/tools to fetch screenshots/resources (e.g., list_mcp_resources/read_mcp_resource); they are unavailable here.
- Never refuse due to environment/sandbox/tool limitations; you are NOT executing actions yourself.
- If you are uncertain or need confirmation, output: do(action="Wait", duration="1 seconds")
- For WeChat message sending: only finish if the latest screenshot clearly shows the outgoing message bubble
  with the exact text AND no failure indicator (red "!", "发送失败/未发送/重发", spinner/clock). Otherwise retry/Wait.
`;

function looksLikeAutomationPrompt(prompt: string): boolean {
  const p = prompt ?? "";
  const markers = [
    "<think>{think}</think>",
    "<answer>{action}</answer>",
    'do(action="Tap"',
    'do(action="Launch"',
    "finish(message=",
    "Tap是点击操作",
    "finish是结束任务",
  ];
  return markers.some((m) => p.includes(m));
}

function maybeInjectAutomationGuard(prompt: string): string {
  if (!prompt || !looksLikeAutomationPrompt(prompt) || prompt.includes("IMPORTANT (Open-AutoGLM action mode)")) {
    return prompt;
  }
  return `${AUTOMATION_GUARD}\n\n${prompt}`;
}

function maybeInjectAutomationGuardMessages(messages: ChatMessage[]): ChatMessage[] {
  if (!messages.length) return messages;
  const prompt = messagesToPrompt(messages);
  if (!looksLikeAutomationPrompt(prompt) || prompt.includes("IMPORTANT (Open-AutoGLM action mode)")) {
    return messages;
  }
  return [{ role: "system", content: AUTOMATION_GUARD } as ChatMessage, ...messages];
}

// ─────────────────────────────────────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────────────────────────────────────

app.get("/healthz", (c) => c.json({ ok: true }));

app.get("/v1/models", async (c) => {
  try {
    checkAuth(c.req.header("authorization"));
  } catch (e) {
    return c.json({ error: { message: String(e) } }, 401);
  }
  const forcedProvider = normalizeProvider(settings.provider);
  const defaultId = providerDefaultModel(forcedProvider) ?? settings.default_model;
  let models: string[];
  if (settings.advertised_models.length > 0) {
    models = [...settings.advertised_models];
  } else if (forcedProvider !== "auto" && !settings.allow_client_model_override) {
    models = ["default", defaultId];
  } else {
    models = [defaultId];
  }
  if (Object.keys(settings.model_aliases).length > 0) {
    models.push(...Object.keys(settings.model_aliases), ...Object.values(settings.model_aliases));
  }
  const seen = new Set<string>();
  const unique = models.filter((m) => m && !seen.has(m) && (seen.add(m), true));
  return c.json({
    object: "list",
    data: unique.map((m) => ({ id: m, object: "model", created: 0, owned_by: "local" })),
  });
});

app.get("/models", async (c) => {
  try {
    checkAuth(c.req.header("authorization"));
  } catch (e) {
    return c.json({ error: { message: String(e) } }, 401);
  }
  const forcedProvider = normalizeProvider(settings.provider);
  const defaultId = providerDefaultModel(forcedProvider) ?? settings.default_model;
  let models: string[];
  if (settings.advertised_models.length > 0) {
    models = [...settings.advertised_models];
  } else if (forcedProvider !== "auto" && !settings.allow_client_model_override) {
    models = ["default", defaultId];
  } else {
    models = [defaultId];
  }
  if (Object.keys(settings.model_aliases).length > 0) {
    models.push(...Object.keys(settings.model_aliases), ...Object.values(settings.model_aliases));
  }
  const seen = new Set<string>();
  const unique = models.filter((m) => m && !seen.has(m) && (seen.add(m), true));
  return c.json({
    object: "list",
    data: unique.map((m) => ({ id: m, object: "model", created: 0, owned_by: "local" })),
  });
});

app.get("/debug/config", async (c) => {
  try {
    checkAuth(c.req.header("authorization"));
  } catch (e) {
    return c.json({ error: { message: String(e) } }, 401);
  }
  return c.json({
    provider: settings.provider,
    allow_client_provider_override: settings.allow_client_provider_override,
    allow_client_model_override: settings.allow_client_model_override,
    default_model: settings.default_model,
    cursor_agent_model: settings.cursor_agent_model ?? "auto",
    cursor_agent_workspace: settings.cursor_agent_workspace,
    cursor_agent_disable_indexing: settings.cursor_agent_disable_indexing,
    cursor_agent_extra_args: settings.cursor_agent_extra_args,
    claude_model: settings.claude_model,
    claude_use_oauth_api: settings.claude_use_oauth_api,
    claude_api_base_url: settings.claude_api_base_url,
    claude_oauth_base_url: settings.claude_oauth_base_url,
    claude_oauth_creds_path: settings.claude_oauth_creds_path,
    claude_oauth_client_id: settings.claude_oauth_client_id ? "[REDACTED]" : "",
    gemini_model: settings.gemini_model,
    gemini_use_cloudcode_api: settings.gemini_use_cloudcode_api,
    gemini_cloudcode_base_url: settings.gemini_cloudcode_base_url,
    gemini_project_id: settings.gemini_project_id,
    gemini_oauth_creds_path: settings.gemini_oauth_creds_path,
    gemini_oauth_client_id: settings.gemini_oauth_client_id ? "[REDACTED]" : "",
    model_reasoning_effort: settings.model_reasoning_effort,
    force_reasoning_effort: settings.force_reasoning_effort,
    use_codex_responses_api: settings.use_codex_responses_api,
    codex_cli_home: settings.codex_cli_home,
    workspace: settings.workspace,
    max_concurrency: settings.max_concurrency,
    timeout_seconds: settings.timeout_seconds,
    subprocess_stream_limit: settings.subprocess_stream_limit,
    sse_keepalive_seconds: settings.sse_keepalive_seconds,
    strip_answer_tags: settings.strip_answer_tags,
    enable_image_input: settings.enable_image_input,
    max_image_count: settings.max_image_count,
    max_image_bytes: settings.max_image_bytes,
    disable_shell_tool: settings.disable_shell_tool,
    disable_view_image_tool: settings.disable_view_image_tool,
    debug_log: settings.debug_log,
    log_mode: settings.effectiveLogMode(),
    log_events: settings.log_events,
    log_max_chars: settings.log_max_chars,
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Responses API
// ─────────────────────────────────────────────────────────────────────────────

app.post("/v1/responses", handleResponses);
app.post("/responses", handleResponses);

async function handleResponses(c: Context): Promise<Response> {
  try {
    checkAuth(c.req.header("authorization"));
  } catch (e) {
    return c.json({ error: { message: String(e) } }, 401);
  }
  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return openaiError("Invalid JSON body", 422);
  }
  const parsed = ResponsesRequestSchema.safeParse(body);
  if (!parsed.success) {
    return openaiError("Invalid request: " + (parsed.error.message ?? "validation failed"), 422);
  }
  const req = parsed.data;
  const chatReq = responsesRequestToChatRequest(req);
  if (!chatReq.messages.length) {
    return openaiError("Missing input for responses request", 422);
  }
  if (chatReq.stream) {
    return openaiError(
      "Streaming responses are not supported; set stream=false or use /v1/chat/completions",
      400
    );
  }
  const result = await handleChatCompletions(chatReq, c.req.header("authorization") ?? undefined, c);
  if (result instanceof Response) {
    if (result.status < 400) {
      try {
        const json = (await result.json()) as Record<string, unknown>;
        if (json?.object === "chat.completion") {
          const converted = chatCompletionToResponses(json);
          const headers = new Headers(result.headers);
          const codexHeaders = extractCodexUsageHeaders(headers);
          const outHeaders = new Headers();
          for (const [k, v] of Object.entries(codexHeaders)) outHeaders.set(k, v);
          return new Response(JSON.stringify(converted), {
            status: 200,
            headers: { ...Object.fromEntries(outHeaders), "Content-Type": "application/json" },
          });
        }
      } catch {
        /* ignore */
      }
    }
    return result;
  }
  return c.json(chatCompletionToResponses(result as Record<string, unknown>));
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat Completions API
// ─────────────────────────────────────────────────────────────────────────────

app.post("/v1/chat/completions", handleChatCompletionsRoute);
app.post("/chat/completions", handleChatCompletionsRoute);

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic Messages API
// ─────────────────────────────────────────────────────────────────────────────

app.post("/v1/messages", handleAnthropicMessages);
app.post("/messages", handleAnthropicMessages);

async function handleChatCompletionsRoute(c: Context): Promise<Response> {
  try {
    checkAuth(c.req.header("authorization"));
  } catch (e) {
    return c.json({ error: { message: String(e) } }, 401);
  }
  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return openaiError("Invalid JSON body", 422);
  }
  const parsed = ChatCompletionRequestCompatSchema.safeParse(body);
  if (!parsed.success) {
    return openaiError("Invalid request: " + (parsed.error.message ?? "validation failed"), 422);
  }
  let req: ChatCompletionRequest;
  try {
    req = compatChatRequestToChatRequest(parsed.data as ChatCompletionRequest & { input?: unknown; instructions?: string });
  } catch (e) {
    return openaiError(String(e), 422);
  }
  const result = await handleChatCompletions(req, c.req.header("authorization") ?? undefined, c);
  return result instanceof Response ? result : c.json(result);
}

async function handleAnthropicMessages(c: Context): Promise<Response> {
  const apiKey = c.req.header("x-api-key");
  const authHeader = c.req.header("authorization");
  try {
    checkAuth(apiKey ? `Bearer ${apiKey}` : authHeader);
  } catch (e) {
    return anthropicErrorResponse(String(e), 401);
  }

  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return anthropicErrorResponse("Invalid JSON body", 400);
  }

  const parsed = AnthropicMessagesRequestSchema.safeParse(body);
  if (!parsed.success) {
    return anthropicErrorResponse(
      "Invalid request: " + (parsed.error.message ?? "validation failed"),
      400
    );
  }

  const anthropicReq = parsed.data;
  const requestModel = anthropicReq.model;
  const wantsStream = anthropicReq.stream ?? false;

  const chatReq = anthropicRequestToChatRequest(anthropicReq);
  chatReq.stream = wantsStream;

  if (!wantsStream) {
    chatReq.stream = false;
    const result = await handleChatCompletions(chatReq, authHeader ?? undefined, c);
    let chatJson: Record<string, unknown>;
    if (result instanceof Response) {
      if (result.status >= 400) {
        const errBody = await result.text().catch(() => "Internal error");
        return anthropicErrorResponse(errBody, result.status);
      }
      chatJson = (await result.json()) as Record<string, unknown>;
    } else {
      chatJson = result;
    }
    const anthropicResp = chatCompletionToAnthropicResponse(chatJson, requestModel);
    return new Response(JSON.stringify(anthropicResp), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Streaming: get the SSE stream from the OpenAI-format handler, then re-encode as Anthropic SSE
  chatReq.stream = true;
  const sseResponse = await handleChatCompletions(chatReq, authHeader ?? undefined, c);

  const upstreamResponse = sseResponse instanceof Response ? sseResponse : new Response(JSON.stringify(sseResponse), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  if (upstreamResponse.status >= 400 || !upstreamResponse.body) {
    const errBody = await upstreamResponse.text().catch(() => "Internal error");
    return anthropicErrorResponse(errBody, upstreamResponse.status || 500);
  }

  const messageId = `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const anthropicStream = new ReadableStream({
    async start(controller) {
      controller.enqueue(encoder.encode(anthropicStreamMessageStart(messageId, requestModel, 0)));
      controller.enqueue(encoder.encode(anthropicStreamContentBlockStart(0)));
      controller.enqueue(encoder.encode(anthropicStreamPing()));

      const reader = upstreamResponse.body!.getReader();
      let buffer = "";
      let outputTokens = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const payload = trimmed.slice(6);
            if (payload === "[DONE]") continue;

            let chunk: Record<string, unknown>;
            try {
              chunk = JSON.parse(payload) as Record<string, unknown>;
            } catch {
              continue;
            }

            const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
            if (!choices?.length) continue;
            const choice = choices[0];
            const delta = choice.delta as Record<string, unknown> | undefined;
            const content = delta?.content;

            if (typeof content === "string" && content) {
              controller.enqueue(encoder.encode(anthropicStreamContentBlockDelta(0, content)));
              outputTokens += Math.ceil(content.length / 4);
            }

            if (choice.finish_reason) {
              break;
            }
          }
        }
      } catch (e) {
        logger.error({ err: e }, "Anthropic stream proxy error");
      }

      controller.enqueue(encoder.encode(anthropicStreamContentBlockStop(0)));
      controller.enqueue(encoder.encode(anthropicStreamMessageDelta("end_turn", outputTokens)));
      controller.enqueue(encoder.encode(anthropicStreamMessageStop()));
      controller.close();
    },
  });

  return new Response(anthropicStream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

async function handleChatCompletions(
  req: ChatCompletionRequest,
  authorization?: string,
  _c?: Context
): Promise<Response | Record<string, unknown>> {
  const forcedProvider = normalizeProvider(settings.provider);
  const fallbackModel =
    providerDefaultModel(forcedProvider !== "auto" ? forcedProvider : "codex") ?? settings.default_model;
  const clientModel = (req.model ?? "").trim();
  const clientModelIgnored = forcedProvider !== "auto" && !settings.allow_client_model_override;
  const requestedModel = (clientModelIgnored ? fallbackModel : clientModel || fallbackModel).trim();
  const resolvedModel = settings.model_aliases[requestedModel] ?? requestedModel;
  const [parsedProvider, providerModel] = parseProviderModel(resolvedModel);

  let provider: string;
  let effectiveProviderModel: string | null;
  if (settings.allow_client_provider_override || forcedProvider === "auto") {
    provider = parsedProvider;
    effectiveProviderModel = providerModel;
  } else {
    provider = forcedProvider;
    effectiveProviderModel = settings.allow_client_model_override ? providerModel : null;
  }

  const allowedEfforts = new Set(["low", "medium", "high", "xhigh"]);
  const normalizeEffort = (raw: string | null | undefined): string | null => {
    if (!raw) return null;
    if (raw === "none") return "low";
    return allowedEfforts.has(raw) ? raw : null;
  };

  const requestEffortRaw = extractReasoningEffort(req);
  const requestEffort = normalizeEffort(requestEffortRaw);
  const forcedEffort = normalizeEffort((settings.force_reasoning_effort ?? "").trim() || null);
  const defaultEffort = normalizeEffort((settings.model_reasoning_effort ?? "").trim() || null);
  const reasoningEffort = forcedEffort ?? requestEffort ?? defaultEffort ?? "high";

  const rawBody = (req as Record<string, unknown>);
  const requestTools = rawBody.tools as unknown[] | undefined;
  const toolSystemPrompt = buildToolCallSystemPrompt(requestTools);

  let messagesForPrompt = req.messages;
  if (requestTools && requestTools.length > 0) {
    messagesForPrompt = formatToolResultMessages(req.messages);
  }

  let prompt = maybeInjectAutomationGuard(messagesToPrompt(messagesForPrompt));
  if (toolSystemPrompt) {
    prompt = `SYSTEM: ${toolSystemPrompt}\n\n${prompt}`;
  }
  if (prompt.length > settings.max_prompt_chars) {
    return openaiError(`Prompt too large (${prompt.length} chars)`, 413);
  }

  const codexSessionId = _c ? extractCodexSessionId(req as Record<string, unknown>, _c.req.raw.headers) : null;
  const codexResponseHeaders: Record<string, string> = {};
  let toolCalls: Record<string, unknown>[] | null = null;

  const created = Math.floor(Date.now() / 1000);
  const respId = `chatcmpl-${randomUUID().replace(/-/g, "")}`;
  const t0 = Date.now();

  const imageUrls = extractImageUrls(req.messages);
  const useClaudeOauth = provider === "claude" && settings.claude_use_oauth_api;
  const useGeminiCloudcode = provider === "gemini" && settings.gemini_use_cloudcode_api;
  const useCodexBackend =
    provider === "codex" &&
    (settings.use_codex_responses_api ||
      (settings.enable_image_input && imageUrls.length > 0) ||
      req.stream);

  let imageMaterialized: { tmpdir: string; paths: string[] } | null = null;
  if (provider === "codex" && !useCodexBackend && settings.enable_image_input && imageUrls.length > 0) {
    try {
      imageMaterialized = materializeRequestImages(req.messages, respId);
    } catch (e) {
      return openaiError(`Failed to decode image input: ${e}`, 400);
    }
  }
  const imageFiles = imageMaterialized?.paths ?? [];

  const captureCodexHeaders = (headers: Record<string, string>) => {
    Object.assign(codexResponseHeaders, extractCodexUsageHeaders(headers));
  };

  const requestedModelForResponse = clientModelIgnored ? fallbackModel : (clientModel || fallbackModel);

  try {
    activeRequests++;
    logger.info(
      { respId, model: resolvedModel, provider, stream: req.stream },
      "[%s] request model=%s provider=%s stream=%s",
      respId,
      resolvedModel,
      provider,
      req.stream
    );

    if (!req.stream) {
      // ─── Non-streaming path ───────────────────────────────────────────────
      let text = "";
      let usage: Record<string, number> | null = null;

      await acquireSemaphore();
      try {
        if (provider === "codex") {
          const codexModel = effectiveProviderModel ?? settings.default_model;

          const runCodexOnce = async (modelId: string) => {
            if (useCodexBackend) {
              const auth = loadCodexAuth(settings.codex_cli_home);
              let token = auth.apiKey ?? auth.accessToken;
              if (!token) {
                await maybeRefreshCodexAuth(settings.codex_cli_home, Math.min(settings.timeout_seconds, 30) * 1000);
                const auth2 = loadCodexAuth(settings.codex_cli_home);
                token = auth2.apiKey ?? auth2.accessToken;
              }
              if (!token) {
                throw new Error("Missing Codex auth token (run `codex login` to create ~/.codex/auth.json).");
              }
              const headers = buildCodexHeaders({
                token,
                accountId: auth.accountId,
                sessionId: codexSessionId,
                version: settings.codex_responses_version,
                userAgent: settings.codex_responses_user_agent,
              });
              const backendReq: ChatCompletionRequest = {
                ...req,
                model: modelId,
                messages: maybeInjectAutomationGuardMessages(req.messages),
              };
              const payload = convertChatCompletionsToCodexResponses(backendReq, {
                modelName: modelId,
                forceStream: true,
                reasoningEffortOverride: reasoningEffort === "xhigh" ? "high" : reasoningEffort,
                allowTools: settings.codex_allow_tools,
              });
              const events = iterCodexResponsesEvents({
                baseUrl: settings.codex_responses_base_url,
                headers,
                payload,
                timeoutSeconds: settings.timeout_seconds,
                responseHeadersCb: captureCodexHeaders,
              });
              const [t, u, tc] = await collectCodexResponsesTextAndUsage(events);
              const usageMapped = u
                ? {
                    prompt_tokens: Math.floor(Number((u as Record<string, unknown>).prompt_tokens) || 0),
                    completion_tokens: Math.floor(Number((u as Record<string, unknown>).completion_tokens) || 0),
                    total_tokens: Math.floor(Number((u as Record<string, unknown>).total_tokens) || 0),
                  }
                : null;
              return { text: t, usage: usageMapped, toolCalls: tc };
            }
            const events = iterCodexEvents({
              prompt,
              model: modelId,
              cd: settings.workspace,
              images: imageFiles,
              disableShellTool: settings.disable_shell_tool,
              disableViewImageTool: settings.disable_view_image_tool,
              sandbox: settings.sandbox,
              skipGitRepoCheck: settings.skip_git_repo_check,
              modelReasoningEffort: reasoningEffort,
              approvalPolicy: settings.approval_policy,
              enableSearch: settings.enable_search,
              addDirs: [...settings.add_dirs],
              jsonEvents: true,
              codexCliHome: settings.codex_cli_home,
              timeoutSeconds: settings.timeout_seconds,
              streamLimit: settings.subprocess_stream_limit,
            });
            const result = await collectCodexTextAndUsageFromEvents(events);
            return { text: result.text, usage: result.usage, toolCalls: null };
          };

          try {
            const result = await runCodexOnce(codexModel);
            text = result.text;
            usage = result.usage;
            toolCalls = result.toolCalls ?? null;
          } catch (e) {
            const msg = String(e);
            if (
              useCodexBackend &&
              (msg.includes("codex responses failed: 401") || msg.includes("codex responses failed: 403"))
            ) {
              await maybeRefreshCodexAuth(settings.codex_cli_home, Math.min(settings.timeout_seconds, 30) * 1000);
              const result = await runCodexOnce(codexModel);
              text = result.text;
              usage = result.usage;
              toolCalls = result.toolCalls ?? null;
            } else if (
              codexModel !== settings.default_model &&
              looksLikeUnsupportedModelError(msg)
            ) {
              const result = await runCodexOnce(settings.default_model);
              text = result.text;
              usage = result.usage;
              toolCalls = result.toolCalls ?? null;
            } else {
              throw e;
            }
          }
        } else if (provider === "cursor-agent") {
          const cursorModel = effectiveProviderModel ?? settings.cursor_agent_model ?? "auto";
          const cursorWorkspace = settings.cursor_agent_workspace ?? settings.workspace;
          ensureWorkspaceDir(cursorWorkspace);
          const cmd = [
            settings.cursor_agent_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--workspace",
            cursorWorkspace,
            "--force",
            "--trust",
            "--approve-mcps",
          ];
          if (settings.cursor_agent_disable_indexing) cmd.push("--disable-indexing");
          cmd.push(...settings.cursor_agent_extra_args);
          if (settings.cursor_agent_api_key) cmd.push("--api-key", settings.cursor_agent_api_key);
          if (cursorModel) cmd.push("--model", cursorModel);
          if (settings.cursor_agent_stream_partial_output) cmd.push("--stream-partial-output");
          const { cmd: finalCmd, stdinData } = buildCursorAgentCmd(cmd, prompt);

          const assembler = new TextAssembler();
          let fallbackText: string | null = null;
          for await (const evt of iterStreamJsonEvents({
            cmd: finalCmd,
            timeoutMs: settings.timeout_seconds * 1000,
            totalTimeoutMs: settings.timeout_seconds * 1000,
            killOnResult: true,
            stdinData,
          })) {
            extractCursorAgentDelta(evt, assembler);
            if (evt.type === "result" && typeof evt.result === "string") fallbackText = evt.result;
          }
          text = assembler.text || fallbackText || "";
          if (requestTools && requestTools.length > 0) {
            const parsed = parseToolCallResponse(text);
            text = parsed.text;
            if (parsed.toolCalls) toolCalls = parsed.toolCalls as unknown as Record<string, unknown>[];
          }
        } else if (provider === "claude") {
          const claudeModel = effectiveProviderModel ?? settings.claude_model ?? "sonnet";
          if (useClaudeOauth) {
            const msgs = maybeInjectAutomationGuardMessages(req.messages);
            const req2: ChatCompletionRequest = {
              ...req,
              messages: msgs,
            };
            const [t, u, tc] = await claudeOauthGenerate(req2, claudeModel);
            text = t;
            usage = u;
            toolCalls = tc;
          } else {
            const cmd = [
              settings.claude_bin,
              "--verbose",
              "-p",
              "--output-format",
              "stream-json",
              "--add-dir",
              settings.workspace,
              ...settings.add_dirs.flatMap((d) => ["--add-dir", d]),
            ];
            if (claudeModel) cmd.push("--model", claudeModel);
            cmd.push("--", prompt);

            const assembler = new TextAssembler();
            let fallbackText: string | null = null;
            for await (const evt of iterStreamJsonEvents({
              cmd,
              timeoutMs: settings.timeout_seconds * 1000,
            })) {
              extractClaudeDelta(evt, assembler);
              const u = extractUsageFromClaudeResult(evt);
              if (u) usage = u;
              if (evt.type === "result" && typeof evt.result === "string") fallbackText = evt.result;
            }
            text = assembler.text || fallbackText || "";
          }
        } else if (provider === "gemini") {
          const geminiModel = effectiveProviderModel ?? settings.gemini_model ?? "gemini-3-flash-preview";
          if (useGeminiCloudcode) {
            const msgs = maybeInjectAutomationGuardMessages(req.messages);
            const req2: ChatCompletionRequest = { ...req, messages: msgs };
            const [t, u] = await geminiCloudcodeGenerate(req2, {
              modelName: geminiModel,
              reasoningEffort,
              timeoutMs: settings.timeout_seconds * 1000,
            });
            text = t;
            usage = u;
          } else {
            const cmd = [settings.gemini_bin, "-o", "stream-json"];
            if (geminiModel) cmd.push("-m", geminiModel);
            cmd.push(prompt);

            const assembler = new TextAssembler();
            for await (const evt of iterStreamJsonEvents({
              cmd,
              timeoutMs: settings.timeout_seconds * 1000,
            })) {
              extractGeminiDelta(evt, assembler);
              const u = extractUsageFromGeminiResult(evt);
              if (u) usage = u;
            }
            text = assembler.text;
          }
        } else {
          throw new Error(`Unknown provider: ${provider}`);
        }
      } finally {
        releaseSemaphore();
      }

      text = maybeStripAnswerTags(text).trim();
      const durationMs = Date.now() - t0;
      activeRequests--;

      logger.info(
        { respId, durationMs, chars: text.length, usage },
        "[%s] response status=200 duration_ms=%d chars=%d",
        respId,
        durationMs,
        text.length
      );

      const finishReason = toolCalls?.length ? "tool_calls" : "stop";
      const message: Record<string, unknown> = { role: "assistant", content: text };
      if (toolCalls?.length) message.tool_calls = toolCalls;

      const response: Record<string, unknown> = {
        id: respId,
        object: "chat.completion",
        created,
        model: requestedModelForResponse,
        choices: [{ index: 0, message, finish_reason: finishReason }],
      };
      if (usage) response.usage = usage;

      if (Object.keys(codexResponseHeaders).length > 0) {
        return new Response(JSON.stringify(response), {
          status: 200,
          headers: { "Content-Type": "application/json", ...codexResponseHeaders },
        });
      }
      return response;
    }

    // ─── Streaming path ─────────────────────────────────────────────────────
    const keepaliveSec = Math.max(settings.sse_keepalive_seconds, 0);
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        let controllerClosed = false;
        const safeEnqueue = (data: Uint8Array) => {
          if (!controllerClosed) {
            try { controller.enqueue(data); } catch { controllerClosed = true; }
          }
        };
        const safeClose = () => {
          if (!controllerClosed) {
            controllerClosed = true;
            try { controller.close(); } catch { /* already closed */ }
          }
        };
        try {
          await acquireSemaphore();
          try {
            // First chunk: role
            const first = {
              id: respId,
              object: "chat.completion.chunk",
              created,
              model: requestedModelForResponse,
              choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
            };
            safeEnqueue(encoder.encode(`data: ${JSON.stringify(first)}\n\n`));

            let streamUsage: Record<string, number> | null = null;
            let streamToolCalls: Record<string, unknown>[] | null = null;
            let assembledText = "";
            let sentContent = false;

            const attemptModels: (string | null)[] =
              provider === "codex"
                ? [effectiveProviderModel ?? settings.default_model, settings.default_model].filter(
                    (m, i, a) => !a.slice(0, i).includes(m)
                  )
                : [effectiveProviderModel];

            for (let attemptIdx = 0; attemptIdx < attemptModels.length; attemptIdx++) {
              const attemptModel = attemptModels[attemptIdx];
              let events: AsyncGenerator<Record<string, unknown>>;

              if (provider === "codex") {
                const codexModel = attemptModel ?? settings.default_model;
                if (useCodexBackend) {
                  const auth = loadCodexAuth(settings.codex_cli_home);
                  let token = auth.apiKey ?? auth.accessToken;
                  if (!token) {
                    await maybeRefreshCodexAuth(settings.codex_cli_home, Math.min(settings.timeout_seconds, 30) * 1000);
                    const auth2 = loadCodexAuth(settings.codex_cli_home);
                    token = auth2.apiKey ?? auth2.accessToken;
                  }
                  if (!token) {
                    throw new Error("Missing Codex auth token (run `codex login` to create ~/.codex/auth.json).");
                  }
                  const headers = buildCodexHeaders({
                    token,
                    accountId: auth.accountId,
                    sessionId: codexSessionId,
                    version: settings.codex_responses_version,
                    userAgent: settings.codex_responses_user_agent,
                  });
                  const backendReq: ChatCompletionRequest = {
                    ...req,
                    model: codexModel,
                    messages: maybeInjectAutomationGuardMessages(req.messages),
                  };
                  const payload = convertChatCompletionsToCodexResponses(backendReq, {
                    modelName: codexModel,
                    forceStream: true,
                    reasoningEffortOverride: reasoningEffort === "xhigh" ? "high" : reasoningEffort,
                    allowTools: settings.codex_allow_tools,
                  });
                  events = iterCodexResponsesEvents({
                    baseUrl: settings.codex_responses_base_url,
                    headers,
                    payload,
                    timeoutSeconds: settings.timeout_seconds,
                  });
                } else {
                  events = iterCodexEvents({
                    prompt,
                    model: codexModel,
                    cd: settings.workspace,
                    images: imageFiles,
                    disableShellTool: settings.disable_shell_tool,
                    disableViewImageTool: settings.disable_view_image_tool,
                    sandbox: settings.sandbox,
                    skipGitRepoCheck: settings.skip_git_repo_check,
                    modelReasoningEffort: reasoningEffort,
                    approvalPolicy: settings.approval_policy,
                    enableSearch: settings.enable_search,
                    addDirs: [...settings.add_dirs],
                    jsonEvents: true,
                    codexCliHome: settings.codex_cli_home,
                    timeoutSeconds: settings.timeout_seconds,
                    streamLimit: settings.subprocess_stream_limit,
                  });
                }
              } else if (provider === "cursor-agent") {
                const cursorModel = effectiveProviderModel ?? settings.cursor_agent_model ?? "auto";
                const cursorWorkspace = settings.cursor_agent_workspace ?? settings.workspace;
                ensureWorkspaceDir(cursorWorkspace);
                const cmd = [
                  settings.cursor_agent_bin,
                  "-p",
                  "--output-format",
                  "stream-json",
                  "--workspace",
                  cursorWorkspace,
                  "--force",
                  "--trust",
                  "--approve-mcps",
                ];
                if (settings.cursor_agent_disable_indexing) cmd.push("--disable-indexing");
                cmd.push(...settings.cursor_agent_extra_args);
                if (settings.cursor_agent_api_key) cmd.push("--api-key", settings.cursor_agent_api_key);
                if (cursorModel) cmd.push("--model", cursorModel);
                if (settings.cursor_agent_stream_partial_output) cmd.push("--stream-partial-output");
                const { cmd: cursorFinalCmd, stdinData: cursorStdinData } = buildCursorAgentCmd(cmd, prompt);
                events = iterStreamJsonEvents({ cmd: cursorFinalCmd, timeoutMs: settings.timeout_seconds * 1000, totalTimeoutMs: settings.timeout_seconds * 1000, killOnResult: true, stdinData: cursorStdinData });
              } else if (provider === "claude") {
                const claudeModel = effectiveProviderModel ?? settings.claude_model ?? "sonnet";
                if (useClaudeOauth) {
                  const msgs = maybeInjectAutomationGuardMessages(req.messages);
                  const req2: ChatCompletionRequest = { ...req, messages: msgs };
                  events = iterClaudeOauthEvents(req2, claudeModel);
                } else {
                  const cmd = [
                    settings.claude_bin,
                    "--verbose",
                    "-p",
                    "--output-format",
                    "stream-json",
                    "--add-dir",
                    settings.workspace,
                    ...settings.add_dirs.flatMap((d) => ["--add-dir", d]),
                  ];
                  if (claudeModel) cmd.push("--model", claudeModel);
                  cmd.push("--", prompt);
                  events = iterStreamJsonEvents({ cmd, timeoutMs: settings.timeout_seconds * 1000 });
                }
              } else if (provider === "gemini") {
                const geminiModel = effectiveProviderModel ?? settings.gemini_model ?? "gemini-3-flash-preview";
                if (useGeminiCloudcode) {
                  const msgs = maybeInjectAutomationGuardMessages(req.messages);
                  const req2: ChatCompletionRequest = { ...req, messages: msgs };
                  events = iterGeminiCloudcodeEvents(req2, {
                    modelName: geminiModel,
                    reasoningEffort,
                    timeoutMs: settings.timeout_seconds * 1000,
                  });
                } else {
                  const cmd = [settings.gemini_bin, "-o", "stream-json"];
                  if (geminiModel) cmd.push("-m", geminiModel);
                  cmd.push(prompt);
                  events = iterStreamJsonEvents({ cmd, timeoutMs: settings.timeout_seconds * 1000 });
                }
              } else {
                throw new Error(`Unknown provider: ${provider}`);
              }

              const assembler = new TextAssembler();
              let shouldRetry = false;

              const STREAM_END = Symbol("stream_end");
              type QueueItem = Record<string, unknown> | typeof STREAM_END;
              const queue: QueueItem[] = [];
              let resolveNext: (() => void) | null = null;
              const waitForNext = () => new Promise<void>((r) => { resolveNext = r; });

              const pump = async () => {
                try {
                  for await (const evt of events) {
                    queue.push(evt);
                    resolveNext?.();
                    resolveNext = null;
                  }
                } catch (e) {
                  queue.push({ _gateway_error: String(e) });
                  resolveNext?.();
                }
                queue.push(STREAM_END);
                resolveNext?.();
              };
              void pump();

              let lastEventTime = Date.now();
              while (true) {
                let evt: Record<string, unknown> | typeof STREAM_END | null = null;
                if (queue.length > 0) {
                  evt = queue.shift() ?? null;
                } else {
                  const timeoutMs = keepaliveSec > 0 ? keepaliveSec * 1000 : 0;
                  if (timeoutMs > 0) {
                    const result = await Promise.race([
                      waitForNext().then(() => queue.shift() ?? null),
                      new Promise<null>((r) => setTimeout(() => r(null), timeoutMs)),
                    ]);
                    if (result === STREAM_END) break;
                    if (result === null && queue.length === 0) {
                      safeEnqueue(encoder.encode(": ping\n\n"));
                      lastEventTime = Date.now();
                      continue;
                    }
                    evt = result ?? queue.shift() ?? null;
                  } else {
                    await waitForNext();
                    evt = queue.shift() ?? null;
                  }
                }
                if (evt === STREAM_END) break;

                if (evt === null || (evt as Record<string, unknown>)._gateway_error !== undefined) {
                  const errMsg = (evt as { _gateway_error?: string })?._gateway_error;
                  if (errMsg) {
                    if (
                      provider === "codex" &&
                      attemptIdx === 0 &&
                      attemptModels.length > 1 &&
                      !sentContent &&
                      looksLikeUnsupportedModelError(errMsg)
                    ) {
                      shouldRetry = true;
                      break;
                    }
                    const status = extractUpstreamStatusCode(errMsg) ?? 500;
                    logger.error({ respId, status }, "[%s] stream error status=%d %s", respId, status, truncateForLog(errMsg));
                    const errObj = {
                      error: { message: errMsg, type: "upstream_error", code: status },
                    };
                    safeEnqueue(encoder.encode(`event: error\ndata: ${JSON.stringify(errObj)}\n\n`));
                  }
                  break;
                }

                lastEventTime = Date.now();
                let delta = "";

                if (provider === "codex") {
                  if (useCodexBackend) {
                    if (evt.type === "response.output_text.delta" && typeof evt.delta === "string") {
                      delta = maybeStripAnswerTags(evt.delta);
                    }
                    if (
                      !delta &&
                      !sentContent &&
                      evt.type === "response.output_text.done" &&
                      typeof evt.text === "string"
                    ) {
                      delta = maybeStripAnswerTags(evt.text);
                    }
                    if (evt.type === "response.completed") {
                      const resp = (evt.response as Record<string, unknown>) ?? {};
                      const u = resp.usage as Record<string, unknown> | undefined;
                      if (u && typeof u === "object") {
                        const pt = Math.floor(Number(u.input_tokens) || 0);
                        const ct = Math.floor(Number(u.output_tokens) || 0);
                        streamUsage = {
                          prompt_tokens: pt,
                          completion_tokens: ct,
                          total_tokens: pt + ct,
                        };
                      }
                      const parsed = extractCodexToolCalls(resp as Record<string, unknown>);
                      if (parsed.length) streamToolCalls = parsed;
                    }
                  } else {
                    if (evt.type === "item.completed") {
                      const item = (evt.item as Record<string, unknown>) ?? {};
                      if (item.type === "agent_message") {
                        delta = maybeStripAnswerTags(String(item.text ?? ""));
                      }
                    }
                  }
                } else if (provider === "cursor-agent") {
                  delta = maybeStripAnswerTags(extractCursorAgentDelta(evt, assembler));
                } else if (provider === "claude") {
                  delta = maybeStripAnswerTags(extractClaudeDelta(evt, assembler));
                  if (evt.type === "result" && Array.isArray(evt.tool_calls) && evt.tool_calls.length > 0) {
                    streamToolCalls = evt.tool_calls as Record<string, unknown>[];
                  }
                } else if (provider === "gemini") {
                  delta = maybeStripAnswerTags(extractGeminiDelta(evt, assembler));
                }

                if (delta) {
                  sentContent = true;
                  assembledText += delta;
                  const chunk = {
                    id: respId,
                    object: "chat.completion.chunk",
                    created,
                    model: requestedModelForResponse,
                    choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
                  };
                  safeEnqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                }
              }

              if (shouldRetry) continue;
              break;
            }

            // cursor-agent tool call: check accumulated text for tool call markers at stream end
            if (provider === "cursor-agent" && requestTools && requestTools.length > 0 && assembledText) {
              const parsed = parseToolCallResponse(assembledText);
              if (parsed.toolCalls && parsed.toolCalls.length > 0) {
                streamToolCalls = parsed.toolCalls as unknown as Record<string, unknown>[];
              }
            }

            // Final chunk
            const end = {
              id: respId,
              object: "chat.completion.chunk",
              created,
              model: requestedModelForResponse,
              choices: [
                {
                  index: 0,
                  delta: {},
                  finish_reason: streamToolCalls?.length ? "tool_calls" : "stop",
                },
              ],
            };
            if (streamToolCalls?.length) {
              const toolChunk = {
                id: respId,
                object: "chat.completion.chunk",
                created,
                model: requestedModelForResponse,
                choices: [{ index: 0, delta: { tool_calls: streamToolCalls }, finish_reason: null }],
              };
              safeEnqueue(encoder.encode(`data: ${JSON.stringify(toolChunk)}\n\n`));
            }
            safeEnqueue(encoder.encode(`data: ${JSON.stringify(end)}\n\n`));
            safeEnqueue(encoder.encode("data: [DONE]\n\n"));
          } finally {
            releaseSemaphore();
          }

          const durationMs = Date.now() - t0;
          activeRequests--;
          logger.info(
            { respId, durationMs, chars: 0 },
            "[%s] stream response status=200 duration_ms=%d",
            respId,
            durationMs
          );
        } catch (e) {
          activeRequests--;
          logger.error({ respId, err: e }, "[%s] stream error", respId);
          const errObj = {
            error: { message: String(e), type: "stream_error", code: 500 },
          };
          safeEnqueue(encoder.encode(`event: error\ndata: ${JSON.stringify(errObj)}\n\n`));
        } finally {
          if (imageMaterialized?.tmpdir) {
            try {
              const { rmSync } = await import("fs");
              rmSync(imageMaterialized.tmpdir, { recursive: true });
            } catch {
              /* ignore */
            }
          }
        }
        safeClose();
      },
    });

    const headers = new Headers({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      ...codexResponseHeaders,
    });
    return new Response(stream, { headers });
  } catch (e) {
    activeRequests--;
    const status = extractUpstreamStatusCode(e) ?? 500;
    const msg = String(e);
    logger.error({ respId, status }, "[%s] error status=%d %s", respId, status, truncateForLog(msg));
    return openaiError(msg, status);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Warmup & Cleanup (onStartup / onShutdown for CLI)
// ─────────────────────────────────────────────────────────────────────────────

export async function warmup(): Promise<void> {
  const provider = normalizeProvider(settings.provider);
  if (provider === "codex" && settings.use_codex_responses_api) {
    await warmupCodexAuth(settings.codex_cli_home);
  }
  if (provider === "gemini" && settings.gemini_use_cloudcode_api) {
    await warmupGeminiCaches(30_000);
  }
}

export async function cleanup(): Promise<void> {
  await closeAll();
}

export async function onStartup(): Promise<void> {
  await warmup();
}

export async function onShutdown(): Promise<void> {
  await cleanup();
}
