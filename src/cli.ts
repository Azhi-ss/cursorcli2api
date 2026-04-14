#!/usr/bin/env node

/**
 * CLI entry point. Ported from Python cli.py.
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { program } from "commander";
import type { Socket } from "node:net";

function normalizeProvider(raw: string | undefined): string | null {
  if (!raw) return null;
  const v = raw.trim().toLowerCase();
  if (!v) return null;
  if (["auto", "codex", "claude", "gemini"].includes(v)) return v;
  if (["cursor-agent", "cursor_agent", "cursoragent", "cursor"].includes(v)) return "cursor-agent";
  return null;
}

function maybeLoadDotenv(path: string): void {
  try {
    if (!existsSync(path)) return;
    const stat = statSync(path);
    if (!stat.isFile()) return;
  } catch {
    return;
  }
  const content = readFileSync(path, { encoding: "utf-8" });
  for (const rawLine of content.split("\n")) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("export ")) line = line.slice(7).trim();
    const eqIdx = line.indexOf("=");
    if (eqIdx < 0) continue;
    const key = line.slice(0, eqIdx).trim();
    let value = line.slice(eqIdx + 1).trim();
    if (!key) continue;
    if (value.length >= 2 && (value[0] === "'" || value[0] === '"') && value[value.length - 1] === value[0]) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

function defaultEnvCandidates(): string[] {
  const cwdEnv = join(process.cwd(), ".env");
  if (existsSync(cwdEnv)) {
    try {
      if (statSync(cwdEnv).isFile()) return [cwdEnv];
    } catch {
      /* ignore */
    }
  }
  return [];
}

async function main(): Promise<void> {
  program
    .name("agent-cli-to-api")
    .description("Expose agent CLIs as an OpenAI-compatible /v1 API gateway.")
    .argument("[provider]", "Provider to use: codex|gemini|claude|cursor-agent (or `doctor`).")
    .argument("[mode]", "Optional mode: curl (log request curl commands).")
    .option("--host <host>", "Bind host", process.env.CODEX_HOST ?? "127.0.0.1")
    .option("--port <port>", "Bind port", process.env.CODEX_PORT ?? "8000")
    .option("--log-level <level>", "Log level", process.env.CODEX_LOG_LEVEL ?? "info")
    .option("--log-curl", "Log copy-pastable curl commands for incoming requests")
    .option("--env-file <path>", "Optionally load environment variables from this .env file")
    .option("--preset <name>", "Optional config preset (sets recommended env defaults)", process.env.CODEX_PRESET)
    .option("--auto-env", "Auto-load .env from the current directory (default: off)")
    .parse();

  const opts = program.opts<{
    host: string;
    port: string;
    logLevel: string;
    logCurl: boolean;
    envFile?: string;
    preset?: string;
    autoEnv: boolean;
  }>();

  const args = program.args;
  const providerArg = args[0];
  const modeArg = args[1];

  if (opts.envFile) {
    maybeLoadDotenv(opts.envFile);
    if (existsSync(opts.envFile)) {
      console.error(`[agent-cli-to-api] loaded env: ${opts.envFile}`);
    }
  } else if (opts.autoEnv) {
    for (const candidate of defaultEnvCandidates()) {
      maybeLoadDotenv(candidate);
      console.error(`[agent-cli-to-api] loaded env: ${candidate}`);
      break;
    }
  } else {
    process.env.CODEX_NO_DOTENV = process.env.CODEX_NO_DOTENV ?? "1";
  }

  const normalizedProvider = normalizeProvider(providerArg);
  const providerRaw = (providerArg ?? "").trim().toLowerCase().replace(/_/g, "-");

  if (providerArg && !normalizedProvider && providerRaw !== "doctor") {
    console.error(`Unknown provider: ${providerArg}`);
    process.exit(1);
  }

  const modeRaw = (modeArg ?? "").trim().toLowerCase();
  if (modeRaw) {
    if (modeRaw === "curl") {
      process.env.CODEX_LOG_REQUEST_CURL = process.env.CODEX_LOG_REQUEST_CURL ?? "1";
    } else {
      console.error(`Unknown mode: ${modeArg}`);
      process.exit(1);
    }
  }
  if (opts.logCurl) {
    process.env.CODEX_LOG_REQUEST_CURL = process.env.CODEX_LOG_REQUEST_CURL ?? "1";
  }

  if (normalizedProvider) {
    process.env.CODEX_PROVIDER = normalizedProvider;
  }

  if (opts.preset) {
    process.env.CODEX_PRESET = opts.preset;
  } else if (normalizedProvider && !process.env.CODEX_PRESET) {
    if (normalizedProvider === "codex") {
      process.env.CODEX_PRESET = process.env.CODEX_PRESET ?? "codex-fast";
    } else if (normalizedProvider === "cursor-agent") {
      process.env.CODEX_PRESET = process.env.CODEX_PRESET ?? "cursor-auto";
    } else if (normalizedProvider === "gemini") {
      const { homedir } = await import("node:os");
      const credsPath = (process.env.GEMINI_OAUTH_CREDS_PATH ?? "~/.gemini/oauth_creds.json")
        .replace(/^~/, homedir());
      if (existsSync(credsPath)) {
        process.env.CODEX_PRESET = process.env.CODEX_PRESET ?? "gemini-cloudcode";
      }
    } else if (normalizedProvider === "claude") {
      const { homedir } = await import("node:os");
      const credsPath = (process.env.CLAUDE_OAUTH_CREDS_PATH ?? "~/.claude/oauth_creds.json")
        .replace(/^~/, homedir());
      if (existsSync(credsPath)) {
        process.env.CODEX_PRESET = process.env.CODEX_PRESET ?? "claude-oauth";
        console.error(`[agent-cli-to-api] detected Claude OAuth creds: ${credsPath}`);
      } else {
        const { readFileSync } = await import("node:fs");
        const { join } = await import("node:path");
        const cliSettingsPath = join(homedir(), ".claude", "settings.json");
        if (existsSync(cliSettingsPath)) {
          try {
            const data = JSON.parse(readFileSync(cliSettingsPath, "utf-8")) as Record<string, unknown>;
            const env = (data.env as Record<string, unknown>) ?? {};
            if (
              typeof env.ANTHROPIC_AUTH_TOKEN === "string" &&
              env.ANTHROPIC_AUTH_TOKEN &&
              typeof env.ANTHROPIC_BASE_URL === "string" &&
              env.ANTHROPIC_BASE_URL
            ) {
              process.env.CODEX_PRESET = process.env.CODEX_PRESET ?? "claude-oauth";
              console.error("[agent-cli-to-api] detected Claude CLI config (API key mode)");
            } else {
              console.error(
                "[agent-cli-to-api] settings.json exists but missing ANTHROPIC_AUTH_TOKEN or ANTHROPIC_BASE_URL"
              );
            }
          } catch (e) {
            console.error(`[agent-cli-to-api] failed to parse settings.json: ${e}`);
          }
        } else {
          console.error("[agent-cli-to-api] no OAuth creds or CLI config found, using CLI mode");
        }
      }
    }
  }

  if (process.stderr.isTTY) {
    process.env.CODEX_RICH_LOGS = process.env.CODEX_RICH_LOGS ?? "1";
    process.env.CODEX_LOG_RENDER_MARKDOWN = process.env.CODEX_LOG_RENDER_MARKDOWN ?? "1";
    process.env.CODEX_LOG_STREAM_INLINE = process.env.CODEX_LOG_STREAM_INLINE ?? "1";
    process.env.CODEX_LOG_STREAM_INLINE_SUPPRESS_FINAL = process.env.CODEX_LOG_STREAM_INLINE_SUPPRESS_FINAL ?? "0";
  }

  if (providerRaw === "doctor") {
    const { runDoctor } = await import("./doctor.js");
    const code = await runDoctor();
    process.exit(code);
  }

  const host = opts.host;
  const port = parseInt(opts.port, 10) || 8000;
  const logLevel = opts.logLevel;

  try {
    console.error(
      `[agent-cli-to-api] starting provider=${normalizedProvider ?? process.env.CODEX_PROVIDER ?? "auto"} host=${host} port=${port} log_level=${logLevel}`
    );
  } catch {
    /* ignore */
  }

  const { serve } = await import("@hono/node-server");
  const { app, onStartup, onShutdown } = await import("./server.js");

  await onStartup();
  const server = serve({ fetch: app.fetch, port, hostname: host });

  const benignSocketErrorCodes = new Set(["EPIPE", "ECONNRESET"]);
  const attachSocketGuards = (socket: Socket) => {
    socket.on("error", (err: NodeJS.ErrnoException) => {
      if (err?.code && benignSocketErrorCodes.has(err.code)) {
        console.error(`[agent-cli-to-api] ignored socket ${err.code} from disconnected client`);
        return;
      }
      console.error("[agent-cli-to-api] socket error:", err);
    });
  };

  server.on("connection", attachSocketGuards);
  server.on("clientError", (err: NodeJS.ErrnoException) => {
    if (err?.code && benignSocketErrorCodes.has(err.code)) {
      console.error(`[agent-cli-to-api] ignored clientError ${err.code}`);
      return;
    }
    console.error("[agent-cli-to-api] clientError:", err);
  });

  process.on("SIGTERM", async () => {
    await onShutdown();
    process.exit(0);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
