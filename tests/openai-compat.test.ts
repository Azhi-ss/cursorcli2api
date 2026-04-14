import test from "node:test";
import assert from "node:assert/strict";

import {
  compatChatRequestToChatRequest,
  normalizeToolingRequest,
  responsesRequestToChatRequest,
  type ChatCompletionRequest,
} from "../src/lib/openai-compat.js";

test("normalizeToolingRequest mirrors top-level tool fields into model_extra", () => {
  const request = normalizeToolingRequest({
    model: "claude:sonnet",
    stream: false,
    max_tokens: null,
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ],
    tool_choice: "auto",
    parallel_tool_calls: false,
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: "{\"city\":\"Beijing\"}" },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_1",
        content: "sunny",
      },
    ],
  } as ChatCompletionRequest & Record<string, unknown>);

  const extra = (request as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(extra.tools, (request as Record<string, unknown>).tools);
  assert.equal(extra.tool_choice, "auto");
  assert.equal(extra.parallel_tool_calls, false);

  const assistantExtra = ((request.messages[0] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  const toolExtra = ((request.messages[1] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  assert.deepEqual(assistantExtra.tool_calls, (request.messages[0] as Record<string, unknown>).tool_calls);
  assert.equal(toolExtra.tool_call_id, "call_1");
});

test("normalizeToolingRequest preserves existing model_extra precedence", () => {
  const topLevelTools = [{ type: "function", function: { name: "wrong" } }];
  const modelExtraTools = [{ type: "function", function: { name: "right" } }];

  const request = normalizeToolingRequest({
    model: "codex:auto",
    stream: false,
    max_tokens: null,
    tools: topLevelTools,
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "wrong", arguments: "{}" } }],
        model_extra: {
          tool_calls: [{ id: "call_1", type: "function", function: { name: "right", arguments: "{}" } }],
        },
      },
    ],
    model_extra: {
      tools: modelExtraTools,
      tool_choice: "required",
    },
  } as ChatCompletionRequest & Record<string, unknown>);

  const extra = (request as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(extra.tools, modelExtraTools);
  assert.equal(extra.tool_choice, "required");

  const assistantExtra = ((request.messages[0] as Record<string, unknown>).model_extra ?? {}) as Record<string, unknown>;
  assert.deepEqual(assistantExtra.tool_calls, [{ id: "call_1", type: "function", function: { name: "right", arguments: "{}" } }]);
});

test("compat and responses request conversion keep top-level tool fields available in model_extra", () => {
  const compatRequest = compatChatRequestToChatRequest({
    model: "claude:sonnet",
    stream: false,
    messages: [{ role: "user", content: "hello" }],
    tools: [{ type: "function", function: { name: "lookup" } }],
    tool_choice: { type: "function", function: { name: "lookup" } },
  } as ChatCompletionRequest & Record<string, unknown>);

  const compatExtra = (compatRequest as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(compatExtra.tools, [{ type: "function", function: { name: "lookup" } }]);
  assert.deepEqual(compatExtra.tool_choice, { type: "function", function: { name: "lookup" } });

  const responsesRequest = responsesRequestToChatRequest({
    model: "gemini:flash",
    input: [{ role: "user", content: [{ type: "input_text", text: "hi" }] }],
    stream: false,
    tools: [{ type: "function", function: { name: "lookup" } }],
    parallel_tool_calls: true,
  } as Record<string, unknown>);

  const responsesExtra = (responsesRequest as Record<string, unknown>).model_extra as Record<string, unknown>;
  assert.deepEqual(responsesExtra.tools, [{ type: "function", function: { name: "lookup" } }]);
  assert.equal(responsesExtra.parallel_tool_calls, true);
});
