import test from "node:test";
import assert from "node:assert/strict";

import { extractToolCallsFromAnthropicResponse } from "../src/providers/claude-oauth.js";

test("extractToolCallsFromAnthropicResponse maps tool_use blocks to OpenAI tool_calls", () => {
  const toolCalls = extractToolCallsFromAnthropicResponse({
    id: "msg_123",
    content: [
      { type: "text", text: "" },
      {
        type: "tool_use",
        id: "toolu_1",
        name: "get_weather",
        input: { city: "Beijing" },
      },
    ],
  });

  assert.deepEqual(toolCalls, [
    {
      id: "toolu_1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Beijing\"}",
      },
    },
  ]);
});

test("extractToolCallsFromAnthropicResponse returns null when there is no tool_use", () => {
  assert.equal(
    extractToolCallsFromAnthropicResponse({
      content: [{ type: "text", text: "hello" }],
    }),
    null
  );
});
