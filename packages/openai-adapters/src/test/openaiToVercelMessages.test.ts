import type { ChatCompletionMessageParam } from "openai/resources/index.js";
import { describe, expect, test } from "vitest";
import { convertOpenAIMessagesToVercel } from "../openaiToVercelMessages.js";

describe("convertOpenAIMessagesToVercel", () => {
  describe("system messages", () => {
    test("converts simple system message", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "system", content: "You are a helpful assistant." },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "system",
        content: "You are a helpful assistant.",
      });
    });

    test("converts system message with non-string content", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "system",
          content: [{ type: "text", text: "System prompt" }],
        } as ChatCompletionMessageParam,
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("system");
      // Non-string content should be JSON stringified
      expect(typeof result[0].content).toBe("string");
    });
  });

  describe("user messages", () => {
    test("converts simple user message", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "user", content: "Hello!" },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "user",
        content: "Hello!",
      });
    });

    test("preserves user message with array content (multimodal)", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            {
              type: "image_url",
              image_url: { url: "https://example.com/image.png" },
            },
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("user");
      expect(Array.isArray(result[0].content)).toBe(true);
      expect(result[0].content).toHaveLength(2);
    });
  });

  describe("assistant messages", () => {
    test("converts simple assistant message", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "assistant", content: "I can help you with that." },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "assistant",
        content: "I can help you with that.",
      });
    });

    test("converts assistant message with empty content", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "assistant", content: null },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "assistant",
        content: "",
      });
    });

    test("converts assistant message with tool calls", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "/path/to/file"}',
              },
            },
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0].role).toBe("assistant");
      expect(Array.isArray(result[0].content)).toBe(true);

      const content = result[0].content as any[];
      expect(content).toHaveLength(1);
      expect(content[0]).toEqual({
        type: "tool-call",
        toolCallId: "call_123",
        toolName: "readFile",
        input: { filepath: "/path/to/file" },
      });
    });

    test("converts assistant message with multiple tool calls", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "/file1"}',
              },
            },
            {
              id: "call_2",
              type: "function",
              function: {
                name: "writeFile",
                arguments: '{"filepath": "/file2", "content": "test"}',
              },
            },
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      const content = result[0].content as any[];
      expect(content).toHaveLength(2);
      expect(content[0].toolName).toBe("readFile");
      expect(content[1].toolName).toBe("writeFile");
    });

    test("converts assistant message with tool calls and text content", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: "Let me read that file for you.",
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "/path/to/file"}',
              },
            },
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      const content = result[0].content as any[];
      expect(content).toHaveLength(2);
      expect(content[0]).toEqual({
        type: "text",
        text: "Let me read that file for you.",
      });
      expect(content[1].type).toBe("tool-call");
    });

    test("handles malformed JSON in tool call arguments", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "testTool",
                arguments: "not valid json",
              },
            },
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      const content = result[0].content as any[];
      // Malformed JSON should be passed through as string
      expect(content[0].input).toBe("not valid json");
    });

    test("preserves thought_signature from Google provider options", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "testTool",
                arguments: "{}",
              },
              extra_content: {
                google: {
                  thought_signature: "sig_abc123",
                },
              },
            } as any,
          ],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      const content = result[0].content as any[];
      expect(content[0].providerOptions).toEqual({
        google: {
          thoughtSignature: "sig_abc123",
        },
      });
    });
  });

  describe("tool messages", () => {
    test("converts tool result message with matching assistant call", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "/path/to/file"}',
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_123",
          content: "File contents here",
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(2);
      expect(result[1].role).toBe("tool");

      const content = result[1].content as any[];
      expect(content).toHaveLength(1);
      expect(content[0]).toEqual({
        type: "tool-result",
        toolCallId: "call_123",
        toolName: "readFile",
        output: {
          type: "text",
          value: "File contents here",
        },
      });
    });

    test("uses unknown_tool for tool result without matching call", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "tool",
          tool_call_id: "unknown_call_id",
          content: "Some result",
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      const content = result[0].content as any[];
      expect(content[0].toolName).toBe("unknown_tool");
    });

    test("converts tool result with non-string content", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              type: "function",
              function: {
                name: "getJson",
                arguments: "{}",
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_123",
          content: [{ type: "text", text: "JSON result" }],
        } as any,
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      const content = result[1].content as any[];
      // Non-string content should be JSON stringified
      expect(typeof content[0].output.value).toBe("string");
    });
  });

  describe("multi-turn conversations", () => {
    test("handles complete multi-turn tool conversation", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "system", content: "You are a file assistant." },
        { role: "user", content: "Read the config file." },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "config.json"}',
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_1",
          content: '{"setting": true}',
        },
        {
          role: "assistant",
          content: "The config file contains: setting is true.",
        },
        { role: "user", content: "Thanks!" },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(6);
      expect(result[0].role).toBe("system");
      expect(result[1].role).toBe("user");
      expect(result[2].role).toBe("assistant");
      expect(result[3].role).toBe("tool");
      expect(result[4].role).toBe("assistant");
      expect(result[5].role).toBe("user");

      // Verify tool call conversion
      const toolCallContent = result[2].content as any[];
      expect(toolCallContent[0].type).toBe("tool-call");
      expect(toolCallContent[0].toolName).toBe("readFile");

      // Verify tool result conversion
      const toolResultContent = result[3].content as any[];
      expect(toolResultContent[0].type).toBe("tool-result");
      expect(toolResultContent[0].toolName).toBe("readFile");
    });

    test("handles multiple tool calls with their results", () => {
      const messages: ChatCompletionMessageParam[] = [
        { role: "user", content: "Read both files." },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "file1.txt"}',
              },
            },
            {
              id: "call_2",
              type: "function",
              function: {
                name: "readFile",
                arguments: '{"filepath": "file2.txt"}',
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_1",
          content: "Content of file 1",
        },
        {
          role: "tool",
          tool_call_id: "call_2",
          content: "Content of file 2",
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(4);

      // Check both tool results have correct tool names
      const toolResult1 = result[2].content as any[];
      const toolResult2 = result[3].content as any[];

      expect(toolResult1[0].toolCallId).toBe("call_1");
      expect(toolResult1[0].toolName).toBe("readFile");
      expect(toolResult2[0].toolCallId).toBe("call_2");
      expect(toolResult2[0].toolName).toBe("readFile");
    });
  });

  describe("edge cases", () => {
    test("handles empty messages array", () => {
      const result = convertOpenAIMessagesToVercel([]);
      expect(result).toEqual([]);
    });

    test("handles assistant message with empty tool_calls array", () => {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: "assistant",
          content: "Regular response",
          tool_calls: [],
        },
      ];

      const result = convertOpenAIMessagesToVercel(messages);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        role: "assistant",
        content: "Regular response",
      });
    });
  });
});
