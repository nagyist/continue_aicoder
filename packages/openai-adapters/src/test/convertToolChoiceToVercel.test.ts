import type { ChatCompletionToolChoiceOption } from "openai/resources/index.js";
import { describe, expect, test } from "vitest";
import { convertToolChoiceToVercel } from "../convertToolChoiceToVercel.js";

describe("convertToolChoiceToVercel", () => {
  test("returns undefined for undefined input", () => {
    const result = convertToolChoiceToVercel(undefined);
    expect(result).toBeUndefined();
  });

  test("passes through 'auto' string value", () => {
    const result = convertToolChoiceToVercel("auto");
    expect(result).toBe("auto");
  });

  test("passes through 'none' string value", () => {
    const result = convertToolChoiceToVercel("none");
    expect(result).toBe("none");
  });

  test("passes through 'required' string value", () => {
    const result = convertToolChoiceToVercel("required");
    expect(result).toBe("required");
  });

  test("converts function object format to Vercel tool format", () => {
    const toolChoice: ChatCompletionToolChoiceOption = {
      type: "function",
      function: {
        name: "readFile",
      },
    };

    const result = convertToolChoiceToVercel(toolChoice);

    expect(result).toEqual({
      type: "tool",
      toolName: "readFile",
    });
  });

  test("converts function object with different tool name", () => {
    const toolChoice: ChatCompletionToolChoiceOption = {
      type: "function",
      function: {
        name: "writeFile",
      },
    };

    const result = convertToolChoiceToVercel(toolChoice);

    expect(result).toEqual({
      type: "tool",
      toolName: "writeFile",
    });
  });

  test("handles function name with special characters", () => {
    const toolChoice: ChatCompletionToolChoiceOption = {
      type: "function",
      function: {
        name: "my_complex-tool.v2",
      },
    };

    const result = convertToolChoiceToVercel(toolChoice);

    expect(result).toEqual({
      type: "tool",
      toolName: "my_complex-tool.v2",
    });
  });

  test("returns undefined for unknown object format", () => {
    const toolChoice = {
      type: "unknown_type",
    } as unknown as ChatCompletionToolChoiceOption;

    const result = convertToolChoiceToVercel(toolChoice);

    expect(result).toBeUndefined();
  });
});
