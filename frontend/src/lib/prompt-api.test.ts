import { describe, expect, it } from "vitest";
import {
  draftVariables,
  promptDraftVariables,
  parsePromptConfig,
  canonicalJSON,
} from "./prompt-api";
describe("template variable inputs", () => {
  it("collects chat variables across role-separated messages", () => {
    expect(
      promptDraftVariables({
        prompt_type: "chat",
        template_text: "",
        template_format: "mustache",
        description: "",
        messages: [
          { role: "system", content: "Use {{language}}" },
          { role: "user", content: "{{query}} {{language}}" },
        ],
      }),
    ).toEqual(["language", "query"]);
  });
  it("validates finite config objects and compares keys independently of order", () => {
    expect(() => parsePromptConfig("[]")).toThrow();
    expect(() => parsePromptConfig('{"temperature":1e309}')).toThrow();
    expect(
      canonicalJSON(
        parsePromptConfig('{"temperature":0,"nested":{"b":2,"a":1}}'),
      ),
    ).toBe(canonicalJSON({ nested: { a: 1, b: 2 }, temperature: 0 }));
  });
  it("deduplicates mustache names without treating JSON braces as variables", () => {
    expect(
      draftVariables(
        '{"answer": "{{ query }}"} {{query}} {{customer_name}}',
        "mustache",
      ),
    ).toEqual(["customer_name", "query"]);
  });
  it("does not collect escaped brace variables", () => {
    expect(draftVariables("{{literal}} {question}", "fstring")).toEqual([
      "question",
    ]);
  });
  it("collects a variable wrapped in escaped literal braces", () => {
    expect(
      draftVariables("{{{query}}} {{{{literal}}}} {other}", "fstring"),
    ).toEqual(["other", "query"]);
  });
});
