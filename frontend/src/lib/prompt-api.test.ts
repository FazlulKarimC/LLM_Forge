import { describe, expect, it } from "vitest";
import { draftVariables } from "./prompt-api";
describe("template variable inputs", () => {
  it("deduplicates mustache names without treating JSON braces as variables", () => {
    expect(draftVariables('{"answer": "{{ query }}"} {{query}} {{customer_name}}', "mustache")).toEqual(["customer_name", "query"]);
  });
  it("does not collect escaped brace variables", () => {
    expect(draftVariables("{{literal}} {question}", "fstring")).toEqual(["question"]);
  });
  it("collects a variable wrapped in escaped literal braces", () => {
    expect(draftVariables("{{{query}}} {{{{literal}}}} {other}", "fstring")).toEqual(["other", "query"]);
  });
});
