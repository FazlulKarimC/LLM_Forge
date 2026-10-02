"use client";
import type { ChatMessage } from "@/lib/prompt-api";
import { inputClass } from "./prompt-ui";

export function ChatEditor({
  messages,
  onChange,
  disabled,
}: {
  messages: ChatMessage[];
  onChange: (messages: ChatMessage[]) => void;
  disabled: boolean;
}) {
  const change = (index: number, value: Partial<ChatMessage>) =>
    onChange(
      messages.map((message, i) =>
        i === index ? { ...message, ...value } : message,
      ),
    );
  function move(index: number, direction: number) {
    const next = [...messages];
    [next[index], next[index + direction]] = [
      next[index + direction],
      next[index],
    ];
    onChange(next);
  }
  return (
    <section aria-label="Chat messages" className="space-y-3">
      {messages.map((message, index) => (
        <div
          key={index}
          className="rounded-lg border border-(--border) p-3 space-y-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex-1 text-xs">
              Message {index + 1} role
              <select
                className={inputClass}
                value={message.role}
                disabled={disabled}
                onChange={(event) =>
                  change(index, {
                    role: event.target.value as ChatMessage["role"],
                  })
                }
              >
                <option value="system">System</option>
                <option value="user">User</option>
                <option value="assistant">Assistant</option>
              </select>
            </label>
            <button
              className="btn-ghost"
              aria-label={`Move message ${index + 1} up`}
              disabled={disabled || index === 0}
              onClick={() => move(index, -1)}
            >
              ↑
            </button>
            <button
              className="btn-ghost"
              aria-label={`Move message ${index + 1} down`}
              disabled={disabled || index === messages.length - 1}
              onClick={() => move(index, 1)}
            >
              ↓
            </button>
            <button
              className="btn-ghost"
              aria-label={`Remove message ${index + 1}`}
              disabled={disabled || messages.length === 1}
              onClick={() => onChange(messages.filter((_, i) => i !== index))}
            >
              Remove
            </button>
          </div>
          <label className="block text-xs">
            Message {index + 1} content
            <textarea
              className={`${inputClass} font-mono min-h-32`}
              value={message.content}
              maxLength={50_000}
              spellCheck={false}
              disabled={disabled}
              onChange={(event) =>
                change(index, { content: event.target.value })
              }
            />
          </label>
        </div>
      ))}
      <button
        className="btn-secondary"
        disabled={disabled || messages.length >= 100}
        onClick={() => onChange([...messages, { role: "user", content: "" }])}
      >
        Add message
      </button>
      <p className="text-xs text-(--muted-foreground)">
        Messages keep their order and roles when compiled and sent to a model.
        Use template variables inside each message.
      </p>
    </section>
  );
}
