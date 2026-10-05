"use client";
import { useEffect, useRef, useState } from "react";
import type { DatasetCase, DatasetDetail } from "@/lib/evaluation-api";

export const sample: DatasetCase[] = [
  { name: "Greeting", inputs: { query: "Hello" }, expected_output: "Hello" },
  {
    name: "Farewell",
    inputs: { query: "Goodbye" },
    expected_output: "Goodbye",
  },
];
export const pretty = (cases: DatasetCase[]) => JSON.stringify(cases, null, 2);

export function useDatasetEditor() {
  const [archived, setArchived] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<DatasetDetail | null>(null);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [text, setText] = useState(pretty(sample));
  const [format, setFormat] = useState<"json" | "csv">("json");
  const [importText, setImportText] = useState("");
  const [importPreview, setImportPreview] = useState<DatasetCase[] | null>(
    null,
  );
  const [caseView, setCaseView] = useState<"table" | "json">("table");
  const [caseIndex, setCaseIndex] = useState<number | null>(null);
  const casePanel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (caseIndex !== null) casePanel.current?.focus();
  }, [caseIndex]);
  const [caseDraft, setCaseDraft] = useState<{
    name: string;
    inputs: string;
    expected_output: string;
  } | null>(null);
  const [historyOffset, setHistoryOffset] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  return { archived, setArchived, offset, setOffset, selected, setSelected, editing, setEditing, name, setName, description, setDescription, text, setText, format, setFormat, importText, setImportText, importPreview, setImportPreview, caseView, setCaseView, caseIndex, setCaseIndex, caseDraft, setCaseDraft, historyOffset, setHistoryOffset, busy, setBusy, error, setError, message, setMessage, casePanel };
}
