"use client";

/**
 * New Experiment Page
 * 
 * Form for creating a new experiment configuration.
 * Uses TanStack Query for mutation.
 * Styled with DESIGN_SYSTEM.md.
 */

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { motion } from "framer-motion";
import { createExperiment, runExperiment, getAvailableModels, ExperimentConfig, CreateExperimentRequest, AgentConfig } from "@/lib/api";


export default function NewExperimentPage() {
    const router = useRouter();
    const queryClient = useQueryClient();

    // Fetch models from backend
    const { data: modelsData } = useQuery({
        queryKey: ["available-models"],
        queryFn: getAvailableModels,
        staleTime: 1000 * 60 * 5, // 5 minutes (reduced from 1 hour to ensure fresh model lists)
    });
    const availableModels = modelsData?.models ?? [
        { value: "meta-llama/Llama-3.2-1B-Instruct", label: "Llama 3.2 (1B)", description: "Fast, efficient — default" },
    ];

    const [formData, setFormData] = useState<{
        name: string;
        description: string;
        model_name: string;
        reasoning_method: "naive" | "cot" | "react";
        dataset_name: string;
        temperature: number;
        max_tokens: number;
        num_samples: number;
        retrieval_method: "none" | "naive" | "hybrid" | "reranked";
        rag_top_k: number;
        agent_max_iterations: number;
        agent_tools: string[];
        enable_batching: boolean;
        batch_size: number;
        enable_caching: boolean;
        cache_max_size: number;
        seed: number | "";
    }>({
        name: "",
        description: "",
        model_name: "meta-llama/Llama-3.2-1B-Instruct",
        reasoning_method: "naive",
        dataset_name: "trivia_qa",
        temperature: 0.1,
        max_tokens: 150,
        num_samples: 10,
        retrieval_method: "none",
        rag_top_k: 5,
        agent_max_iterations: 5,
        agent_tools: ["wikipedia_search", "calculator"],
        enable_batching: false,
        batch_size: 8,
        enable_caching: false,
        cache_max_size: 256,
        seed: "",
    });

    const runAfterCreateRef = useRef(false);
    const [validationError, setValidationError] = useState<string | null>(null);
    const [runError, setRunError] = useState<string | null>(null);

    // Custom LLM States
    const [customBaseUrl, setCustomBaseUrl] = useState(() => {
        if (typeof window !== "undefined") return localStorage.getItem("customBaseUrl") || "http://localhost:8000/v1";
        return "http://localhost:8000/v1";
    });
    const [customApiKey, setCustomApiKey] = useState(() => {
        if (typeof window !== "undefined") return localStorage.getItem("customApiKey") || "";
        return "";
    });
    const [customModelId, setCustomModelId] = useState(() => {
        if (typeof window !== "undefined") return localStorage.getItem("customModelId") || "";
        return "";
    });

    const createMutation = useMutation({
        mutationFn: createExperiment,
        onSuccess: async (experiment) => {
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
            queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });

            if (runAfterCreateRef.current) {
                try {
                    await runExperiment(
                        experiment.id,
                        formData.model_name === "custom_hosted" ? customBaseUrl : undefined,
                        formData.model_name === "custom_hosted" ? customApiKey : undefined
                    );
                } catch (err) {
                    // Surface the error as a banner — user can retry from the detail page
                    setRunError(
                        err instanceof Error
                            ? `Experiment created but failed to start: ${err.message}`
                            : 'Experiment created but failed to start. Retry from the detail page.'
                    );
                }
            }
            router.push(`/experiments/${experiment.id}`);
        },
    });

    const handleSubmit = (e: React.FormEvent, shouldRun: boolean = false) => {
        e.preventDefault();

        // --- Client-side validation ---
        if (!formData.name.trim()) {
            setValidationError("Experiment name is required.");
            return;
        }
        if (formData.num_samples < 1 || formData.num_samples > 500) {
            setValidationError("Samples must be between 1 and 500.");
            return;
        }
        if (formData.enable_batching && (formData.batch_size < 1 || formData.batch_size > 32)) {
            setValidationError("Batch size must be between 1 and 32.");
            return;
        }
        if (formData.enable_caching && (formData.cache_max_size < 8 || formData.cache_max_size > 4096)) {
            setValidationError("Cache max size must be between 8 and 4096.");
            return;
        }
        setValidationError(null);
        runAfterCreateRef.current = shouldRun;
        if (formData.model_name === "custom_hosted") {
            if (!customBaseUrl.trim() || !customModelId.trim()) {
                setValidationError("Base URL and Model ID are required for custom hosted models.");
                return;
            }
            // Save to local storage for future use
            if (typeof window !== "undefined") {
                try {
                    const settings = JSON.parse(localStorage.getItem("customLLMSettings") || "{}");
                    settings[customModelId] = { baseUrl: customBaseUrl, apiKey: customApiKey };
                    localStorage.setItem("customLLMSettings", JSON.stringify(settings));
                } catch (e) {
                    console.error("Failed to save custom LLM settings", e);
                }
                // Backward compatibility
                localStorage.setItem("customBaseUrl", customBaseUrl);
                localStorage.setItem("customApiKey", customApiKey);
                localStorage.setItem("customModelId", customModelId);
            }
        }

        const config: ExperimentConfig = {
            model_name: formData.model_name === "custom_hosted" ? customModelId : formData.model_name,
            reasoning_method: formData.reasoning_method,
            dataset_name: formData.dataset_name,
            hyperparameters: {
                temperature: formData.temperature,
                max_tokens: formData.max_tokens,
                ...(formData.seed !== "" ? { seed: formData.seed as number } : {})
            },
            num_samples: formData.num_samples,
        };

        if (formData.retrieval_method !== "none") {
            config.rag = {
                retrieval_method: formData.retrieval_method,
                top_k: formData.rag_top_k,
            };
        }

        if (formData.reasoning_method === "react") {
            config.agent = {
                max_iterations: formData.agent_max_iterations,
                tools: formData.agent_tools,
            };
        }

        // Optimization settings (Phase 8)
        if (formData.enable_batching || formData.enable_caching) {
            config.optimization = {
                enable_batching: formData.enable_batching,
                batch_size: formData.batch_size,
                enable_caching: formData.enable_caching,
                cache_max_size: formData.cache_max_size,
                enable_profiling: true,
            };
        }

        const request: CreateExperimentRequest = {
            name: formData.name,
            description: formData.description || undefined,
            config,
        };

        createMutation.mutate(request);
    };

    const selectedModel = availableModels.find(m => m.value === formData.model_name);

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-primary hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <h1 className="text-2xl font-serif text-(--text-heading) mt-1">
                        New Experiment
                    </h1>
                </div>
            </header>

            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-7xl mx-auto px-4 py-8"
            >
                <form onSubmit={(e) => handleSubmit(e, false)} className="card p-6 space-y-6">
                    {/* Validation Error Display */}
                    {validationError && (
                        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                            <p className="text-(--error) text-sm font-medium">⚠ {validationError}</p>
                        </div>
                    )}

                    {/* Run Error Display (Create & Run flow) */}
                    {runError && (
                        <div className="bg-yellow-50 border border-yellow-300 rounded-xl p-4">
                            <p className="text-yellow-800 text-sm font-medium">⚠ {runError}</p>
                        </div>
                    )}

                    {/* API Error Display */}
                    {createMutation.error && (

                        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                            <p className="text-(--error)">
                                {createMutation.error instanceof Error ? createMutation.error.message : "Failed to create experiment"}
                            </p>
                        </div>
                    )}

                    {/* Basic Info */}
                    <section>
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Basic Information</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">
                                    Experiment Name *
                                </label>
                                <input
                                    type="text"
                                    required
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) focus:outline-none focus:ring-2 focus:ring-primary"
                                    placeholder="e.g., naive_vs_cot_comparison"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">
                                    Description
                                </label>
                                <textarea
                                    value={formData.description}
                                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) focus:outline-none focus:ring-2 focus:ring-primary"
                                    rows={2}
                                    placeholder="Optional description..."
                                />
                            </div>
                        </div>
                    </section>

                    {/* Model Configuration */}
                    <section>
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Model Configuration</h2>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Model</label>
                                <select
                                    value={formData.model_name}
                                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                >
                                    <optgroup label="Free HF Serverless API">
                                        {availableModels.map(model => (
                                            <option key={model.value} value={model.value}>
                                                {model.label}
                                            </option>
                                        ))}
                                    </optgroup>
                                    <optgroup label="Custom Options">
                                        <option value="custom_hosted">🔌 Custom Hosted Model (OpenAI-Compatible)</option>
                                    </optgroup>
                                </select>
                                {formData.model_name === "custom_hosted" ? (
                                    <p className="text-xs text-(--text-muted) mt-1">Connect to vLLM, Ollama, Together AI, groq, or any OpenAI-compatible API endpoint.</p>
                                ) : selectedModel ? (
                                    <p className="text-xs text-(--text-muted) mt-1">{selectedModel.description}</p>
                                ) : null}
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Reasoning Method</label>
                                <select
                                    value={formData.reasoning_method}
                                    onChange={(e) => setFormData({ ...formData, reasoning_method: e.target.value as "naive" | "cot" | "react" })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                >
                                    <option value="naive">Naive</option>
                                    <option value="cot">Chain of Thought</option>
                                    <option value="react">ReAct Agent</option>
                                </select>
                            </div>
                        </div>

                        {/* Custom Model Settings */}
                        {formData.model_name === "custom_hosted" && (
                            <div className="mt-4 p-4 bg-(--bg-page) rounded-lg border border-border space-y-4">
                                <h3 className="text-sm font-semibold text-(--text-heading) flex items-center gap-2">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="m10 13-2 2-2-2m2 2v-6m7 0 2 2-2 2m-2-2v6m-4-7V4h8v3" />
                                    </svg>
                                    Custom API Endpoint Settings
                                </h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">Base URL *</label>
                                        <input
                                            type="url"
                                            required={formData.model_name === "custom_hosted"}
                                            value={customBaseUrl}
                                            onChange={(e) => setCustomBaseUrl(e.target.value)}
                                            className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) font-mono text-sm"
                                            placeholder="http://localhost:8000/v1"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">API Key (Optional)</label>
                                        <input
                                            type="password"
                                            value={customApiKey}
                                            onChange={(e) => setCustomApiKey(e.target.value)}
                                            className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) font-mono text-sm"
                                            placeholder="sk-..."
                                        />
                                        <p className="text-xs text-(--text-muted) mt-1">Not saved to DB. Only sent when running.</p>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">Model ID *</label>
                                        <input
                                            type="text"
                                            required={formData.model_name === "custom_hosted"}
                                            value={customModelId}
                                            onChange={(e) => setCustomModelId(e.target.value)}
                                            className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) font-mono text-sm"
                                            placeholder="e.g., Llama-3-8B-Instruct"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </section>

                    {/* Hyperparameters */}
                    <section>
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Hyperparameters</h2>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Temperature</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="2"
                                    value={formData.temperature}
                                    onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Max Tokens</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="4096"
                                    value={formData.max_tokens}
                                    onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Seed (Optional)</label>
                                <input
                                    type="number"
                                    min="0"
                                    value={formData.seed}
                                    onChange={(e) => setFormData({ ...formData, seed: e.target.value === "" ? "" : parseInt(e.target.value) })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                    placeholder="Random"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Samples</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="500"
                                    value={formData.num_samples}
                                    onChange={(e) => setFormData({ ...formData, num_samples: parseInt(e.target.value) })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                />
                                <p className="text-xs text-(--text-muted) mt-1">Max 500 samples</p>
                            </div>
                        </div>
                    </section>

                    {/* Dataset & Retrieval */}
                    <section>
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">Dataset & Retrieval</h2>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Dataset</label>
                                <select
                                    value={formData.dataset_name}
                                    onChange={(e) => setFormData({ ...formData, dataset_name: e.target.value })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                >
                                    <optgroup label="📋 General">
                                        <option value="trivia_qa">TriviaQA (100 Qs)</option>
                                        <option value="commonsense_qa">Commonsense QA (30 Qs)</option>
                                        <option value="sample">Sample Questions (10 Qs)</option>
                                    </optgroup>
                                    <optgroup label="📚 RAG">
                                        <option value="knowledge_base">Knowledge Base QA (50 Qs)</option>
                                    </optgroup>
                                    <optgroup label="🧠 Reasoning">
                                        <option value="multi_hop">Multi-Hop QA (40 Qs)</option>
                                        <option value="math_reasoning">Math Reasoning (40 Qs)</option>
                                    </optgroup>
                                    <optgroup label="🤖 Agent">
                                        <option value="react_bench">ReAct Agent Bench (30 Qs)</option>
                                    </optgroup>
                                </select>
                                <p className="text-xs text-(--text-muted) mt-1">
                                    {formData.dataset_name === "trivia_qa" && "Single-hop factual recall questions"}
                                    {formData.dataset_name === "commonsense_qa" && "Everyday knowledge and reasoning"}
                                    {formData.dataset_name === "sample" && "Built-in smoke test questions"}
                                    {formData.dataset_name === "knowledge_base" && "⭐ Questions answerable from indexed articles — ideal for RAG"}
                                    {formData.dataset_name === "multi_hop" && "Requires combining 2+ facts — ideal for CoT & ReAct"}
                                    {formData.dataset_name === "math_reasoning" && "GSM8K-style word problems — ideal for CoT & calculator"}
                                    {formData.dataset_name === "react_bench" && "⭐ Multi-tool questions requiring search + calculation"}
                                </p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-(--text-body)">Retrieval Method</label>
                                <select
                                    value={formData.retrieval_method}
                                    onChange={(e) => setFormData({ ...formData, retrieval_method: e.target.value as "none" | "naive" | "hybrid" | "reranked" })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                >
                                    <option value="none">No RAG</option>
                                    <option value="naive">Naive RAG</option>
                                    <option value="hybrid">Hybrid RAG</option>
                                    <option value="reranked">Reranked RAG</option>
                                </select>
                            </div>
                        </div>
                        {/* RAG-specific settings */}
                        {formData.retrieval_method !== "none" && (
                            <div className="mt-4 p-4 bg-(--bg-page) rounded-lg border border-border space-y-3">
                                <h3 className="text-sm font-semibold text-(--text-heading) flex items-center gap-2">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                                    </svg>
                                    RAG Settings
                                </h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">Top-K Chunks</label>
                                        <input
                                            type="number"
                                            min="1"
                                            max="20"
                                            value={formData.rag_top_k}
                                            onChange={(e) => setFormData({ ...formData, rag_top_k: parseInt(e.target.value) || 5 })}
                                            className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                        />
                                        <p className="text-xs text-(--text-muted) mt-1">Number of context chunks to retrieve</p>
                                    </div>
                                    <div className="flex items-end">
                                        <p className="text-xs text-(--text-muted) p-2 bg-(--bg-card) rounded-lg border border-border">
                                            💡 <strong>Naive</strong>: Dense retrieval&ensp;•&ensp;<strong>Hybrid</strong>: Dense + BM25&ensp;•&ensp;<strong>Reranked</strong>: Hybrid + cross-encoder
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </section>

                    {/* Agent Settings (Phase 6) */}
                    {formData.reasoning_method === "react" && (
                        <section>
                            <h2 className="text-lg font-serif text-(--text-heading) mb-4">Agent Settings</h2>
                            <div className="p-4 bg-(--bg-page) rounded-lg border border-border space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">Max Iterations</label>
                                        <input
                                            type="number"
                                            min="1"
                                            max="20"
                                            value={formData.agent_max_iterations}
                                            onChange={(e) => setFormData({ ...formData, agent_max_iterations: parseInt(e.target.value) || 5 })}
                                            className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                        />
                                        <p className="text-xs text-(--text-muted) mt-1">Max Thought→Action→Observation loops</p>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-(--text-body)">Tools</label>
                                        <div className="mt-2 space-y-2">
                                            {["wikipedia_search", "calculator", "retrieval"].map(tool => (
                                                <label key={tool} className="flex items-center gap-2 text-sm text-(--text-body) cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={formData.agent_tools.includes(tool)}
                                                        onChange={(e) => {
                                                            const tools = e.target.checked
                                                                ? [...formData.agent_tools, tool]
                                                                : formData.agent_tools.filter(t => t !== tool);
                                                            setFormData({ ...formData, agent_tools: tools });
                                                        }}
                                                        className="rounded border-border"
                                                    />
                                                    {tool === "wikipedia_search" ? "🌐 Wikipedia" : tool === "calculator" ? "🧮 Calculator" : "📚 Retrieval (RAG)"}
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                <p className="text-xs text-(--text-muted) p-2 bg-(--bg-card) rounded-lg border border-border">
                                    🤖 The agent will reason step-by-step, using tools to gather information before answering. More iterations = more thorough but slower and more expensive.
                                </p>
                            </div>
                        </section>
                    )}

                    {/* Optimization Settings (Phase 8) */}
                    <section>
                        <h2 className="text-lg font-serif text-(--text-heading) mb-4">⚡ Optimization</h2>
                        <div className="p-4 bg-(--bg-page) rounded-lg border border-border space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="flex items-center gap-2 text-sm font-medium text-(--text-body) cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={formData.enable_batching}
                                            onChange={(e) => setFormData({ ...formData, enable_batching: e.target.checked })}
                                            className="rounded border-border"
                                        />
                                        Enable Batching
                                    </label>
                                    {formData.enable_batching && (
                                        <div className="mt-2">
                                            <label className="block text-xs text-(--text-muted)">Batch Size</label>
                                            <input
                                                type="number"
                                                min="1"
                                                max="32"
                                                value={formData.batch_size}
                                                onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) || 8 })}
                                                className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) text-sm"
                                            />
                                            <p className="text-xs text-(--text-muted) mt-1">Concurrent API calls per batch</p>
                                        </div>
                                    )}
                                </div>
                                <div>
                                    <label className="flex items-center gap-2 text-sm font-medium text-(--text-body) cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={formData.enable_caching}
                                            onChange={(e) => setFormData({ ...formData, enable_caching: e.target.checked })}
                                            className="rounded border-border"
                                        />
                                        Enable Caching
                                    </label>
                                    {formData.enable_caching && (
                                        <div className="mt-2">
                                            <label className="block text-xs text-(--text-muted)">Max Cache Entries</label>
                                            <input
                                                type="number"
                                                min="16"
                                                max="2048"
                                                value={formData.cache_max_size}
                                                onChange={(e) => setFormData({ ...formData, cache_max_size: parseInt(e.target.value) || 256 })}
                                                className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body) text-sm"
                                            />
                                            <p className="text-xs text-(--text-muted) mt-1">LRU cache for identical prompts</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                            <p className="text-xs text-(--text-muted) p-2 bg-(--bg-card) rounded-lg border border-border">
                                ⚡ <strong>Batching</strong> parallelizes API calls for faster wall-clock time.
                                <strong>Caching</strong> stores results for repeated prompts.
                                Profiling is always enabled.
                                {formData.reasoning_method === "react" && " Note: Batching is disabled for ReAct agent (requires iterative tool calling)."}
                            </p>
                        </div>
                    </section>

                    {/* Submit */}
                    <div className="flex justify-end gap-3 pt-4 border-t border-border">
                        <Link href="/experiments" className="px-4 py-2 text-(--text-body) hover:underline">
                            Cancel
                        </Link>
                        <button
                            type="submit"
                            disabled={createMutation.isPending}
                            className="px-6 py-2 rounded-full border border-border text-(--text-body) hover:bg-(--bg-page) transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                        >
                            {createMutation.isPending && !runAfterCreateRef.current ? "Creating..." : "Create Experiment"}
                        </button>
                        <button
                            type="button"
                            disabled={createMutation.isPending}
                            onClick={(e) => handleSubmit(e, true)}
                            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer inline-flex items-center gap-2"
                        >
                            {createMutation.isPending && runAfterCreateRef.current ? (
                                <>
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                    </svg>
                                    Creating & Running...
                                </>
                            ) : (
                                <>
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                        <polygon points="5,3 19,12 5,21" />
                                    </svg>
                                    Create & Run
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </motion.main>
        </div>
    );
}
