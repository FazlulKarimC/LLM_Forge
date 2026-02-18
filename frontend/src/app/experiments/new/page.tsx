"use client";

/**
 * New Experiment Page
 * 
 * Form for creating a new experiment configuration.
 * Uses TanStack Query for mutation.
 * Styled with DESIGN_SYSTEM.md.
 */

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { createExperiment, runExperiment, ExperimentConfig, CreateExperimentRequest } from "@/lib/api";

// Free-tier models confirmed working on HuggingFace Inference API
const AVAILABLE_MODELS = [
    { value: "meta-llama/Llama-3.2-1B-Instruct", label: "Llama 3.2 (1B)", description: "Fast, efficient — default" },
    { value: "Qwen/Qwen2.5-3B-Instruct", label: "Qwen 2.5 (3B)", description: "Strong multilingual" },
    { value: "google/gemma-2-2b-it", label: "Gemma 2 (2B)", description: "Google's compact model" },
    { value: "microsoft/Phi-3.5-mini-instruct", label: "Phi-3.5 Mini (3.8B)", description: "Strong reasoning" },
];

export default function NewExperimentPage() {
    const router = useRouter();
    const queryClient = useQueryClient();

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
    });

    const [runAfterCreate, setRunAfterCreate] = useState(false);

    const createMutation = useMutation({
        mutationFn: createExperiment,
        onSuccess: async (experiment) => {
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
            queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });

            if (runAfterCreate) {
                try {
                    await runExperiment(experiment.id);
                } catch {
                    // Still navigate even if run fails — user can retry from detail page
                }
            }
            router.push(`/experiments/${experiment.id}`);
        },
    });

    const handleSubmit = (e: React.FormEvent, shouldRun: boolean = false) => {
        e.preventDefault();
        setRunAfterCreate(shouldRun);

        const config: ExperimentConfig = {
            model_name: formData.model_name,
            reasoning_method: formData.reasoning_method,
            dataset_name: formData.dataset_name,
            hyperparameters: {
                temperature: formData.temperature,
                max_tokens: formData.max_tokens,
            },
            num_samples: formData.num_samples,
        };

        if (formData.retrieval_method !== "none") {
            config.rag = { retrieval_method: formData.retrieval_method };
        }

        const request: CreateExperimentRequest = {
            name: formData.name,
            description: formData.description || undefined,
            config,
        };

        createMutation.mutate(request);
    };

    const selectedModel = AVAILABLE_MODELS.find(m => m.value === formData.model_name);

    return (
        <div className="min-h-screen bg-(--bg-page)">
            <header className="bg-(--bg-card) shadow-sm border-b border-border">
                <div className="max-w-3xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-primary hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <h1 className="text-2xl font-serif text-(--text-heading) mt-1">
                        New Experiment
                    </h1>
                </div>
            </header>

            <main className="max-w-3xl mx-auto px-4 py-8">
                <form onSubmit={(e) => handleSubmit(e, false)} className="card p-6 space-y-6">
                    {/* Error Display */}
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
                                    {AVAILABLE_MODELS.map(model => (
                                        <option key={model.value} value={model.value}>
                                            {model.label}
                                        </option>
                                    ))}
                                </select>
                                {selectedModel && (
                                    <p className="text-xs text-(--text-muted) mt-1">{selectedModel.description}</p>
                                )}
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
                                <label className="block text-sm font-medium text-(--text-body)">Samples</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="10000"
                                    value={formData.num_samples}
                                    onChange={(e) => setFormData({ ...formData, num_samples: parseInt(e.target.value) })}
                                    className="mt-1 block w-full border border-border rounded-lg px-3 py-2 bg-(--bg-card) text-(--text-body)"
                                />
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
                                    <option value="trivia_qa">TriviaQA</option>
                                    <option value="hotpot_qa">HotpotQA</option>
                                    <option value="gsm8k">GSM8K (Math)</option>
                                </select>
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
                            {createMutation.isPending && !runAfterCreate ? "Creating..." : "Create Experiment"}
                        </button>
                        <button
                            type="button"
                            disabled={createMutation.isPending}
                            onClick={(e) => handleSubmit(e, true)}
                            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer inline-flex items-center gap-2"
                        >
                            {createMutation.isPending && runAfterCreate ? (
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
            </main>
        </div>
    );
}
