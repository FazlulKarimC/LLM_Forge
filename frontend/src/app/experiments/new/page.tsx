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
import { createExperiment, ExperimentConfig, CreateExperimentRequest } from "@/lib/api";

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
        model_name: "microsoft/phi-2",
        reasoning_method: "naive",
        dataset_name: "trivia_qa",
        temperature: 0.7,
        max_tokens: 256,
        num_samples: 100,
        retrieval_method: "none",
    });

    const mutation = useMutation({
        mutationFn: createExperiment,
        onSuccess: (experiment) => {
            queryClient.invalidateQueries({ queryKey: ["experiments"] });
            queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
            router.push(`/experiments/${experiment.id}`);
        },
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

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

        mutation.mutate(request);
    };

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
                <form onSubmit={handleSubmit} className="card p-6 space-y-6">
                    {/* Error Display */}
                    {mutation.error && (
                        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                            <p className="text-(--error)">
                                {mutation.error instanceof Error ? mutation.error.message : "Failed to create experiment"}
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
                                    <option value="microsoft/phi-2">Phi-2 (2.7B)</option>
                                    <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama (1.1B)</option>
                                    <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B</option>
                                </select>
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
                    <div className="flex justify-end gap-4 pt-4 border-t border-border">
                        <Link href="/experiments" className="px-4 py-2 text-(--text-body) hover:underline">
                            Cancel
                        </Link>
                        <button
                            type="submit"
                            disabled={mutation.isPending}
                            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {mutation.isPending ? "Creating..." : "Create Experiment"}
                        </button>
                    </div>
                </form>
            </main>
        </div>
    );
}
