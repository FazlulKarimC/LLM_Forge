/**
 * New Experiment Page
 * 
 * Form for creating a new experiment configuration.
 * 
 * TODO (Iteration 1): Implement form submission
 * TODO (Iteration 2): Add config validation
 * TODO (Iteration 3): Add config templates
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function NewExperimentPage() {
    const [formData, setFormData] = useState({
        name: '',
        description: '',
        model_name: 'microsoft/phi-2',
        reasoning_method: 'naive',
        dataset_name: 'trivia_qa',
        temperature: 0.7,
        max_tokens: 256,
        num_samples: 100,
        retrieval_method: 'none',
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        // TODO (Iteration 1): Submit to API
        console.log('Submitting experiment:', formData);
        alert('Not implemented: Iteration 1');
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow">
                <div className="max-w-3xl mx-auto px-4 py-6">
                    <Link href="/experiments" className="text-blue-600 hover:underline text-sm">
                        ← Back to Experiments
                    </Link>
                    <h1 className="text-2xl font-bold text-gray-900 mt-1">
                        New Experiment
                    </h1>
                </div>
            </header>

            <main className="max-w-3xl mx-auto px-4 py-8">
                <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6 space-y-6">
                    {/* Basic Info */}
                    <div>
                        <h2 className="text-lg font-semibold mb-4">Basic Information</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Experiment Name *
                                </label>
                                <input
                                    type="text"
                                    required
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                    placeholder="e.g., naive_vs_cot_comparison"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Description
                                </label>
                                <textarea
                                    value={formData.description}
                                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                    rows={2}
                                    placeholder="Optional description..."
                                />
                            </div>
                        </div>
                    </div>

                    {/* Model Configuration */}
                    <div>
                        <h2 className="text-lg font-semibold mb-4">Model Configuration</h2>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Model
                                </label>
                                <select
                                    value={formData.model_name}
                                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                >
                                    <option value="microsoft/phi-2">Phi-2 (2.7B)</option>
                                    <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama (1.1B)</option>
                                    <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B (Colab)</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Reasoning Method
                                </label>
                                <select
                                    value={formData.reasoning_method}
                                    onChange={(e) => setFormData({ ...formData, reasoning_method: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                >
                                    <option value="naive">Naive</option>
                                    <option value="cot">Chain of Thought</option>
                                    <option value="react">ReAct Agent</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Hyperparameters */}
                    <div>
                        <h2 className="text-lg font-semibold mb-4">Hyperparameters</h2>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Temperature
                                </label>
                                <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="2"
                                    value={formData.temperature}
                                    onChange={(e) => setFormData({ ...formData, temperature: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Max Tokens
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    max="4096"
                                    value={formData.max_tokens}
                                    onChange={(e) => setFormData({ ...formData, max_tokens: parseInt(e.target.value) })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Samples
                                </label>
                                <input
                                    type="number"
                                    min="1"
                                    max="10000"
                                    value={formData.num_samples}
                                    onChange={(e) => setFormData({ ...formData, num_samples: parseInt(e.target.value) })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Dataset & Retrieval */}
                    <div>
                        <h2 className="text-lg font-semibold mb-4">Dataset & Retrieval</h2>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Dataset
                                </label>
                                <select
                                    value={formData.dataset_name}
                                    onChange={(e) => setFormData({ ...formData, dataset_name: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                >
                                    <option value="trivia_qa">TriviaQA</option>
                                    <option value="hotpot_qa">HotpotQA</option>
                                    <option value="gsm8k">GSM8K (Math)</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">
                                    Retrieval Method
                                </label>
                                <select
                                    value={formData.retrieval_method}
                                    onChange={(e) => setFormData({ ...formData, retrieval_method: e.target.value })}
                                    className="mt-1 block w-full border rounded-md px-3 py-2"
                                >
                                    <option value="none">No RAG</option>
                                    <option value="naive">Naive RAG</option>
                                    <option value="hybrid">Hybrid RAG</option>
                                    <option value="reranked">Reranked RAG</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Submit */}
                    <div className="flex justify-end gap-4 pt-4 border-t">
                        <Link
                            href="/experiments"
                            className="px-4 py-2 text-gray-700 hover:text-gray-900"
                        >
                            Cancel
                        </Link>
                        <button
                            type="submit"
                            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                        >
                            Create Experiment
                        </button>
                    </div>
                </form>
            </main>
        </div>
    );
}
