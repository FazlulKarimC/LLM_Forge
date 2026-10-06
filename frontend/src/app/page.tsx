import Link from "next/link";
import type { Metadata } from "next";
import { ArrowRight, ArrowUpRight, Check, Code2, GitCompareArrows, Github, FileText, FlaskConical } from "lucide-react";
import { PublicHeader, repositoryUrl } from "@/components/public-header";
import { PublicFooter } from "@/components/public-footer";
import { ProductWalkthrough } from "@/features/landing/product-walkthrough";
import { ProductVisual } from "@/features/landing/product-visual";
import { productPreviews } from "@/features/landing/product-previews";

export const metadata: Metadata = {
  title: "LLMForge — Compare models, develop prompts, evaluate outputs",
  description: "Compare models, develop prompts, and evaluate outputs in one workspace. Explore LLMForge's public product walkthrough, then try the provider-free demo.",
};

const steps = [
  { title: "Configure", text: "Choose models for a benchmark, or save a prompt version and a fixed set of test cases." },
  { title: "Compare", text: "Inspect benchmark trade-offs or compare compatible evaluation runs. Read the outputs behind the numbers." },
  { title: "Improve", text: "Use disagreements and failed checks to guide your next change. Test it again before moving a prompt release label." },
];
const faqs = [
  { question: "Do I need to sign in?", answer: "The walkthrough and documentation are public. Sign in to open the workspace, select a project and save prompts, datasets or runs." },
  { question: "Do I need a provider API key?", answer: "The Echo demo needs no model credits. Live prompt generation, judging and provider-backed model runs need the relevant provider credentials. Evaluation keys stay in memory and must be supplied again after an interruption." },
  { question: "Are these real model results?", answer: "These are screenshots of the actual application with deliberately prepared examples. Seeded benchmark measurements and Echo demo checks illustrate the workflow; they do not measure real model quality or establish a winning model." },
  { question: "How do I try the workspace demo?", answer: "Sign in, select a project and choose Create demo examples on Overview. This prepares an Echo demo prompt and a Greetings dataset. Follow the demo guide to run checks and compare results." },
  { question: "Can I run it locally and use it in my application?", answer: "Yes. The repository documents local setup with Next.js, FastAPI, PostgreSQL and Clerk. The Python SDK fetches released prompts and submits evaluation results; CLI quality gates can check them in CI." },
];

export default function LandingPage() {
  return <div className="public-shell landing-page">
    <PublicHeader />
    <main id="main-content" className="page-width landing-content">
      <section className="landing-hero" aria-labelledby="hero-title">
        <p className="landing-eyebrow"><span />A workspace for better LLM decisions</p>
        <h1 id="hero-title">Know which LLM setup<br className="hidden sm:block" />{" "}<span>works best.</span></h1>
        <p className="landing-hero-description">Compare models, develop prompts, and evaluate outputs in one workspace—so you can make decisions with evidence.</p>
        <div className="flex flex-wrap gap-3">
          <a href="#walkthrough" className="btn-primary">View walkthrough <ArrowRight aria-hidden="true" className="size-4" /></a>
          <a href={repositoryUrl} target="_blank" rel="noreferrer" className="btn-secondary"><Github aria-hidden="true" className="size-4" />View GitHub</a>
        </div>
        <div className="hero-capabilities" aria-label="Core capabilities">
          <span><GitCompareArrows />Model benchmarks</span><span><FileText />Versioned prompts</span><span><FlaskConical />Reproducible evaluations</span>
        </div>
      </section>
      <ProductWalkthrough />
      <section id="workflow" aria-labelledby="workflow-title" className="landing-section">
        <div className="landing-section-intro"><div><p className="section-label">From a question to a decision</p>
          <h2 id="workflow-title">A clear path to the next improvement.</h2></div>
          <p>Benchmarks compare model setups. Evaluations test saved prompt versions. Both help you decide what to change next.</p></div>
        <ol className="workflow-steps">{steps.map((step, index) => <li key={step.title}>
          <span className="workflow-number">0{index + 1}</span><h3>{step.title}</h3><p>{step.text}</p>
        </li>)}</ol>
      </section>
      <section id="product" aria-labelledby="product-title" className="landing-section">
        <div className="landing-section-intro"><div><p className="section-label">Build with evidence</p>
          <h2 id="product-title">The tools behind a better setup.</h2></div><p>Keep configurations, outputs and checks close to the decisions they inform.</p></div>
        <div className="capability-list">{productPreviews.map((preview, index) => <article key={preview.id} className="capability-row">
          <div className="capability-copy"><p className="section-label">0{index + 1} / {preview.label}</p>
            <h3>{preview.benefit}</h3><p>{preview.detail}</p>
            <a href="#walkthrough" className="landing-text-link">Explore the walkthrough <ArrowRight aria-hidden="true" className="size-4" /></a>
          </div><figure className="capability-figure"><ProductVisual id={preview.id} /><figcaption>{preview.caption}</figcaption></figure>
        </article>)}</div>
      </section>
      <section aria-labelledby="evidence-title" className="landing-section evidence-section">
        <div className="landing-section-intro"><div><p className="section-label">Open code. Inspectable results.</p>
          <h2 id="evidence-title">Try the workflow. See how it works.</h2></div><p>LLMForge is a personal engineering project, with a working demo and public implementation.</p></div>
        <div className="evidence-grid">
          <article className="panel"><Github className="evidence-icon" aria-hidden="true" /><h3>Built in the open</h3>
            <p>Inspect the code, setup instructions and benchmark methodology in the repository.</p>
            <a href={repositoryUrl} target="_blank" rel="noreferrer" className="landing-text-link">View repository <ArrowUpRight className="size-4" aria-hidden="true" /></a></article>
          <article className="panel"><Code2 className="evidence-icon" aria-hidden="true" /><h3>Connect your application</h3>
            <p>Fetch released prompts with the Python SDK, submit evaluation results and apply quality thresholds in CI.</p>
            <Link href="/docs#sdk" className="landing-text-link">Read the SDK guide <ArrowRight className="size-4" aria-hidden="true" /></Link></article>
          <article className="panel"><Check className="evidence-icon" aria-hidden="true" /><h3>Start without model credits</h3>
            <p>The provider-free Echo demo makes prompt versions, dataset checks and comparison easy to explore.</p>
            <Link href="/docs#demo" className="landing-text-link">Follow the demo <ArrowRight className="size-4" aria-hidden="true" /></Link></article>
        </div>
      </section>
      <section aria-labelledby="faq-title" className="landing-section faq-section">
        <div><p className="section-label">Before you start</p><h2 id="faq-title">A few useful answers.</h2></div>
        <div className="faq-list">{faqs.map((faq) => <details key={faq.question}><summary>{faq.question}<span aria-hidden="true">+</span></summary><p>{faq.answer}</p></details>)}</div>
      </section>
      <section className="landing-final-cta" aria-labelledby="start-title"><div><p className="section-label">Your next experiment starts here</p>
        <h2 id="start-title">Make the next change with confidence.</h2><p>Sign in to your workspace, then choose <strong>Create demo examples</strong> on Overview.</p></div>
        <div className="flex flex-wrap gap-3"><Link href="/dashboard" className="btn-primary">Open workspace <ArrowRight className="size-4" aria-hidden="true" /></Link>
          <Link href="/docs#demo" className="btn-secondary">Read the demo guide</Link></div></section>
    </main>
    <PublicFooter />
  </div>;
}
