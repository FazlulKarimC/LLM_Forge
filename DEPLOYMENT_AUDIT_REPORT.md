# Deployment Audit Report: Vercel (Frontend) & Hugging Face Spaces (Backend)

We have conducted a thorough audit of the `LlmForge` codebase (Frontend & Backend) against the latest free-tier constraints of Vercel (Hobby) and Hugging Face Spaces. 

Here are the detailed findings and action items required for a successful free deployment.

---

## 1. Frontend Audit (Vercel Hobby Tier)
**Verdict: Highly Compatible (with minor configuration)**

### Vercel Free Tier Constraints
- **Serverless Timeout:** 10 to 60 seconds (default 10s for Hobby).
- **Execution Payload:** Max 4.5 MB for Request/Response body.
- **Build Time:** Max 45 minutes per deployment.

### Codebase Analysis
- **Architecture Win:** All pages in `src/app/` are explicitly marked with `"use client";`. This means the Next.js app is entirely Client-Side Rendered (CSR). 
- **Why this matters:** When the browser makes a request to the backend, it bypasses the Next.js Serverless Function timeout. If the backend (on HF Spaces) takes 30-60 seconds to "wake up" from sleep, a Server Component (`getServerSideProps` equivalent) would hit the 10-second Vercel timeout and throw a 504 error. Because you are using client-side `fetch`, the browser will patiently wait for the backend to wake up. This is a massive architectural advantage for free-tier deployments.

### Action Items for Vercel
- [ ] **Environment Variable:** You MUST set `NEXT_PUBLIC_API_URL` in your Vercel project settings to point to your deployed Hugging Face Space URL (e.g., `https://your-username-llmforge.hf.space/api/v1`). If forgotten, the API client defaults to `localhost:8000`.
- [ ] **CORS Configuration:** Ensure `CORS_ORIGINS` in your backend environment variables includes your auto-generated Vercel domain (e.g., `https://llmforge-frontend.vercel.app`).

---

## 2. Backend Audit (Hugging Face Spaces Free Tier)
**Verdict: Compatible (Requires Database Architecture Shift & Config Care)**

### Hugging Face Spaces Free Tier Constraints
- **Hardware:** Basic CPU (2 vCPU, 16GB RAM).
- **Sleep Cycle:** Space goes to sleep after 48 hours of inactivity.
- **Storage Mode:** The 50GB disk storage is **ephemeral**. Any files written to the local `/app` directory are wiped when the space restarts or wakes up from sleep.
- **Port Requirement:** Applications must expose port `7860`.

### Codebase Analysis & Critical Risks

**1. The Ephemeral Database Risk (CRITICAL)**
- **Finding:** In `backend/app/core/database.py`, if `DATABASE_URL` is omitted, the app falls back to a local SQLite database (`sqlite+aiosqlite:///./test.db`). 
- **Deploy Impact:** Because HF Spaces storage is ephemeral, if you rely on the local SQLite fallback, **every time your Space sleeps and wakes up, all your experiment data, results, and metrics will be permanently deleted.**
- **Solution:** You MUST use an external, free PostgreSQL provider (such as NeonDB, Supabase, or Render). Set the `DATABASE_URL` environment variable in your HF Spaces Settings -> Variables to point to this external database.

**2. Out of Memory (OOM) Risk**
- **Finding:** The application includes a `transformers_engine.py` inference engine that can load models locally. 
- **Deploy Impact:** Loading a modern LLM (even 7B or quantized 8B) directly into memory will instantly crash the 16GB RAM limit of the HF Spaces free tier.
- **Solution:** You MUST ensure the environment variable `INFERENCE_ENGINE` is set to `hf_api` (or `openai`) and NOT `transformers`. This offloads the heavy lifting to external API endpoints. Set `HF_TOKEN` in your Space secrets so `hf_api_engine` can authenticate.

**3. Port Configuration**
- **Finding:** Your `Dockerfile` is already perfectly configured! It correctly copies the backend, exposes `EXPOSE 7860`, and runs `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]`. No changes needed here.

---

## Summary of Deployment Checklist

### Vercel Dashboard (Frontend)
1. Add Environment Variable: `NEXT_PUBLIC_API_URL` = `https://<YOUR_HF_SPACE_ID>.hf.space/api/v1`
2. Deploy Next.js App

### Hugging Face Spaces Dashboard (Backend)
1. Create a new Docker Space.
2. Push your codebase (ensure `Dockerfile` is at the root or correctly referenced).
3. Add Environment Secrets/Variables:
   - `DATABASE_URL`: Your free Neon/Supabase PostgreSQL connection string. *(DO NOT OMIT THIS)*
   - `INFERENCE_ENGINE`: `hf_api`
   - `HF_TOKEN`: Your Hugging Face access token for external inference.
   - `CORS_ORIGINS`: Your deployed Vercel URL.
   - `ENVIRONMENT`: `production`
