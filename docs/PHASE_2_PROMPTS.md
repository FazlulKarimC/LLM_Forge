# Phase 2: prompts, playground and releases

The prompt hub is available at `/prompts` inside an authenticated LLMForge project.
Organizations created in Clerk are separate from LLMForge workspaces; use the
app's Settings and workspace switcher to manage its organizations and projects.

## Demonstration walkthrough

1. Open **Prompts → New prompt**, name it `support-answer`, and keep the default
   `{{query}}` template or write your own. Describe what the prompt is used for.
2. Enter a sample `query` in the playground. Preview the compiled prompt, then
   select **Demo** and run it without an API key. Demo output is a preview, not
   real LLM inference, and does not report token usage.
3. Create the prompt. Change the template and save a new version with a short
   explanation. The older version's text, format, hash and ID stay unchanged.
4. Select a saved version in history, promote it to `staging`, then confirm a
   promotion to `production`. Saving another version does not move either label.
5. In **Settings → Project API keys**, create a key. Copy its secret immediately;
   the database stores only its SHA-256 hash and a short display prefix.
6. Fetch the production prompt using the commands below. Promote an earlier
   version to production and fetch again to demonstrate rollback.
7. Revoke the key; the next fetch returns 401. Archive the prompt; it disappears
   from SDK reads but its snapshots remain available for benchmark reproducibility.
   Restore it from the archived prompt list when needed.

For real model output, choose Groq, OpenRouter or OpenAI, provide that provider's
API key, and enter a model ID your account supports. Compare up to three targets.
Credentials remain in page memory and are sent only for the current request;
they are not saved in browser storage, prompt versions or database run records.
Each target reports its own success or error. Request timeout is 30 seconds,
output limit is 2,048 tokens, and automatic retries/fallbacks are disabled.
Some models reject sampling settings; use a compatible chat model or adjust them.
Model availability and credits depend on your provider account.

## Fetch a released prompt

The SDK-facing endpoint needs only a read-only project API key. It does not use
a Clerk session or an `X-Project-ID` header: the key determines the project.
No production label means 404; there is no implicit latest-version fallback.

PowerShell:

```powershell
$env:LLMFORGE_API_KEY = 'YOUR_COPIED_PROJECT_KEY'
curl.exe -H "Authorization: Bearer $env:LLMFORGE_API_KEY" `
  "http://localhost:8000/api/v1/sdk/prompts/support-answer?label=production"
```

Bash:

```bash
export LLMFORGE_API_KEY='YOUR_COPIED_PROJECT_KEY'
curl -H "Authorization: Bearer $LLMFORGE_API_KEY" \
  'http://localhost:8000/api/v1/sdk/prompts/support-answer?label=production'
```

Use `?label=staging` for staging, or `?version=2` for a fixed snapshot. Supplying
both selectors is invalid. Prompt names are case-sensitive and must be URL-encoded.
The response includes ID, version, template text/format, required variables,
SHA-256 hash and timestamp. Conditional reads support `If-None-Match` / ETag;
promoting or rolling back a label changes the returned snapshot immediately.
Keys grant `prompts:read` only. They cannot invoke the playground or dashboard APIs.
Workspace owners can create/revoke up to 20 active keys per project.

Python example for a mustache prompt, using only the standard library:

```python
import json
import os
import re
from urllib.request import Request, urlopen

request = Request(
    "http://localhost:8000/api/v1/sdk/prompts/support-answer?label=production",
    headers={"Authorization": f"Bearer {os.environ['LLMFORGE_API_KEY']}"},
)
with urlopen(request, timeout=15) as response:
    prompt = json.load(response)

assert prompt["template_format"] == "mustache"
inputs = {"query": "How do I reset my password?"}
missing = set(prompt["variables"]) - inputs.keys()
if missing:
    raise ValueError(f"Missing variables: {sorted(missing)}")
compiled = re.sub(
    r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}",
    lambda match: inputs[match.group(1)],
    prompt["template_text"],
)
print(f"Using prompt v{prompt['version']}: {compiled}")
```

The packaged Python client, caching policy and CI helpers belong to Phase 4.
Phase 2 provides the authenticated fetch contract they will use.

## Template and version rules

- Mustache uses `{{variable}}`; literal JSON braces need no escaping.
- Brace format uses `{variable}` and `{{` / `}}` for literal braces. This is
  restricted string substitution, not Python expression evaluation.
- Variable names support letters, digits and underscores, starting with a letter
  or underscore. Attributes, format specifications, conversions, loops and
  sections are rejected. Variable values are substituted once and remain literal.
- Templates are limited to 50,000 characters and 100 variables. Input values
  are limited to 10,000 characters each; compiled prompts to 100,000 characters.
- Names are unique per project, including archived prompts. Rename changes the
  fetch name; existing snapshot names remain historical metadata.
- `base_version` detects stale editor saves with 409. PostgreSQL locks the prompt
  while allocating a version, preventing duplicate version numbers.
- Identical text/format cannot create another version. Roll back by moving a label.
- The benchmark runner accepts both formats and maps `query`, `question`, and
  `input` to the sample question, with its existing context/example variables.

## API and models

| API | Purpose |
| --- | --- |
| `/api/v1/prompt-library` | List/search and create stable prompts |
| `/api/v1/prompt-library/{id}` | Read/update metadata; DELETE archives |
| `/api/v1/prompt-library/{id}/versions` | Paginated history and immutable saves |
| `/api/v1/prompt-library/{id}/labels/{label}` | Atomic release promotion/removal |
| `/api/v1/prompt-library/compile` | Validate and compile editor drafts |
| `/api/v1/prompt-library/playground` | Request-only generation |
| `/api/v1/project-keys` | Owner-managed read-only credentials |
| `/api/v1/sdk/prompts/{name}` | Released or pinned snapshot fetch |

Dashboard routes require Clerk authentication and authorized project selection.
The old `/api/v1/prompts` snapshot API remains available for benchmark clients.
New tables: `prompts`, `prompt_labels`, `project_api_keys`. `prompt_versions`
now belongs to a stable prompt and records its template format. Migration
`j3k4l5m6n7o8` preserves existing text and IDs while normalizing old version numbering.
No playground output or provider key is persisted.

## Verification

```powershell
# backend
python -m pytest -q
# frontend
npm test
npm run lint
npx tsc --noEmit
npm run build
```

Tests cover compilation, stale saves, rollback, ETags, key hashing/revocation,
tenant boundaries, archive/restore, provider-error redaction and UI interactions.
Live PostgreSQL smoke checks additionally verify concurrent version allocation
and SDK reads. Actual provider availability needs a real provider key.
