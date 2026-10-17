# Product walkthrough screenshots

Captured from the actual LLMForge application on October 6, 2026. Every view has
dark/light desktop screenshots and a readable mobile crop. Images contain only
product regions: no account menu, organization identity, credentials, private
inputs or execution request identifiers. They are static public assets; no
application API or project permissions are used to display them.

- **Benchmarks:** two deliberately seeded examples, named Compact setup (A) and
  Deliberate setup (B), using fictitious `example/*` model IDs and twelve public
  classification examples. The measurements illustrate the comparison interface;
  they are not measurements of real models. The mobile crop shows exact-match
  accuracy and median latency from the same desktop view.
- **Prompts:** the existing Echo demo v2 text template, `Reply: {{query}}`, with
  the public input `Hello`. The playground runs in Demo mode and explicitly
  states that no model was called. The mobile crop focuses on the editor.
- **Evaluations:** a fresh provider-free run of Echo demo v2 against Greetings
  v1. `Reply: Hello` differs from the reference `Hello`, so exact-match checks
  fail. The mobile crop shows that concrete failure and its reason. No provider
  credits or keys were used.

Desktop images are 1153 × 783 pixels. Mobile dimensions are recorded beside the
preview metadata so the browser can reserve the correct aspect ratio. Recapture
these assets if the product layout changes; retain visible example disclosures.

Design and responsive behavior follow the single [frontend design guide](../../../DESIGN_SYSTEM.md); this file records asset provenance only.
