# Sample Documents for LLM RAG Demo

This folder contains example documents you can upload to the app to try the 3-step flow.

- project_overview.md: What the demo is and why
- architecture.md: How the system works end-to-end
- api_reference.md: Endpoint details and sample requests
- prompt_recipes.md: Good questions to ask the model
- troubleshooting.txt: Common issues and fixes
- policies.md: Data handling and safety notes
- usage_scenarios.md: Example use-cases to try

Quick upload with curl (server must be running on http://localhost:5173):

curl -s -X POST http://localhost:5173/upload \
  -F 'files[]=@sample_docs/project_overview.md' \
  -F 'files[]=@sample_docs/architecture.md' \
  -F 'files[]=@sample_docs/api_reference.md' \
  -F 'files[]=@sample_docs/prompt_recipes.md' \
  -F 'files[]=@sample_docs/troubleshooting.txt' \
  -F 'files[]=@sample_docs/policies.md' \
  -F 'files[]=@sample_docs/usage_scenarios.md'

Then call /load, then /ask in the UI or via curl.
