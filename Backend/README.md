# Backend

FastAPI backend for P&ID upload, preprocessing, and OpenRouter-powered detection.

## Environment

Create a `.env` file in this folder based on `.env.example`.

Required:

- `OPENROUTER_API_KEY`

Optional overrides:

- `OPENROUTER_BASE_URL`
- `OPENROUTER_QWEN_MODEL`
- `OPENROUTER_CLAUDE_MODEL`
- `OPENROUTER_QWEN_MAX_TOKENS`
- `OPENROUTER_CLAUDE_MAX_TOKENS`
- `OPENROUTER_SITE_URL`
- `OPENROUTER_APP_NAME`

Model roles:

- Qwen: primary P&ID component detector
- Claude: verification and validation pass over Qwen candidates

## Run

```powershell
cd c:\Users\aadeshpande\Desktop\Sarla-Project\Backend
. .\myenv\Scripts\Activate.ps1
uvicorn Backend.main:app --reload
```

If PowerShell blocks script activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\myenv\Scripts\Activate.ps1
```
