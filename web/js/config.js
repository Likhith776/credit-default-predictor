// Frontend config — set API_BASE to your deployed FastAPI origin
// (e.g. "https://credit-default-api.onrender.com") before deploying.
// Local dev default points at `uvicorn api.main:app --port 8000`.
window.API_BASE = window.API_BASE || "http://localhost:8000";
