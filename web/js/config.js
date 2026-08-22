// Frontend config — API_BASE points at the deployed FastAPI service on Render.
// Local dev: run `uvicorn api.main:app --port 8000` and use
// "http://localhost:8000" instead.
window.API_BASE = window.API_BASE || "https://credit-default-predictor-k5am.onrender.com";
