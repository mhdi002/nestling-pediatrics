"""
Shared Nestling API contracts used by web UI and tests.
Keep in sync with app/api/routes.py and web/app.js.
"""

API_ROUTES = {
    "health": "GET /api/health",
    "children_create": "POST /api/children",
    "children_list": "GET /api/children",
    "children_get": "GET /api/children/{id}",
    "children_dossier": "GET /api/children/{id}/dossier",
    "sessions_create": "POST /api/sessions",
    "sessions_list": "GET /api/sessions",
    "sessions_get": "GET /api/sessions/{id}",
    "chat": "POST /api/chat",
    "chat_stream": "POST /api/chat/stream",
    "chat_vision": "POST /api/chat/vision",
    "growth": "POST /api/growth",
    "growth_curves": "GET /api/growth/curves",
    "asq_questions": "GET /api/asq/{age}/questions",
    "asq_score": "POST /api/asq/score",
    "mchat_questions": "GET /api/mchat/questions",
    "mchat_score": "POST /api/mchat/score",
    "overlays": "GET /api/overlays/{filename}",
}
