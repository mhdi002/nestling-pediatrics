"""
Shared Nestling API contracts used by web UI and tests.
Keep in sync with app/api/routes.py and web/app.js.
"""

API_ROUTES = {
    "health": "GET /api/health",
    "children_create": "POST /api/children",
    "children_list": "GET /api/children",
    "children_get": "GET /api/children/{id}",
    "sessions": "POST /api/sessions",
    "chat": "POST /api/chat",
    "growth": "POST /api/growth",
    "asq_questions": "GET /api/asq/{age}/questions",
    "asq_score": "POST /api/asq/score",
    "mchat_questions": "GET /api/mchat/questions",
    "mchat_score": "POST /api/mchat/score",
    "overlays": "GET /api/overlays/{filename}",
}
