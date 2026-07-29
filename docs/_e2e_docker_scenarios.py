"""Docker-oriented E2E: growth analysis intents + vision RAG path (works without Bonsai)."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8000"


def post_json(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post_multipart(path, fields, files):
    import uuid

    boundary = "----NestlingBoundary" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += str(v).encode("utf-8") + b"\r\n"
    for name, (filename, content, ctype) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    health = get("/api/health")
    print("HEALTH", health)
    failed = 0

    child = post_json(
        "/api/children",
        {"name": "DockerBaby", "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    sid = post_json("/api/sessions", {"child_id": cid}).get("session_id")

    post_json(
        "/api/growth",
        {
            "child_id": cid,
            "sex": "male",
            "measure": "weight",
            "weeks": 40,
            "value": 3.2,
            "gestational_age_weeks": 39,
        },
    )

    turns = [
        ("show my child chart", lambda o: "growth" in (o.get("intents") or [])),
        (
            "is my child in right track ?",
            lambda o: "growth_analysis" in (o.get("intents") or [])
            and "I hear you" not in (o.get("reply") or ""),
        ),
        (
            "is he okey?",
            lambda o: "growth_analysis" in (o.get("intents") or [])
            and (
                "usual" in (o.get("reply") or "").lower()
                or "typical" in (o.get("reply") or "").lower()
                or "centile" in (o.get("reply") or "").lower()
                or "10th" in (o.get("reply") or "").lower()
            ),
        ),
        ("analyze", lambda o: "growth_analysis" in (o.get("intents") or [])),
    ]
    for msg, check in turns:
        out = post_json(
            "/api/chat",
            {"session_id": sid, "child_id": cid, "message": msg, "ui_lang": "en"},
        )
        ok = bool(check(out))
        print(("PASS" if ok else "FAIL"), msg, out.get("intents"), list(out.keys())[:12])
        print(" ", (out.get("reply") or "")[:220].replace("\n", " | "))
        if not ok:
            failed += 1
            print("  RAW_KEYS", sorted(out.keys()))
            if out.get("detail"):
                print("  DETAIL", out.get("detail"))

    img = ROOT / "docs" / "fixtures" / "rash_palm.png"
    if not img.exists():
        # copy from cursor assets if present
        alt = Path(
            r"C:\Users\mhf\.cursor\projects\c-Users-mhf-Desktop-pedriatrics\assets\c__Users_mhf_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-f13c9592-1430-443a-8b3d-de344c735057.png"
        )
        img.parent.mkdir(parents=True, exist_ok=True)
        if alt.exists():
            img.write_bytes(alt.read_bytes())
        else:
            # minimal 1x1 png
            import base64

            img.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                )
            )

    vision = post_multipart(
        "/api/chat/vision",
        {
            "message": "My baby has red spots on the palm. What should I watch for?",
            "session_id": sid,
            "child_id": cid,
            "ui_lang": "en",
        },
        {"image": (img.name, img.read_bytes(), "image/png")},
    )
    vreply = (vision.get("reply") or "").lower()
    vok = "pediatrician" in vreply or "clinician" in vreply or "rash" in vreply or "hand" in vreply
    print(("PASS" if vok else "FAIL"), "vision photo", vision.get("vision", {}).get("mode"))
    print(" ", (vision.get("reply") or "")[:280].replace("\n", " | "))
    if not vok:
        failed += 1

    print("FAILED", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
