from pathlib import Path

CHILD_OLD = """    def close(self):
        self.conn.close()
"""
CHILD_NEW = """    def close(self):
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
"""

CHAT_OLD = """    def close(self) -> None:
        self.conn.close()
"""
CHAT_NEW = """    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
"""

p = Path("assistant/memory/child_db.py")
t = p.read_text(encoding="utf-8")
if CHILD_OLD not in t:
    raise SystemExit("child_db close not found")
p.write_text(t.replace(CHILD_OLD, CHILD_NEW), encoding="utf-8")
print("patched child_db")

p = Path("assistant/memory/chat_memory.py")
t = p.read_text(encoding="utf-8")
if CHAT_OLD not in t:
    raise SystemExit("chat_memory close not found")
p.write_text(t.replace(CHAT_OLD, CHAT_NEW), encoding="utf-8")
print("patched chat_memory")
