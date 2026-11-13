#!/usr/bin/env python3
"""
Nestling parent assistant.

Every chat turn runs on the TypeScript RLM orchestrator (vendor/rlm-ide): a
model works through a sandboxed Python REPL, reads the care library under
knowledge/ as files, and calls the deterministic clinical tools for every
growth and screening number. assistant/rlm/ hosts it from the FastAPI app and
keeps long-term memory; assistant/tools/ holds the clinical tools.
"""

__version__ = "0.1.0"
