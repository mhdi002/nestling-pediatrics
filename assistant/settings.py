"""Central typed settings — all NESTLING_* / LLM env vars live here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths (overrideable for tests and containers)
    nestling_root: Path = ROOT
    nestling_data_dir: Path | None = None
    nestling_child_db: Path | None = None
    nestling_chat_db: Path | None = None

    # Models
    nestling_load_models: bool = False
    nestling_use_llm: bool = True
    # Empty by default: the sidecar is opt-in, and an unreachable default URL
    # would make every request pay a failed connection probe.
    nestling_llm_url: str = ""
    nestling_vision_llm_url: str = ""
    # Qwen3.5-2B. The agent has to choose, on its own, which care-library file
    # answers a parent's question and when a clinical tool is needed. Measured
    # on 18 natural parent questions with the reply constrained to the files
    # that exist: Qwen3.5-2B picked the right file 16 times, MiniCPM5-1B about
    # 3 (it asked for the growth tool in reply to "thanks"). A 1B model wrote
    # fluent answers from retrieved text well enough; it cannot drive the RLM
    # loop, so the smallest model that can is the default. ~4.3 GB in bf16.
    nestling_llm_model: str = "Qwen/Qwen3.5-2B"
    # Qwen3.5 is multimodal, so the same sidecar serves photo turns when
    # images are enabled on it (VLLM_LIMIT_MM_IMAGE, sized by deploy.sh). The
    # capability probe still decides at run time: a sidecar that refuses an
    # image degrades the turn to caption-plus-guidance rather than pretending
    # to have looked.
    nestling_vision_model: str = "Qwen/Qwen3.5-2B"
    nestling_tool_max_new_tokens: int = 512
    nestling_llm_gguf: str = ""
    nestling_vision_gguf: str = ""
    nestling_vision_mmproj: str = ""
    nestling_embedding_model: str = "BAAI/bge-m3"

    # Adaptive chat concurrency. The sidecar batches several turns at once
    # (VLLM_MAX_NUM_SEQS, derived from the GPU by scripts/size_llm.py); these
    # let the app widen to match it instead of capping at the AnyIO default of
    # 40 threads, and shed load fast when even that is exceeded. See
    # app/concurrency.py for how they resolve.
    #
    # 0 means "adapt": the effective limit is taken from VLLM_MAX_NUM_SEQS in
    # the environment, or the floor below when that is absent. Set > 0 to pin
    # it regardless of the sidecar.
    nestling_llm_max_concurrency: int = 0
    # Used only when nothing sized the sidecar (a hand-started stack, or
    # app-only generation): a few parents at once, not a number that melts an
    # un-sized host.
    nestling_chat_concurrency_floor: int = 8
    # How many turns may wait for a slot before load is shed with 503. 0 =
    # derive (half the concurrency, bounded). More just moves the timeout
    # pileup behind the gate instead of removing it.
    nestling_chat_queue_slack: int = 0
    # How long a queued turn waits for a slot before it is shed. Kept below the
    # proxy's chat timeout so the client gets a fast, honest 503 rather than a
    # 504 from the back of the queue. Doubles as the Retry-After hint.
    nestling_chat_acquire_timeout_s: float = 20.0
    # AnyIO worker-thread pool size. 0 = derive from the concurrency above
    # (enough to hold every admitted and queued turn plus a reserve for the
    # rest of the API). Set > 0 to pin it.
    nestling_worker_threads: int = 0

    # LLM transport
    # MUST stay below the load balancer's chat timeout
    # (NESTLING_LB_CHAT_TIMEOUT, 120s by default in docker-compose.yml), with
    # headroom for retrieval, tools and translation. At 180s the proxy always
    # gave up first, so a slow generation surfaced as a hung request and a 504
    # instead of degrading to the extractive answer the app already had ready.
    # tests/test_timeout_hierarchy.py asserts the relationship still holds.
    nestling_llm_timeout: float = 60.0
    nestling_llm_probe_timeout: float = 1.5
    # The vision probe sends a real one-pixel image and waits for a token, so
    # it is a generation, not a ping. At four times the ping timeout it kept
    # timing out on a busy GPU and the app reported 'vision model is not
    # ready' for a sidecar that was serving images fine. It runs once per
    # process and the verdict is cached, so it can afford to wait.
    nestling_vision_probe_timeout: float = 45.0
    nestling_llm_ready_cache_seconds: float = 5.0
    nestling_llm_error_detail_chars: int = 400

    # API / security
    nestling_cors_origins: str = "*"
    nestling_api_key: str | None = None
    # Interactive login. When a username and password hash are configured the
    # UI shows a sign-in form and the API accepts the resulting bearer token.
    # The API key above keeps working for programmatic clients.
    nestling_auth_username: str | None = None
    # PBKDF2-HMAC-SHA256, formatted "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>".
    nestling_auth_password_hash: str | None = None
    # Signing key for session tokens. Generated by deploy.sh; if unset, a random
    # one is created per process (which invalidates sessions on restart).
    nestling_auth_secret: str | None = None
    nestling_session_ttl_hours: int = 12
    nestling_max_upload_bytes: int = 8_000_000
    nestling_max_image_pixels: int = 40_000_000
    nestling_allowed_upload_types: str = "image/png,image/jpeg,image/webp,image/gif,image/bmp"
    nestling_session_list_limit: int = 40
    nestling_session_list_max_limit: int = 200
    nestling_dossier_overlay_limit: int = 12
    nestling_tool_overlay_limit: int = 8
    nestling_session_title_chars: int = 60
    nestling_stream_chunk_chars: int = 24

    # Runtime translation (deep-translator has no built-in timeout)
    nestling_translate_timeout: float = 6.0
    nestling_translate_workers: int = 4
    # Shortest run of Latin letters that marks a block as leaked English needing
    # translation when forcing a reply into Persian (ensure_pure_lang).
    nestling_translate_min_latin_run: int = 4

    # Mean days per month, for deriving an age from a date of birth.
    nestling_days_per_month: float = 30.4375

    # Chat memory
    nestling_history_window: int = 12
    nestling_summary_trigger_turns: int = 16
    nestling_history_response_limit: int = 200
    nestling_summary_max_chars: int = 4000
    nestling_summary_fold_max_chars: int = 3500
    nestling_summary_fold_turns: int = 24
    nestling_summary_turn_chars: int = 180
    nestling_memory_recent_chars: int = 1200
    nestling_medical_query_chars: int = 500
    # Shortest child name we will look for inside a free-text message; one-letter
    # names would match almost any sentence.
    nestling_child_name_min_chars: int = 2

    # Long-term, per-CHILD memory. Chat sessions are disposable; a child's
    # developmental thread is not, so salient turns are written to the child's
    # event timeline and replayed into every later session about that child.
    nestling_child_memory_enabled: bool = True
    nestling_child_memory_kind: str = "chat_note"
    # Which intents make a turn worth remembering beyond the session. Small
    # talk and slot corrections are deliberately excluded — they would drown
    # the digest in noise within a few sessions.
    nestling_child_memory_intents: str = "medical,screening,growth_analysis,growth"
    nestling_child_memory_events: int = 8
    nestling_child_memory_event_chars: int = 200
    nestling_child_memory_note_chars: int = 240
    nestling_child_memory_growth: int = 3
    nestling_child_memory_screenings: int = 2
    nestling_child_memory_max_chars: int = 1500

    # LLM generation defaults
    llm_temperature: float = 0.7
    llm_top_p: float = 0.95
    llm_max_tokens_default: int = 512
    llm_max_tokens_vision: int = 700
    llm_max_tokens_chat: int = 320
    llm_rag_temperature: float = 0.75
    llm_vision_temperature: float = 0.2
    llm_vision_top_p: float = 0.9
    # Qwen3 reasons before answering. When the budget runs out mid-think the
    # reply comes back EMPTY and the app serves the extractive fallback -- on
    # every turn, for as long as the model is served that way.
    #
    # OFF by default, because it was measured and did not work. Against a local
    # qwen3-vl:4b the retry escalated 900 -> 2700 -> 4096 tokens and the model
    # still produced no answer, reasoning for 17,980 characters: it was not
    # short of room, it simply does not stop. All the retry bought was three
    # times the latency on a turn a parent was waiting for.
    #
    # The real fix for such a model is to stop it reasoning -- see
    # nestling_llm_chat_template_kwargs -- or to serve one that does not. Raise
    # this above 1.0 only for a model that genuinely needs a little more room,
    # where one retry ends the problem rather than repeating it.
    llm_reasoning_retry_multiplier: float = 1.0
    llm_reasoning_retry_max_tokens: int = 4096
    # Extra arguments for the server's chat template, as a JSON object.
    # {"enable_thinking": false} is the documented way to stop Qwen3 reasoning,
    # and it is why the deployed vLLM answers rather than thinking forever --
    # Ollama's OpenAI shim ignores it, which is how that failure was found.
    # Configurable rather than hardcoded so a server that rejects the field can
    # be given "{}" instead of failing every request.
    nestling_llm_chat_template_kwargs: str = '{"enable_thinking": false}'
    llm_prompt_context_chars: int = 1800
    llm_prompt_query_chars: int = 400
    llm_answer_max_chars: int = 1200

    # When no intent rule matches, a message at least this many words long is
    # treated as a real question and sent to retrieval rather than answered
    # with the generic "I'm listening" menu. Keyword lists cannot enumerate
    # every way a parent describes a symptom.
    nestling_retrieval_fallback_min_words: int = 3
    nestling_retrieval_fallback_min_chars: int = 8

    # ---- Memory system (assistant/memory/) --------------------------------
    # Where the native memory backend keeps facts and episodes. Defaults
    # beside the chat database so one volume holds all conversational state.
    nestling_memory_db: str = ""
    # Which backend serves semantic + episodic memory. Names are resolved
    # against config/memory.yaml; an unavailable backend falls back to
    # "native", which needs no network, no graph database and no embeddings.
    nestling_memory_backend: str = "native"
    # Consolidation: after this many episodic turns in a session, distil the
    # durable facts out of them into semantic memory and mark them folded.
    # Low enough that a long chat does not outrun the context window before
    # anything is saved, high enough not to pay for extraction every turn.
    nestling_memory_consolidate_every: int = 8
    # A consolidated claim is inferred, not stated, so it ranks below anything
    # the parent said outright when the context budget is tight.
    nestling_memory_inferred_confidence: float = 0.6
    # Longest a single remembered line may be when rendered into the prompt.
    nestling_memory_line_chars: int = 220

    # Graphiti backend. Only read when nestling_memory_backend selects it.
    # Empty URI means the graph is not configured, which is not an error --
    # it is how the native backend stays the default.
    nestling_memory_graph_uri: str = ""
    nestling_memory_graph_user: str = ""
    nestling_memory_graph_password: str = ""
    # Embeddings for the graph go through the same OpenAI-compatible endpoint
    # as generation, so nothing is fetched from huggingface.co at runtime.
    nestling_memory_embedding_model: str = "Qwen/Qwen3.5-4B"
    # vLLM ignores the key, but the OpenAI client requires one to be set.
    nestling_llm_api_key: str = ""

    # Context budget. The four memory kinds compete for one prompt, and a
    # prompt stuffed with everything is what made the model answer about
    # vaccines when asked about an ulcer. Shares are fractions of
    # llm_prompt_context_chars and are normalised, so they need not sum to 1.
    nestling_memory_share_procedural: float = 0.15
    nestling_memory_share_semantic: float = 0.35
    nestling_memory_share_episodic: float = 0.30
    nestling_memory_share_working: float = 0.20

    # Deadline for lazily loading the embedding model. It is fetched on the
    # first chat request, so an unreachable huggingface.co would otherwise hang
    # that request indefinitely instead of falling back to BM25. Generous
    # enough to load cached weights from disk, far below the proxy timeout.
    nestling_dense_load_timeout: float = 20.0

    # Chart rendering
    chart_dpi: int = 140
    chart_figsize_w: float = 10.0
    chart_figsize_h: float = 6.0
    chart_curve_step_months: float = 0.5
    chart_curve_step_weeks: float = 0.5

    # ---- RLM orchestrator (assistant/rlm/) --------------------------------
    # Chat turns go through the RLM REPL engine. Knowledge is files under
    # knowledge/; embeddings are only for user long-term memory.
    nestling_use_rlm: bool = True
    nestling_knowledge_dir: str = ""
    rlm_ts_url: str = "http://127.0.0.1:8120"
    rlm_ts_autostart: bool = True
    rlm_prompts_dir: str = ""
    rlm_worker_script: str = ""
    rlm_python: str = ""
    rlm_chars_per_token: float = 3.2
    rlm_message_overhead_tokens: int = 6
    rlm_context_length: int = 32768
    rlm_max_output_tokens: int = 2048
    # Compact (session / in-turn)
    rlm_compact_history_share: float = 0.6
    rlm_compact_trigger_ratio: float = 0.8
    rlm_compact_trigger_ratio_in_turn: float = 0.9
    rlm_compact_keep_recent_ratio: float = 0.35
    rlm_compact_max_output_ratio: float = 0.25
    rlm_compact_max_rows_per_1k_history: float = 4.0
    rlm_compact_newest_limit_ratio: float = 0.6
    rlm_compact_summary_max_share: float = 0.5
    # RLM loop
    rlm_max_steps: int = 24
    rlm_step_warn_remaining: int = 5
    rlm_model_stall_sec: float = 300.0
    rlm_thinking_max_share: float = 0.6
    rlm_thinking_max_tokens: int = 2048
    rlm_thinking_max_sec: int = 120
    rlm_repeat_limit: int = 4
    rlm_max_cuts_in_a_row: int = 3
    rlm_max_identical_cells: int = 2
    # Do not fold history until this many user turns exist, unless the window
    # would otherwise overflow (false compaction on a short thread).
    rlm_compact_min_user_turns: int = 3
    rlm_empty_search_abort: int = 2
    rlm_grep_topic_stopwords: str = (
        "please,look,lookup,guidance,month,months,knowledge,files,search,grep,"
        "read,file,summarize,parent,using,about,what,does,guidelines,short,"
        "answer,sentence,sentences,give,keep,find,from,with,that,this,exist,"
        "name,three,hello,using,english"
    )
    rlm_cell_timeout_sec: int = 60
    # Executable markdown fence labels (comma-separated). Small models often
    # emit ```python instead of ```repl; both run as REPL cells when listed.
    rlm_cell_fence_labels: str = "repl,python,py"
    # Namespace prefixes that may be recovered as REPL cells when a small
    # model describes a tool call in prose instead of a fence.
    rlm_call_prefixes: str = ""
    rlm_temperature: float = 0.2
    rlm_cell_output_limit_chars: int = 16000
    rlm_cell_output_max_share: float = 0.25
    rlm_tool_docs_max_share: float = 0.25
    rlm_old_output_keep_chars: int = 1500
    rlm_max_subagent_depth: int = 0
    rlm_kernel_memory_mb: int = 2048
    rlm_sandbox_no_network: bool = True
    rlm_sandbox_prefer_bwrap: bool = True
    rlm_kernel_ready_timeout_sec: float = 20.0
    rlm_rewrite_min_lines: int = 40
    rlm_rewrite_max_changed: int = 8
    rlm_empty_stream_retries: int = 2
    rlm_empty_stream_backoff_sec: float = 0.5
    rlm_guard_window_chars: int = 12000
    rlm_guard_check_every_chars: int = 256
    rlm_guard_needle_chars: int = 24
    rlm_fs_read_max_chars: int = 60000
    rlm_fs_glob_max_results: int = 500
    rlm_search_max_results: int = 100
    rlm_grep_file_max_bytes: int = 2_000_000
    rlm_pdf_max_pages: int = 8
    rlm_tool_allowlist: str = "fs,search,clinical"
    # LTM only (STM vector upserts are disabled in code)
    rlm_memory_enabled: bool = True
    rlm_memory_db: str = ""
    rlm_embed_url: str = ""
    rlm_embed_model: str = ""
    rlm_memory_min_score: float = 0.55
    rlm_memory_user_limit: int = 6
    rlm_memory_always_recall_facts: int = 12
    rlm_memory_prompt_max_chars: int = 4000
    rlm_memory_fact_max_adds_per_turn: int = 3
    rlm_memory_fact_max_chars: int = 200
    rlm_memory_fact_min_user_chars: int = 12
    rlm_memory_fact_require_user_grounding: bool = True
    rlm_memory_long_term_scope: str = "workspace"
    rlm_memory_session_limit: int = 0  # 0 = no short-term vector recall

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.nestling_cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def child_memory_intents(self) -> frozenset[str]:
        raw = (self.nestling_child_memory_intents or "").strip()
        return frozenset(i.strip().lower() for i in raw.split(",") if i.strip())

    @property
    def allowed_upload_types(self) -> frozenset[str]:
        raw = (self.nestling_allowed_upload_types or "").strip()
        return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())

@lru_cache
def get_settings() -> Settings:
    return Settings()

def reset_settings() -> None:
    """Drop the cached Settings so a changed environment is picked up (tests)."""
    get_settings.cache_clear()
