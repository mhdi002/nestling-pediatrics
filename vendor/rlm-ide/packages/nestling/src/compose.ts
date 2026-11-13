import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultSettings, type Settings } from "@rlm/contracts";
import { Orchestrator, TaskBoards, type ChatModelPort, type ChatRequest, type EmbeddingPort, type ToolImpl } from "@rlm/core";
import { OpenAICompatChat, OpenAICompatEmbed } from "../../adapters/src/llm/openai_compat.ts";
import { PythonKernelFactory } from "../../adapters/src/kernel/python_kernel.ts";
import { ConsoleLogger, FilePrompts, Workspace } from "../../adapters/src/misc.ts";
import { ProcessManager } from "../../adapters/src/tools/shell.ts";
import { ToolRegistry, createBuiltinTools } from "../../adapters/src/tools/builtin.ts";
import { buildCatalog, listFiles, nearestFiles } from "./catalog.ts";
import { createClinicalTools } from "./clinical.ts";
import { createLibraryTools } from "./library.ts";
import { MemoryConversationRepo, MemoryEdits, MemoryPinned, MemorySnapshots, MemoryTaskBoards, MemoryVectors } from "./stores.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const vendorRoot = path.resolve(here, "../../..");
const nestlingRoot = process.env.NESTLING_ROOT || path.resolve(vendorRoot, "../..");

export interface NestlingHost {
  orch: Orchestrator;
  chats: MemoryConversationRepo;
  pinned: MemoryPinned;
  settings: Settings;
  agent: ChatModelPort;
  boards: TaskBoards;
  /** Rebuilds the care-library catalog from the files on disk. */
  refreshPersona: () => void;
  contextLength: number;
  prompts: FilePrompts;
  tools: ToolRegistry;
  /** The text files of the care library, relative paths. */
  readable: () => string[];
}

/** Rough characters per token, used only to turn a share of the window into a character budget. */
const CHARS_PER_TOKEN = 3.2;
/** The most of the window one opened file may fill; longer files come back as outline plus matching sections. */
const LIBRARY_READ_SHARE = Number(process.env.RLM_LIBRARY_READ_SHARE || 0.16);
const CATALOG_CONTEXT_SHARE = Number(process.env.RLM_CATALOG_CONTEXT_SHARE || 0.1);

export function composeNestling(): NestlingHost {
  const llmUrl = (process.env.NESTLING_LLM_URL || "http://127.0.0.1:8100").replace(/\/+$/, "");
  const model = process.env.NESTLING_LLM_MODEL || "Qwen/Qwen3.5-2B";
  const ctx = Number(process.env.RLM_CONTEXT_LENGTH || 8192);
  const maxOut = Number(process.env.RLM_MAX_OUTPUT_TOKENS || 1024);
  const knowledge = process.env.NESTLING_KNOWLEDGE_DIR || path.join(nestlingRoot, "knowledge");
  const settings = defaultSettings();
  settings.userId = process.env.NESTLING_USER_ID || "nestling";
  // The persona block carries the care-library catalog; it is rebuilt from the files each turn
  // (see refreshPersona), so nothing about any topic is written here.
  settings.persona = "";
  settings.rlm.maxSteps = Number(process.env.RLM_MAX_STEPS || 16);
  settings.rlm.wakeupMinSec = Number(process.env.RLM_WAKEUP_MIN_SEC || 8);
  settings.rlm.selfReview = false;
  settings.rlm.requireEvidence = true;
  settings.rlm.pinFirstMessageAsGoal = false;
  settings.rlm.finishUnplanned = true;
  // What a finished turn read is kept as a short excerpt; a share of the window, so a larger model keeps more.
  settings.rlm.finishedTurnOutputChars = Math.floor(ctx * CHARS_PER_TOKEN * 0.015);
  settings.rlm.sandboxNoNetwork = process.env.RLM_SANDBOX_PREFER_BWRAP === "1";
  settings.rlm.compactTriggerRatio = Number(process.env.RLM_COMPACT_TRIGGER_RATIO || 0.62);
  settings.rlm.toolDocsMaxShare = Number(process.env.RLM_TOOL_DOCS_MAX_SHARE || 0.06);
  settings.compact.triggerRatio = Number(process.env.RLM_COMPACT_TRIGGER_RATIO || 0.62);
  settings.compact.historyShare = Number(process.env.RLM_HISTORY_SHARE || 0.48);
  settings.compact.keepRecentRatio = Number(process.env.RLM_KEEP_RECENT_RATIO || 0.28);
  settings.rlm.cellTimeoutSec = Number(process.env.RLM_CELL_TIMEOUT_SEC || 45);
  settings.rlm.modelStallSec = Number(process.env.RLM_MODEL_STALL_SEC || 240);
  settings.memory.enabled = false;
  settings.memory.sessionLimit = 0;
  settings.memory.userLimit = 0;
  settings.memory.alwaysRecallFacts = 0;
  settings.memory.longTermScope = "user";
  settings.tools.workspaceAllow = ["*"];
  settings.tools.approvalMode = "workspace";
  settings.tools.autoApplyEdits = true;
  settings.tools.actionCheck = false;

  const rawAgent = new OpenAICompatChat(model, {
    baseUrl: `${llmUrl}/v1`,
    apiKey: process.env.NESTLING_LLM_API_KEY || "EMPTY",
    model,
    stallMs: settings.rlm.modelStallSec * 1000,
  });
  const agent: ChatModelPort = {
    modelId: rawAgent.modelId,
    stream(req: ChatRequest) {
      // Diagnostics only: with RLM_DEBUG_PROMPT_DIR set, every request is written out, so what
      // the model was actually shown can be read instead of inferred.
      const dump = process.env.RLM_DEBUG_PROMPT_DIR;
      if (dump) {
        try {
          mkdirSync(dump, { recursive: true });
          writeFileSync(path.join(dump, `req_${Date.now()}_${Math.random().toString(36).slice(2, 6)}.json`), JSON.stringify(req.messages, null, 1));
        } catch {
          /* diagnostics must never break a turn */
        }
      }
      return rawAgent.stream({ ...req, enableThinking: false });
    },
  };
  let embedder: EmbeddingPort | null = null;
  const embedUrl = (process.env.RLM_EMBED_URL || "").replace(/\/+$/, "");
  if (embedUrl) {
    embedder = new OpenAICompatEmbed(process.env.RLM_EMBED_MODEL || "bge-m3", {
      baseUrl: `${embedUrl}/v1`,
      apiKey: "EMPTY",
      model: process.env.RLM_EMBED_MODEL || "bge-m3",
    });
  }

  const pm = new ProcessManager();
  const workspace = new Workspace(pm);
  workspace.set(knowledge);
  // Where conversations, compacted transcripts and task boards are kept across restarts.
  // Unset (tests), they live in memory only.
  const stateDir = process.env.RLM_STATE_DIR || undefined;
  const chats = new MemoryConversationRepo(stateDir);
  const pinned = new MemoryPinned(stateDir);
  const log = new ConsoleLogger("[nestling-rlm]");
  // Nestling's own wording first, then the product's parent-facing prompts, then the stock ones.
  const prompts = new FilePrompts([path.join(nestlingRoot, "prompts", "nestling"), path.join(nestlingRoot, "prompts"), path.join(vendorRoot, "prompts")]);
  const clock = { now: () => Date.now() };
  const boards = new TaskBoards({
    store: new MemoryTaskBoards(stateDir),
    clock,
    settings: () => settings.rlm,
    log,
  });
  const tools = new ToolRegistry([
    ...suggestOnMissingFile(
      knowledge,
      broadenGrep(
      createBuiltinTools({
        processes: pm,
        shellTimeoutSec: () => settings.tools.shellTimeoutSec,
        cellTimeoutSec: () => settings.rlm.cellTimeoutSec,
        editorState: () => ({}),
        searchSession: async () => [],
      })// The care library is read-only to the agent. With the write tools exposed, a parent saying
        // "remember she is allergic to peanuts" made the agent write that into the library itself,
        // where every other family's searches would then find it.
        .filter((t) => (t.descriptor.group === "fs" || t.descriptor.group === "search") && !t.descriptor.sideEffects),
      ),
    ),
    ...createLibraryTools(knowledge, () => Math.floor(ctx * CHARS_PER_TOKEN * LIBRARY_READ_SHARE)),
    ...createClinicalTools(),
    ...boards.tools(),
  ]);

  const orch = new Orchestrator({
    settings: { get: () => settings },
    updateSettings: (p) => Object.assign(settings, p),
    router: {
      chat: async () => agent,
      embedder: async () => embedder,
      modelIdFor: () => model,
    },
    catalog: {
      limits: async () => ({ contextLength: ctx, maxOutputTokens: maxOut }),
    },
    conversations: chats,
    memoryStore: new MemoryVectors(),
    pinned,
    snapshots: new MemorySnapshots(stateDir),
    edits: new MemoryEdits(),
    kernels: new PythonKernelFactory({
      workerScript: path.join(nestlingRoot, "assistant/rlm/worker/worker.py"),
      pythonCandidates: [process.env.RLM_PYTHON ?? "", "python3", "python"],
      sandboxNoNetwork: () => settings.rlm.sandboxNoNetwork,
      memoryMb: () => settings.rlm.kernelMemoryMb,
      log,
    }),
    tools,
    prompts,
    workspace,
    clock,
    log,
    status: { memory() {}, kernel() {} },
    broadcast: () => undefined,
    readMention: async () => "",
    boards,
  });

  const refreshPersona = () => {
    // A share of the window, so a larger model sees more of each file's contents listed.
    settings.persona = buildCatalog(knowledge, Math.floor(ctx * CHARS_PER_TOKEN * CATALOG_CONTEXT_SHARE));
  };
  refreshPersona();
  const readable = () => listFiles(knowledge).filter((f) => /\.(md|txt)$/i.test(f) && f.toUpperCase() !== "INDEX.MD");
  return { orch, chats, pinned, settings, agent, boards, refreshPersona, contextLength: ctx, prompts, tools, readable, knowledgeRoot: knowledge, libraryReadChars: Math.floor(ctx * CHARS_PER_TOKEN * LIBRARY_READ_SHARE) };
}

/** A path that does not exist is answered with the real paths nearest to it. */
function suggestOnMissingFile(root: string, tools: ToolImpl[]): ToolImpl[] {
  return tools.map((t) => {
    if (t.descriptor.group !== "fs" || !("path" in (t.descriptor.parameters.properties ?? {}))) return t;
    return {
      ...t,
      async run(args, ctx) {
        try {
          return await t.run(args, ctx);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          if (!/ENOENT|no such file/i.test(msg)) throw e;
          const near = nearestFiles(root, String(args.path ?? ""));
          throw new Error(`no such file: ${String(args.path ?? "")}.${near.length ? ` Closest existing files: ${near.join(", ")}` : ""}`);
        }
      },
    };
  });
}

function broadenGrep(tools: ToolImpl[]): ToolImpl[] {
  return tools.map((t) => {
    if (t.descriptor.id !== "search.grep") return t;
    return {
      ...t,
      async run(args, ctx) {
        const glob = String(args.glob ?? "");
        const normalized = {
          ...args,
          glob:
            !glob || glob === "*" || glob === "*.*"
              ? undefined
              : glob.startsWith("**/")
                ? glob
                : glob.includes("/")
                  ? glob
                  : `**/${glob}`,
        };
        const tokens = String(args.pattern ?? "")
          .split(/[^\p{L}\p{N}]+/u)
          .filter((w) => w.length >= 3);
        const first = await t.run(normalized, ctx);
        const max = Number(args.max_results ?? 25);
        let raw: unknown[] = Array.isArray(first) ? first : [];
        if (!raw.length) {
          const seen = new Set<string>();
          const out: string[] = [];
          for (const tok of tokens) {
            const more = await t.run({ ...normalized, pattern: tok }, ctx);
            if (!Array.isArray(more)) continue;
            for (const line of more) {
              const s = String(line);
              if (seen.has(s)) continue;
              seen.add(s);
              out.push(s);
              if (out.length >= max) break;
            }
            if (out.length >= max) break;
          }
          raw = out.length ? out : raw;
        }
        return rankHits(raw, tokens).slice(0, max);
      },
    };
  });
}

function rankHits(lines: unknown[], tokens: string[]): unknown[] {
  const scored = lines.map((line, i) => {
    const s = String(line).toLowerCase();
    const path = s.split(":")[0] ?? "";
    let sc = 0;
    for (const tok of tokens) {
      const t = tok.toLowerCase();
      if (path.includes(t)) sc += 8;
      else if (s.includes(t)) sc += 2;
    }
    return { line, i, sc };
  });
  scored.sort((a, b) => b.sc - a.sc || a.i - b.i);
  return scored.map((x) => x.line);
}
