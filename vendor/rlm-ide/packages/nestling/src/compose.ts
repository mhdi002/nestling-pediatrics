import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultSettings, type Settings } from "@rlm/contracts";
import { Orchestrator, TaskBoards, type ChatModelPort, type ChatRequest, type EmbeddingPort, type ToolImpl } from "@rlm/core";
import { OpenAICompatChat, OpenAICompatEmbed } from "../../adapters/src/llm/openai_compat.ts";
import { PythonKernelFactory } from "../../adapters/src/kernel/python_kernel.ts";
import { ConsoleLogger, FilePrompts, Workspace } from "../../adapters/src/misc.ts";
import { ProcessManager } from "../../adapters/src/tools/shell.ts";
import { ToolRegistry, createBuiltinTools } from "../../adapters/src/tools/builtin.ts";
import { createClinicalTools } from "./clinical.ts";
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
}

export function composeNestling(): NestlingHost {
  const llmUrl = (process.env.NESTLING_LLM_URL || "http://127.0.0.1:8100").replace(/\/+$/, "");
  const model = process.env.NESTLING_LLM_MODEL || "openbmb/MiniCPM5-1B";
  const ctx = Number(process.env.RLM_CONTEXT_LENGTH || 8192);
  const maxOut = Number(process.env.RLM_MAX_OUTPUT_TOKENS || 1024);
  const knowledge = process.env.NESTLING_KNOWLEDGE_DIR || path.join(nestlingRoot, "knowledge");
  const settings = defaultSettings();
  settings.userId = process.env.NESTLING_USER_ID || "nestling";
  settings.persona = [
    "You are Nestling, a parent-facing pediatric assistant running the RLM IDE engine.",
    "Knowledge is files in this workspace (INDEX.md, guidelines/, charts/). Do not prefix paths with knowledge/.",
    "When the parent asks to look something up or compute a growth number, your FIRST reply MUST be only one ```repl cell (no prose) that calls search.grep, fs.read_file, or clinical.growth_percentile.",
    "Example:",
    "```repl",
    'print(search.grep(pattern="iron", glob="**/*.md", ignore_case=True, max_results=25))',
    "```",
    "After [REPL output], answer the parent in 2–6 fluent sentences. No code, headings, or file dumps.",
    "If the parent writes Persian, the final answer is Persian only.",
    "Answer only the current question; do not stick to an earlier topic.",
  ].join("\n");
  settings.rlm.maxSteps = Number(process.env.RLM_MAX_STEPS || 16);
  settings.rlm.wakeupMinSec = Number(process.env.RLM_WAKEUP_MIN_SEC || 8);
  settings.rlm.selfReview = false;
  settings.rlm.sandboxNoNetwork = process.env.RLM_SANDBOX_PREFER_BWRAP === "1";
  settings.rlm.compactTriggerRatio = Number(process.env.RLM_COMPACT_TRIGGER_RATIO || 0.62);
  settings.rlm.toolDocsMaxShare = Number(process.env.RLM_TOOL_DOCS_MAX_SHARE || 0.18);
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
  const chats = new MemoryConversationRepo();
  const pinned = new MemoryPinned();
  const log = new ConsoleLogger("[nestling-rlm]");
  const prompts = new FilePrompts(path.join(vendorRoot, "prompts"));
  const clock = { now: () => Date.now() };
  const boards = new TaskBoards({
    store: new MemoryTaskBoards(),
    clock,
    settings: () => settings.rlm,
    log,
  });
  const tools = new ToolRegistry([
    ...broadenGrep(
      createBuiltinTools({
        processes: pm,
        shellTimeoutSec: () => settings.tools.shellTimeoutSec,
        cellTimeoutSec: () => settings.rlm.cellTimeoutSec,
        editorState: () => ({}),
        searchSession: async () => [],
      }).filter((t) => t.descriptor.group === "fs" || t.descriptor.group === "search"),
    ),
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
    snapshots: new MemorySnapshots(),
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

  return { orch, chats, pinned, settings, agent };
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
