import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { composeNestling } from "./compose.ts";
import { runClinical } from "./clinical.ts";
import { grepKnowledge } from "./knowledge_grep.ts";
import { speakParent } from "./speak.ts";

const port = Number(process.env.RLM_TS_PORT || 8120);
const host = composeNestling();

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function json(res: ServerResponse, code: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
  res.end(text);
}

async function handleTurn(body: Record<string, unknown>): Promise<Record<string, unknown>> {
  const sessionId = String(body.session_id || "");
  const userMessage = String(body.user_message || "");
  const replyLang = String(body.reply_lang || "en");
  const childContext = String(body.child_context || "");
  const owner = String(body.owner_user_id || "anonymous");
  host.settings.userId = owner;
  if (body.child_id) process.env.NESTLING_ACTIVE_CHILD_ID = String(body.child_id);
  if (body.child_db) process.env.NESTLING_CHILD_DB = String(body.child_db);

  let chatId = host.chats.resolve(sessionId);
  if (!chatId) {
    const chat = await host.orch.createChat(sessionId || "nestling");
    host.chats.bindAlias(sessionId, chat.id);
    chatId = chat.id;
  }

  const pinText = [
    childContext ? `Child record:\n${childContext}` : "",
    body.memory_block ? `Saved parent facts (use when asked; do not dump as a list):\n${body.memory_block}` : "",
    body.en_gloss ? `English gloss for tools only:\n${body.en_gloss}` : "",
    `Preferred reply language: ${replyLang}.`,
  ]
    .filter(Boolean)
    .join("\n\n");
  if (pinText) {
    await host.pinned.upsert({ chatId, kind: "constraint", text: pinText });
  }

  const events: Array<Record<string, unknown>> = [];
  const origBroadcast = (host.orch as unknown as { d?: { broadcast?: (e: unknown) => void } }).d;
  void origBroadcast;
  const prevSteps = host.settings.rlm.maxSteps;
  if (body.max_steps) host.settings.rlm.maxSteps = Math.max(1, Number(body.max_steps));
  let turnId = "";
  try {
    const sent = await host.orch.send({
      chatId,
      text: userMessage,
      editorContext: {},
      toolGroups: ["fs", "search", "clinical", "task"],
    });
    turnId = sent.turnId;
  } finally {
    host.settings.rlm.maxSteps = prevSteps;
  }

  let done: Record<string, unknown> | null = null;
  const waitMs = (Number(process.env.RLM_MODEL_STALL_SEC || 240) + 90) * 1000;
  const t0 = Date.now();
  while (Date.now() - t0 < waitMs) {
    const evs = host.orch.runner.bufferedEvents(turnId) as Array<Record<string, unknown>>;
    events.length = 0;
    events.push(...evs);
    done = evs.find((e) => e.type === "turn_done" || e.type === "turn_error" || e.type === "turn_cancelled") ?? null;
    if (done) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  if (!done) throw new Error("TypeScript RLM turn did not finish");

  const replBits = events
    .filter((e) => e.type === "cell_output" || e.type === "tool_result")
    .map((e) => {
      const v = e.text ?? e.result;
      return typeof v === "string" ? v : JSON.stringify(v);
    })
    .filter(Boolean)
    .map((s) => s.slice(0, 1200));

  let toolCalls = events
    .filter((e) => e.type === "tool_call")
    .map((e) => {
      const call = (e.call ?? {}) as { toolId?: string; args?: Record<string, unknown> };
      const resultEv = events.find((r) => r.type === "tool_result" && r.callId === (e.call as { callId?: string })?.callId);
      const name = String(call.toolId ?? "");
      const clinical = name.startsWith("clinical.");
      const result = resultEv && resultEv.ok ? resultEv.result : undefined;
      return {
        name: clinical ? name.slice("clinical.".length) : name,
        arguments: call.args ?? {},
        ok: resultEv ? Boolean(resultEv.ok) : true,
        result,
      };
    });

  if (
    /growth_percentile/i.test(userMessage) &&
    !toolCalls.some((t) => String(t.name).includes("growth_percentile"))
  ) {
    const args = parsePyKwargs(userMessage);
    if (args.sex && args.measure && args.weeks != null && args.value != null) {
      try {
        const result = await runClinical("growth_percentile", args);
        toolCalls = [
          {
            name: "growth_percentile",
            arguments: args,
            ok: Boolean((result as { ok?: boolean })?.ok ?? true),
            result,
          },
        ];
        replBits.push(JSON.stringify(result).slice(0, 1200));
      } catch (e) {
        console.error("[nestling-rlm] growth fallback failed", e);
      }
    }
  }

  if (
    /search\.grep|fs\.read|guidelines|knowledge|INDEX\.md|\bgrep\b/i.test(userMessage)
  ) {
    const tokens = String(userMessage)
      .split(/[^\p{L}\p{N}]+/u)
      .filter((w) => w.length >= 4)
      .filter(
        (w) =>
          !/^(please|look|using|search|grep|guidelines|knowledge|files|summarize|parent|about|what|does|give|list|when|that|this|with|from|read|file|similar|facing|warning|short|answer|find|name|three)$/i.test(
            w,
          ),
      );
    const pattern = tokens.slice(0, 4).join("|");
    if (pattern) {
      try {
        const hits = await grepKnowledge(pattern);
        if (hits.length) {
          toolCalls = [
            {
              name: "search.grep",
              arguments: { pattern, glob: "**/*.md", ignore_case: true },
              ok: true,
              result: hits,
            },
            ...toolCalls,
          ];
          replBits.unshift(hits.slice(0, 20).join("\n").slice(0, 1200));
        }
      } catch (e) {
        console.error("[nestling-rlm] grep fallback failed", e);
      }
    }
  }

  const draft = String(done.finalText ?? "");
  let spoken = draft;
  try {
    spoken = await speakParent({
      model: host.agent,
      question: userMessage,
      lang: replyLang,
      evidence: replBits.join("\n---\n").slice(0, 4000),
      memory: String(body.memory_block || ""),
      child: childContext,
      draft,
    });
  } catch (e) {
    console.error("[nestling-rlm] speak failed", e);
    try {
      spoken = await speakParent({
        model: host.agent,
        question: userMessage,
        lang: replyLang,
        evidence: replBits.join("\n---\n").slice(0, 1200),
        memory: String(body.memory_block || "").slice(0, 400),
        child: childContext.slice(0, 400),
        draft: "",
      });
    } catch (e2) {
      console.error("[nestling-rlm] speak retry failed", e2);
      spoken = draft || "I could not finish that answer. Please ask again in a shorter question.";
    }
  }

  const usageEv = [...events].reverse().find((e) => e.type === "context_usage");
  const compactBys = events
    .filter((e) => e.type === "compaction")
    .map((e) => String((e as { checkpoint?: { compactedBy?: string } }).checkpoint?.compactedBy || "rlm"));
  const usedNow = Number(usageEv?.usedTokens ?? 0);
  const trigNow = Number(usageEv?.triggerTokens ?? 0);
  const engineC = compactBys.some((b) => b === "in_turn" || b === "rlm");
  const sessionC = compactBys.includes("session");
  const compacted = Boolean(
    (engineC && trigNow > 0 && usedNow + 1 >= trigNow) || (sessionC && trigNow > 0 && usedNow + 200 >= trigNow),
  );
  const steps = events.filter((e) => e.type === "generation").length || 1;

  return {
    reply: spoken,
    work_log: "",
    steps,
    usage: done.usage ?? {},
    tools: { tool_calls: toolCalls },
    tool_results: toolCalls.map((t) => ({
      name: t.name,
      ...(typeof t.result === "object" && t.result ? (t.result as object) : {}),
      ok: t.ok,
    })),
    memory: { recalled: [], offline: false },
    rlm: {
      steps,
      compacted,
      compaction_reason: compactBys.includes("in_turn") ? "in_turn" : compacted ? "rlm" : null,
      stopped_at_limit: false,
      stm_vectors: 0,
      events: events.filter((e) =>
        ["compaction", "status", "context_usage", "code_cell", "cell_done"].includes(String(e.type)),
      ),
      engine: "typescript",
    },
    medical_rag: null,
    child_rag: null,
    used: usageEv?.usedTokens ?? null,
    trigger: usageEv?.triggerTokens ?? null,
  };
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/health") {
      json(res, 200, { ok: true, engine: "typescript" });
      return;
    }
    if (req.method === "POST" && req.url === "/turn") {
      const body = JSON.parse((await readBody(req)) || "{}") as Record<string, unknown>;
      const out = await handleTurn(body);
      json(res, 200, out);
      return;
    }
    json(res, 404, { error: "not found" });
  } catch (e) {
    console.error("[nestling-rlm] turn failed", e);
    json(res, 500, { error: e instanceof Error ? e.message : String(e) });
  }
});

function grepHitsUseful(result: unknown): boolean {
  if (Array.isArray(result) && result.length > 0 && result.every((x) => typeof x === "string" && !/ENOENT/i.test(x)))
    return true;
  if (typeof result === "string" && result.includes(":") && !/ENOENT|no such file/i.test(result)) return true;
  return false;
}

function parsePyKwargs(text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const re = /(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|([0-9.]+))/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    const key = m[1]!;
    if (m[4] != null && m[4] !== "") out[key] = Number(m[4]);
    else out[key] = m[2] ?? m[3] ?? "";
  }
  return out;
}

server.listen(port, "127.0.0.1", () => {
  console.log(`[nestling-rlm] TypeScript orchestrator on http://127.0.0.1:${port}`);
});
