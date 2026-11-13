import { spawn, spawnSync } from "node:child_process";
import path from "node:path";
import type { ToolImpl } from "@rlm/core";

/** What a turn knows about whose child it is for. Keyed by chat, never held in process-wide state:
 * two parents chatting at once must not see each other's child through a shared variable. */
export interface ChildScope {
  childId: string;
  childDb: string;
  /** The child's age today, when the record has a date of birth. */
  ageMonths?: number;
}
const scopes = new Map<string, ChildScope>();

export function setChildScope(chatId: string, scope: ChildScope): void {
  scopes.set(chatId, scope);
}

export function childScope(chatId: string): ChildScope | undefined {
  return scopes.get(chatId);
}

function root(): string {
  return process.env.NESTLING_ROOT || path.resolve(import.meta.dirname, "../../../../..");
}

function python(): string {
  return process.env.RLM_PYTHON || process.env.NESTLING_PYTHON || "python";
}

function script(): string {
  return path.join(root(), "assistant/rlm/clinical_dispatch.py");
}

interface ToolSpec {
  name: string;
  description?: string;
  parameters?: { type?: string; properties?: Record<string, unknown>; required?: string[] };
}

/** The clinical tools as Python declares them. Asked once at start-up, so the model is shown each
 * tool's real parameters and description; a single generic schema for every tool is what made the
 * model call get_child_summary(measure="iron"). */
function describeTools(): ToolSpec[] {
  const r = spawnSync(python(), [script()], { input: JSON.stringify({ describe: true }), encoding: "utf8", env: process.env });
  if (r.status !== 0) throw new Error(`clinical tools could not be described: ${(r.stderr || r.error?.message || "").slice(0, 400)}`);
  const specs = JSON.parse(r.stdout) as ToolSpec[];
  if (!Array.isArray(specs) || !specs.length) throw new Error("clinical dispatch described no tools");
  return specs;
}

function runDispatch(name: string, args: Record<string, unknown>, scope: ChildScope | undefined): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = spawn(python(), [script()], { env: process.env, stdio: ["pipe", "pipe", "pipe"] });
    let out = "";
    let err = "";
    child.stdout.on("data", (d) => {
      out += String(d);
    });
    child.stderr.on("data", (d) => {
      err += String(d);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(err.trim() || `clinical dispatch exited ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(out));
      } catch {
        reject(new Error(`clinical dispatch returned non-JSON: ${out.slice(0, 400)}`));
      }
    });
    child.stdin.end(JSON.stringify({ name, args, child_id: scope?.childId ?? "", child_db: scope?.childDb ?? "" }));
  });
}

export function createClinicalTools(): ToolImpl[] {
  return describeTools().map((spec) => ({
    descriptor: {
      id: `clinical.${spec.name}`,
      group: "clinical",
      description: spec.description || spec.name,
      parameters: {
        type: "object",
        properties: (spec.parameters?.properties ?? {}) as never,
        required: spec.parameters?.required ?? [],
      },
      returns: "dict",
      sideEffects: false,
      timeoutSec: 60,
    },
    async run(args, ctx) {
      return runDispatch(spec.name, args, scopes.get(ctx.chatId));
    },
  }));
}
