import { spawn } from "node:child_process";
import path from "node:path";
import type { ToolImpl } from "@rlm/core";

const NAMES = [
  "growth_percentile",
  "overlay_growth_on_chart",
  "score_asq_questionnaire",
  "score_mchat",
  "list_asq_questions",
  "list_mchat_questions",
  "get_child_summary",
] as const;

export function runClinical(name: string, args: Record<string, unknown>): Promise<unknown> {
  return runDispatch(name, args, { NESTLING_CHILD_DB: process.env.NESTLING_CHILD_DB ?? "" });
}

function runDispatch(name: string, args: Record<string, unknown>, env: NodeJS.ProcessEnv): Promise<unknown> {
  const root = process.env.NESTLING_ROOT || path.resolve(import.meta.dirname, "../../../../..");
  const script = path.join(root, "assistant/rlm/clinical_dispatch.py");
  const py = process.env.RLM_PYTHON || process.env.NESTLING_PYTHON || "python3";
  return new Promise((resolve, reject) => {
    const child = spawn(py, [script], { env: { ...process.env, ...env }, stdio: ["pipe", "pipe", "pipe"] });
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
    child.stdin.end(JSON.stringify({ name, args }));
  });
}

export function createClinicalTools(): ToolImpl[] {
  return NAMES.map((name) => ({
    descriptor: {
      id: `clinical.${name}`,
      group: "clinical",
      description: `Nestling clinical tool ${name}. Returns JSON. Never invent percentiles.`,
      parameters: {
        type: "object",
        properties: {
          sex: { type: "string" },
          measure: { type: "string" },
          weeks: { type: "number" },
          value: { type: "number" },
          gestational_age_weeks: { type: "number" },
          chart_standard: { type: "string" },
          child_id: { type: "string" },
          percentile: { type: "number" },
          answers: { type: "object" },
          questionnaire: { type: "string" },
        },
        required: [],
      },
      returns: "dict",
      sideEffects: false,
      timeoutSec: 60,
    },
    async run(args) {
      return runDispatch(name, args, {
        NESTLING_CHILD_DB: process.env.NESTLING_CHILD_DB ?? "",
      });
    },
  }));
}
