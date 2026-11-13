import type { ChatSummary, PinnedFactView } from "@rlm/contracts";
import type {
  ConversationRepo,
  EditJournal,
  MemoryFilter,
  MemoryPoint,
  MemoryVectorStore,
  PendingEditRecord,
  PinnedFactsRepo,
  TaskBoardStore,
  TranscriptSnapshot,
  TranscriptSnapshotStore,
} from "@rlm/core";
import type { TaskBoard } from "@rlm/core";
import type { StoredTurn } from "@rlm/core";
import { newId } from "@rlm/core";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

/** A keyed set of records that survives a restart of this process.
 *
 * Given a directory, every record is one JSON file in it, written whole and renamed into place, so
 * a crash leaves the old file or the new one and never half of either. A record is read from disk
 * the first time it is asked for. Without a directory the shelf is memory only, which is what a
 * test wants. A conversation, its compacted transcript and its task board all sit on shelves: the
 * parent's chat must still be there after the orchestrator is redeployed or crashes. */
export class Shelf<V> {
  private readonly cache = new Map<string, V>();
  private listed = false;

  constructor(private readonly dir?: string) {
    if (dir) mkdirSync(dir, { recursive: true });
  }

  private file(key: string): string {
    // Keys come from outside (a session id); the file name must not.
    return path.join(this.dir!, `${createHash("sha256").update(key).digest("hex").slice(0, 40)}.json`);
  }

  get(key: string): V | undefined {
    if (this.cache.has(key)) return this.cache.get(key);
    if (!this.dir) return undefined;
    try {
      const row = JSON.parse(readFileSync(this.file(key), "utf8")) as { key: string; value: V };
      this.cache.set(key, row.value);
      return row.value;
    } catch {
      return undefined;
    }
  }

  set(key: string, value: V): void {
    this.cache.set(key, value);
    if (!this.dir) return;
    const file = this.file(key);
    const tmp = `${file}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify({ key, value }));
    renameSync(tmp, file);
  }

  delete(key: string): void {
    this.cache.delete(key);
    if (this.dir) rmSync(this.file(key), { force: true });
  }

  /** Every record, reading the ones not yet in memory. */
  values(): V[] {
    if (this.dir && !this.listed) {
      for (const name of readdirSync(this.dir)) {
        if (!name.endsWith(".json")) continue;
        try {
          const row = JSON.parse(readFileSync(path.join(this.dir, name), "utf8")) as { key: string; value: V };
          if (!this.cache.has(row.key)) this.cache.set(row.key, row.value);
        } catch {
          /* a file that does not parse is not a record */
        }
      }
      this.listed = true;
    }
    return [...this.cache.values()];
  }
}

const sub = (dir: string | undefined, name: string) => (dir ? path.join(dir, name) : undefined);

interface ChatRecord {
  chat: ChatSummary;
  turns: StoredTurn[];
}

export class MemoryConversationRepo implements ConversationRepo {
  private readonly chats: Shelf<ChatRecord>;
  private readonly aliases: Shelf<string>;
  /** Events are the live trace of a turn, read while it runs; they are not kept across restarts. */
  private readonly events = new Map<string, unknown[]>();
  private readonly turnChat = new Map<string, string>();

  constructor(dir?: string) {
    this.chats = new Shelf<ChatRecord>(sub(dir, "chats"));
    this.aliases = new Shelf<string>(sub(dir, "aliases"));
  }

  bindAlias(externalId: string, chatId: string): void {
    this.aliases.set(externalId, chatId);
  }

  resolve(externalId: string): string | undefined {
    const id = this.aliases.get(externalId);
    return id && this.chats.get(id) ? id : undefined;
  }

  private record(chatId: string): ChatRecord | undefined {
    const r = this.chats.get(chatId);
    if (r) for (const t of r.turns) this.turnChat.set(t.id, chatId);
    return r;
  }

  private recordOfTurn(turnId: string): ChatRecord | undefined {
    const known = this.turnChat.get(turnId);
    if (known) return this.record(known);
    return this.chats.values().find((r) => r.turns.some((t) => t.id === turnId));
  }

  async createChat(title: string, workspaceId: string | null): Promise<ChatSummary> {
    const now = Date.now();
    const chat: ChatSummary = { id: newId("chat"), title, createdAt: now, updatedAt: now, workspaceId };
    this.chats.set(chat.id, { chat, turns: [] });
    return chat;
  }

  async listChats(workspaceId?: string | null): Promise<ChatSummary[]> {
    return this.chats
      .values()
      .map((r) => r.chat)
      .filter((c) => workspaceId === undefined || c.workspaceId === workspaceId);
  }

  async getChat(chatId: string): Promise<ChatSummary | null> {
    return this.record(chatId)?.chat ?? null;
  }

  private touch(chatId: string, patch: Partial<ChatSummary>): void {
    const r = this.record(chatId);
    if (r) this.chats.set(chatId, { ...r, chat: { ...r.chat, ...patch, updatedAt: Date.now() } });
  }

  async setChatWorkspace(chatId: string, workspaceId: string): Promise<void> {
    this.touch(chatId, { workspaceId });
  }

  async renameChat(chatId: string, title: string): Promise<void> {
    this.touch(chatId, { title });
  }

  async deleteChat(chatId: string): Promise<void> {
    this.chats.delete(chatId);
  }

  async getTurns(chatId: string): Promise<StoredTurn[]> {
    return [...(this.record(chatId)?.turns ?? [])];
  }

  async appendTurn(turn: Omit<StoredTurn, "idx">): Promise<StoredTurn> {
    const r = this.record(turn.chatId);
    const turns = r?.turns ?? [];
    const stored: StoredTurn = { ...turn, idx: turns.length, meta: turn.meta ?? {} };
    turns.push(stored);
    this.turnChat.set(stored.id, turn.chatId);
    const now = Date.now();
    const chat = r?.chat ?? { id: turn.chatId, title: "", createdAt: now, updatedAt: now, workspaceId: null };
    this.chats.set(turn.chatId, { chat: { ...chat, updatedAt: now }, turns });
    return stored;
  }

  async updateTurn(
    turnId: string,
    patch: Partial<Pick<StoredTurn, "assistantText" | "status">> & { meta?: StoredTurn["meta"]; events?: unknown[] },
  ): Promise<void> {
    if (patch.events) this.events.set(turnId, patch.events);
    const r = this.recordOfTurn(turnId);
    const t = r?.turns.find((x) => x.id === turnId);
    if (!r || !t) return;
    if (patch.assistantText !== undefined) t.assistantText = patch.assistantText;
    if (patch.status !== undefined) t.status = patch.status;
    if (patch.meta) t.meta = { ...t.meta, ...patch.meta };
    this.chats.set(r.chat.id, r);
  }

  async getTurnEvents(turnId: string): Promise<unknown[]> {
    return this.events.get(turnId) ?? [];
  }

  async setFeedback(turnId: string, value: "up" | "down" | null): Promise<void> {
    const r = this.recordOfTurn(turnId);
    const t = r?.turns.find((x) => x.id === turnId);
    if (!r || !t) return;
    t.meta = { ...t.meta, feedback: value };
    this.chats.set(r.chat.id, r);
  }
}

export class MemoryPinned implements PinnedFactsRepo {
  private readonly rows: Shelf<PinnedFactView[]>;
  constructor(dir?: string) {
    this.rows = new Shelf<PinnedFactView[]>(sub(dir, "pinned"));
  }
  async list(chatId: string): Promise<PinnedFactView[]> {
    return [...(this.rows.get(chatId) ?? [])];
  }
  async upsert(f: Omit<PinnedFactView, "id"> & { id?: string }): Promise<PinnedFactView> {
    const row: PinnedFactView = { id: f.id ?? newId("pin"), chatId: f.chatId, kind: f.kind, text: f.text };
    const list = this.rows.get(f.chatId) ?? [];
    const i = list.findIndex((x) => x.id === row.id || x.kind === row.kind);
    if (i >= 0) list[i] = row;
    else list.push(row);
    this.rows.set(f.chatId, list);
    return row;
  }
  async delete(id: string): Promise<void> {
    for (const list of this.rows.values()) {
      const chatId = list.find((x) => x.id === id)?.chatId;
      if (chatId) this.rows.set(chatId, list.filter((x) => x.id !== id));
    }
  }
}

export class MemoryEdits implements EditJournal {
  private readonly rows: PendingEditRecord[] = [];
  async record(e: Omit<PendingEditRecord, "state">): Promise<void> {
    this.rows.push({ ...e, state: "applied" });
  }
  async list(): Promise<PendingEditRecord[]> {
    return [...this.rows];
  }
  async setState(editId: string, state: PendingEditRecord["state"]): Promise<void> {
    const r = this.rows.find((x) => x.editId === editId);
    if (r) r.state = state;
  }
  async get(editId: string): Promise<PendingEditRecord | null> {
    return this.rows.find((x) => x.editId === editId) ?? null;
  }
}

export class MemoryTaskBoards implements TaskBoardStore {
  private readonly rows: Shelf<TaskBoard>;
  constructor(dir?: string) {
    this.rows = new Shelf<TaskBoard>(sub(dir, "boards"));
  }
  async load(chatId: string): Promise<TaskBoard | null> {
    return this.rows.get(chatId) ?? null;
  }
  async save(b: TaskBoard): Promise<void> {
    this.rows.set(b.chatId, b);
  }
  async delete(chatId: string): Promise<void> {
    this.rows.delete(chatId);
  }
  async list(): Promise<TaskBoard[]> {
    return this.rows.values();
  }
}

export class MemorySnapshots implements TranscriptSnapshotStore {
  private readonly rows: Shelf<TranscriptSnapshot>;
  constructor(dir?: string) {
    this.rows = new Shelf<TranscriptSnapshot>(sub(dir, "snapshots"));
  }
  async load(chatId: string): Promise<TranscriptSnapshot | null> {
    return this.rows.get(chatId) ?? null;
  }
  async save(s: TranscriptSnapshot): Promise<void> {
    this.rows.set(s.chatId, s);
  }
  async delete(chatId: string): Promise<void> {
    this.rows.delete(chatId);
  }
}

function cosine(a: number[], b: number[]): number {
  if (!a.length || a.length !== b.length) return 0;
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na <= 0 || nb <= 0) return 0;
  return d / Math.sqrt(na * nb);
}

function matchFilter(p: MemoryPoint["payload"], f: MemoryFilter): boolean {
  if (p.user_id !== f.user_id) return false;
  if (f.memory_type && p.memory_type !== f.memory_type) return false;
  if (f.kind && p.kind !== f.kind) return false;
  if (f.session_id !== undefined && p.session_id !== f.session_id) return false;
  if (f.workspace_id) {
    if (p.workspace_id === f.workspace_id) return true;
    if (f.include_global && !p.workspace_id) return true;
    return false;
  }
  return true;
}

export class MemoryVectors implements MemoryVectorStore {
  private readonly points: MemoryPoint[] = [];

  async upsert(_modelKey: string, points: MemoryPoint[]): Promise<void> {
    for (const p of points) {
      if (p.payload.memory_type === "short_term") continue;
      const i = this.points.findIndex((x) => x.id === p.id);
      if (i >= 0) this.points[i] = p;
      else this.points.push(p);
    }
  }

  async search(_modelKey: string, vector: number[], filter: MemoryFilter, limit: number): Promise<Array<MemoryPoint & { score: number }>> {
    const scored = this.points
      .filter((p) => matchFilter(p.payload, filter))
      .map((p) => ({ ...p, score: p.vector ? cosine(vector, p.vector) : 0.4 }))
      .sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  async list(filter: MemoryFilter, textQuery?: string): Promise<MemoryPoint[]> {
    const q = (textQuery ?? "").toLowerCase();
    return this.points.filter((p) => matchFilter(p.payload, filter) && (!q || p.payload.text.toLowerCase().includes(q)));
  }

  async get(id: string, filter: MemoryFilter): Promise<MemoryPoint | null> {
    return this.points.find((p) => p.id === id && matchFilter(p.payload, filter)) ?? null;
  }

  async delete(ids: string[], filter: MemoryFilter): Promise<void> {
    const drop = new Set(ids);
    for (let i = this.points.length - 1; i >= 0; i--) {
      if (drop.has(this.points[i]!.id) && matchFilter(this.points[i]!.payload, filter)) this.points.splice(i, 1);
    }
  }

  async deleteAll(filter: MemoryFilter): Promise<void> {
    for (let i = this.points.length - 1; i >= 0; i--) {
      if (matchFilter(this.points[i]!.payload, filter)) this.points.splice(i, 1);
    }
  }
}
