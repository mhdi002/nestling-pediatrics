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

export class MemoryConversationRepo implements ConversationRepo {
  private readonly chats = new Map<string, ChatSummary>();
  private readonly turns = new Map<string, StoredTurn[]>();
  private readonly events = new Map<string, unknown[]>();
  private readonly aliases = new Map<string, string>();

  bindAlias(externalId: string, chatId: string): void {
    this.aliases.set(externalId, chatId);
  }

  resolve(externalId: string): string | undefined {
    return this.aliases.get(externalId);
  }

  async createChat(title: string, workspaceId: string | null): Promise<ChatSummary> {
    const now = Date.now();
    const chat: ChatSummary = { id: newId("chat"), title, createdAt: now, updatedAt: now, workspaceId };
    this.chats.set(chat.id, chat);
    this.turns.set(chat.id, []);
    return chat;
  }

  async listChats(workspaceId?: string | null): Promise<ChatSummary[]> {
    return [...this.chats.values()].filter((c) => workspaceId === undefined || c.workspaceId === workspaceId);
  }

  async getChat(chatId: string): Promise<ChatSummary | null> {
    return this.chats.get(chatId) ?? null;
  }

  async setChatWorkspace(chatId: string, workspaceId: string): Promise<void> {
    const c = this.chats.get(chatId);
    if (c) this.chats.set(chatId, { ...c, workspaceId, updatedAt: Date.now() });
  }

  async renameChat(chatId: string, title: string): Promise<void> {
    const c = this.chats.get(chatId);
    if (c) this.chats.set(chatId, { ...c, title, updatedAt: Date.now() });
  }

  async deleteChat(chatId: string): Promise<void> {
    this.chats.delete(chatId);
    this.turns.delete(chatId);
  }

  async getTurns(chatId: string): Promise<StoredTurn[]> {
    return [...(this.turns.get(chatId) ?? [])];
  }

  async appendTurn(turn: Omit<StoredTurn, "idx">): Promise<StoredTurn> {
    const list = this.turns.get(turn.chatId) ?? [];
    const stored: StoredTurn = { ...turn, idx: list.length, meta: turn.meta ?? {} };
    list.push(stored);
    this.turns.set(turn.chatId, list);
    const chat = this.chats.get(turn.chatId);
    if (chat) this.chats.set(turn.chatId, { ...chat, updatedAt: Date.now() });
    return stored;
  }

  async updateTurn(
    turnId: string,
    patch: Partial<Pick<StoredTurn, "assistantText" | "status">> & { meta?: StoredTurn["meta"]; events?: unknown[] },
  ): Promise<void> {
    for (const list of this.turns.values()) {
      const t = list.find((x) => x.id === turnId);
      if (!t) continue;
      if (patch.assistantText !== undefined) t.assistantText = patch.assistantText;
      if (patch.status !== undefined) t.status = patch.status;
      if (patch.meta) t.meta = { ...t.meta, ...patch.meta };
      if (patch.events) this.events.set(turnId, patch.events);
      return;
    }
  }

  async getTurnEvents(turnId: string): Promise<unknown[]> {
    return this.events.get(turnId) ?? [];
  }

  async setFeedback(turnId: string, value: "up" | "down" | null): Promise<void> {
    for (const list of this.turns.values()) {
      const t = list.find((x) => x.id === turnId);
      if (t) t.meta = { ...t.meta, feedback: value };
    }
  }
}

export class MemoryPinned implements PinnedFactsRepo {
  private readonly rows = new Map<string, PinnedFactView[]>();
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
    for (const [k, list] of this.rows) this.rows.set(k, list.filter((x) => x.id !== id));
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
  private readonly rows = new Map<string, TaskBoard>();
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
    return [...this.rows.values()];
  }
}

export class MemorySnapshots implements TranscriptSnapshotStore {
  private readonly rows = new Map<string, TranscriptSnapshot>();
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
