export class MemoryPilotError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown);
}

export type MemoryInput = {
  content: string;
  kind?: string;
  project?: string;
  tags?: string[];
  importance?: number;
  expires_at?: string;
  metadata?: Record<string, unknown>;
};

export type MemoryChanges = Partial<MemoryInput>;

export type SearchInput = {
  query: string;
  limit?: number;
  project?: string;
  kind?: string;
  tags?: string[];
};

export type ClientOptions = {
  baseUrl: string;
  apiKey: string;
  subject?: string;
};

export class MemoryPilot {
  constructor(options: ClientOptions);
  forSubject(subject: string): MemoryPilot;
  health(): Promise<{ ok: boolean }>;
  addMemory(memory: MemoryInput): Promise<{ memory: unknown; merged: boolean }>;
  listMemories(query?: {
    project?: string;
    kind?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ memories: unknown[]; total: number; count: number; offset: number }>;
  getMemory(id: string): Promise<{ memory: unknown }>;
  updateMemory(id: string, changes: MemoryChanges): Promise<{ memory: unknown }>;
  deleteMemory(id: string): Promise<{ deleted: string }>;
  search(query: SearchInput): Promise<{ results: unknown[]; confidence: unknown }>;
  exportSubject(): Promise<{ memories: unknown[]; facts: unknown[]; end_user_id: string }>;
  deleteSubject(): Promise<{ end_user_id: string; deleted_memories: number }>;
}
