export type MemoryRow = {
  id: string;
  content: string;
  kind: string;
  project?: string | null;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
  }
}

export class Api {
  constructor(
    readonly baseUrl: string,
    readonly apiKey: string,
    readonly subject: string
  ) {}

  async health(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new ApiError(response.status, "Le serveur ne répond pas");
    }
  }

  async list(): Promise<MemoryRow[]> {
    const body = await this.request("GET", "/v1/memories");
    return (body.memories as MemoryRow[]) ?? [];
  }

  async add(content: string): Promise<void> {
    await this.request("POST", "/v1/memories", { content });
  }

  async search(query: string): Promise<MemoryRow[]> {
    const body = await this.request("POST", "/v1/search", { query, limit: 10 });
    const results = (body.results as { memory: MemoryRow }[]) ?? [];
    return results.map((result) => result.memory);
  }

  async exportSubject(): Promise<unknown> {
    return this.request("GET", "/v1/subject/export");
  }

  async deleteSubject(): Promise<unknown> {
    return this.request("DELETE", "/v1/subject");
  }

  private async request(method: string, path: string, body?: unknown): Promise<Record<string, unknown>> {
    const headers: Record<string, string> = {
      authorization: `Bearer ${this.apiKey}`,
      accept: "application/json"
    };
    if (this.subject) {
      headers["x-memorypilot-subject"] = this.subject;
    }
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: body ? { ...headers, "content-type": "application/json" } : headers,
      body: body ? JSON.stringify(body) : undefined
    });
    const text = await response.text();
    const parsed = text ? (JSON.parse(text) as Record<string, unknown>) : {};
    if (!response.ok) {
      throw new ApiError(response.status, String(parsed.error ?? `Erreur ${response.status}`));
    }
    return parsed;
  }
}
