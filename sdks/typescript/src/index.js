/** MemoryPilot server REST client. */

export class MemoryPilotError extends Error {
  constructor(status, body) {
    const message =
      (body && typeof body === "object" && body.error) ||
      `MemoryPilot HTTP ${status}`;
    super(message);
    this.name = "MemoryPilotError";
    this.status = status;
    this.body = body;
  }
}

export class MemoryPilot {
  /**
   * @param {{ baseUrl: string, apiKey: string, subject?: string }} options
   */
  constructor({ baseUrl, apiKey, subject }) {
    if (!baseUrl || !apiKey) {
      throw new Error("baseUrl and apiKey are required");
    }
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.subject = subject;
  }

  /** A client that acts for one end user of the app. */
  forSubject(subject) {
    return new MemoryPilot({
      baseUrl: this.baseUrl,
      apiKey: this.apiKey,
      subject,
    });
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return json(response);
  }

  async addMemory(memory) {
    return this.request("POST", "/v1/memories", memory);
  }

  async listMemories({ project, kind, limit, offset } = {}) {
    const query = new URLSearchParams();
    if (project) query.set("project", project);
    if (kind) query.set("kind", kind);
    if (limit != null) query.set("limit", String(limit));
    if (offset != null) query.set("offset", String(offset));
    const suffix = query.size ? `?${query}` : "";
    return this.request("GET", `/v1/memories${suffix}`);
  }

  async getMemory(id) {
    return this.request("GET", `/v1/memories/${encodeURIComponent(id)}`);
  }

  async updateMemory(id, changes) {
    return this.request("PATCH", `/v1/memories/${encodeURIComponent(id)}`, changes);
  }

  async deleteMemory(id) {
    return this.request("DELETE", `/v1/memories/${encodeURIComponent(id)}`);
  }

  async search(query) {
    return this.request("POST", "/v1/search", query);
  }

  async exportSubject() {
    return this.request("GET", "/v1/subject/export");
  }

  async deleteSubject() {
    return this.request("DELETE", "/v1/subject");
  }

  async request(method, path, body) {
    const headers = {
      authorization: `Bearer ${this.apiKey}`,
      accept: "application/json",
    };
    if (this.subject) {
      headers["x-memorypilot-subject"] = this.subject;
    }
    const init = { method, headers };
    if (body !== undefined) {
      headers["content-type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const response = await fetch(`${this.baseUrl}${path}`, init);
    return json(response);
  }
}

async function json(response) {
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new MemoryPilotError(response.status, body);
  }
  return body;
}
