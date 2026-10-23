import { createServer } from "node:http";
import { test } from "node:test";
import assert from "node:assert/strict";
import { MemoryPilot, MemoryPilotError } from "../src/index.js";

function listen(handler) {
  return new Promise((resolve) => {
    const server = createServer(handler);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () =>
          new Promise((done) => {
            server.close(done);
          }),
      });
    });
  });
}

test("adds a memory for an end user and surfaces HTTP errors", async () => {
  const seen = [];
  const server = await listen((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      seen.push({
        method: request.method,
        url: request.url,
        authorization: request.headers.authorization,
        subject: request.headers["x-memorypilot-subject"],
        body: Buffer.concat(chunks).toString(),
      });
      if (request.url === "/v1/memories" && request.method === "POST") {
        response.writeHead(201, { "content-type": "application/json" });
        response.end(JSON.stringify({ memory: { id: "m1", content: "hello" }, merged: false }));
        return;
      }
      if (request.url === "/v1/search") {
        response.writeHead(403, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: "This key lacks the 'read' permission" }));
        return;
      }
      response.writeHead(404);
      response.end();
    });
  });
  const client = new MemoryPilot({
    baseUrl: server.url,
    apiKey: "mpk_test",
  }).forSubject("alice");
  const created = await client.addMemory({ content: "Alice prefers green tea" });
  assert.equal(created.memory.id, "m1");
  assert.equal(seen[0].authorization, "Bearer mpk_test");
  assert.equal(seen[0].subject, "alice");
  await assert.rejects(() => client.search({ query: "tea" }), (error) => {
    assert.ok(error instanceof MemoryPilotError);
    assert.equal(error.status, 403);
    return true;
  });
  await server.close();
});
