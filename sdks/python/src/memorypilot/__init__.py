"""MemoryPilot server REST client."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class MemoryPilotError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        message = body.get("error") if isinstance(body, dict) else None
        super().__init__(message or f"MemoryPilot HTTP {status}")
        self.status = status
        self.body = body


class MemoryPilot:
    def __init__(self, base_url: str, api_key: str, subject: str | None = None) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.subject = subject

    def for_subject(self, subject: str) -> "MemoryPilot":
        return MemoryPilot(self.base_url, self.api_key, subject)

    def health(self) -> dict[str, Any]:
        return self._send("GET", "/health", auth=False)

    def add_memory(self, **memory: Any) -> dict[str, Any]:
        return self._send("POST", "/v1/memories", memory)

    def list_memories(
        self,
        project: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value in {
                    "project": project,
                    "kind": kind,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if value is not None
            }
        )
        path = "/v1/memories" + (f"?{query}" if query else "")
        return self._send("GET", path)

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        return self._send("GET", f"/v1/memories/{urllib.parse.quote(memory_id)}")

    def update_memory(self, memory_id: str, **changes: Any) -> dict[str, Any]:
        return self._send("PATCH", f"/v1/memories/{urllib.parse.quote(memory_id)}", changes)

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        return self._send("DELETE", f"/v1/memories/{urllib.parse.quote(memory_id)}")

    def search(self, query: str, **filters: Any) -> dict[str, Any]:
        body = {"query": query, **filters}
        return self._send("POST", "/v1/search", body)

    def export_subject(self) -> dict[str, Any]:
        return self._send("GET", "/v1/subject/export")

    def delete_subject(self) -> dict[str, Any]:
        return self._send("DELETE", "/v1/subject")

    def _send(self, method: str, path: str, body: dict[str, Any] | None = None, auth: bool = True) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.subject:
                headers["X-MemoryPilot-Subject"] = self.subject
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read().decode()
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as error:
            raw = error.read().decode()
            parsed: Any
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            raise MemoryPilotError(error.code, parsed) from error
