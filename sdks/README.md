# MemoryPilot SDKs

REST clients for server mode. Code only, not published to npm, PyPI or pub.dev.

| Language   | Directory        | Auth                          |
|------------|------------------|-------------------------------|
| TypeScript | `sdks/typescript` | `Authorization: Bearer`       |
| Python     | `sdks/python`     | plus `X-MemoryPilot-Subject`  |
| Dart       | `sdks/dart`       | same                          |

```ts
import { MemoryPilot } from "@memorypilot/sdk";
const client = new MemoryPilot({ baseUrl, apiKey }).forSubject("alice");
await client.addMemory({ content: "Alice prefers green tea" });
```

`scripts/test-sdks.sh` runs the three test suites (no live server).
