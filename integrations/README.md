# Phase 7 — brancher MemoryPilot

## claude.ai et ChatGPT (serveur)

1. Déployer `memorypilot-server` avec `MEMORYPILOT_PUBLIC_URL=https://…` et `MEMORYPILOT_ALLOWED_ORIGINS=https://claude.ai,https://chatgpt.com`.
2. Créer un tenant, un space, une clé `read,write,subjects`.
3. Dans Claude / ChatGPT : connecteur MCP personnalisé, URL `https://…/mcp`.
4. Le client découvre OAuth (`/.well-known/oauth-authorization-server`), s'enregistre, ouvre `/oauth/authorize`.
5. Coller la clé. Optionnel : un id d'utilisateur final. Chaque personne obtient son jeton `mpt_`.

## iPhone

L'app parle HTTP comme les SDK. En Dart :

```dart
final client = MemoryPilot(baseUrl: 'https://…', apiKey: key).forSubject(userId);
await client.addMemory({'content': note});
```

Raccourci iOS : `POST https://…/v1/memories` avec `Authorization: Bearer` et `X-MemoryPilot-Subject`.

## Slate (local)

Slate reste sur le MemoryPilot de bureau (stdio MCP, base SQLite). Pas le SaaS. Dans la config MCP de Claude Desktop :

```json
{ "mcpServers": { "memorypilot": { "command": "MemoryPilot" } } }
```

Le binaire local est déjà le serveur MCP. Slate n'a pas à embarquer le SaaS.

## LightFrame (SDK TypeScript)

```ts
import { MemoryPilot } from "../../MemoryPilot/sdks/typescript/src/index.js";

const memory = new MemoryPilot({
  baseUrl: process.env.MEMORYPILOT_URL!,
  apiKey: process.env.MEMORYPILOT_KEY!,
}).forSubject(siteId);

await memory.addMemory({
  content: `The site ${siteId} publishes in ${locale}`,
  project: "lightframe",
});
```

## Docix / Konta (SDK Dart)

Dans l'app mobile, dépendance path :

```yaml
memorypilot:
  path: ../../MemoryPilot/sdks/dart
```

Puis `MemoryPilot(baseUrl: …, apiKey: …).forSubject(userId)` pour les notes de collecte.
