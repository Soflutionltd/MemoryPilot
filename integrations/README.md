# Phase 7 : brancher MemoryPilot

## Règle d'isolation

Un produit ou un client = un tenant vierge, ou sa propre instance (image Docker chez lui, sa base Postgres). Jamais la base locale d'Antoine, jamais deux clients sur le même tenant : aucune donnée ne passe d'un client à l'autre. Seuls les credentials sont chiffrés par tenant ; un client qui ne veut pas que l'hébergeur puisse lire ses mémoires héberge l'image lui-même.

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

## Usage d'Antoine (local)

Le MemoryPilot de bureau (binaire, base SQLite, servi par McpHub) est la mémoire de l'agent dans Cursor. Il n'est branché sur aucun produit et ne parle jamais au serveur SaaS. Un client MCP de bureau qui veut la même mémoire locale le lance en stdio :

```json
{ "mcpServers": { "memorypilot": { "command": "MemoryPilot" } } }
```

## LightFrame (SDK TypeScript)

```ts
import { MemoryPilot } from "../../MemoryPilot/sdks/typescript/src/index.js";

// One tenant (and API key) per LightFrame customer; end users inside it.
const memory = new MemoryPilot({
  baseUrl: process.env.MEMORYPILOT_URL!,
  apiKey: customerApiKey,
}).forSubject(userId);

await memory.addMemory({
  content: `${userId} publishes in ${locale}`,
  project: "lightframe",
});
```

Deux sites de deux clients différents ne partagent jamais un tenant : `forSubject` sépare les utilisateurs d'un même client, pas des entreprises.

## Docix / Konta (SDK Dart)

Dans l'app mobile, dépendance path :

```yaml
memorypilot:
  path: ../../MemoryPilot/sdks/dart
```

Puis `MemoryPilot(baseUrl: …, apiKey: …).forSubject(userId)` pour les notes de collecte.
