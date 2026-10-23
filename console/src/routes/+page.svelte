<script lang="ts">
  import { Api, ApiError, type MemoryRow } from "$lib/api";
  import { clearSession, loadSession, saveSession, type Session } from "$lib/session.svelte";

  let session = $state<Session>(loadSession());
  let signedIn = $state(false);
  let error = $state("");
  let memories = $state<MemoryRow[]>([]);
  let draft = $state("");
  let query = $state("");
  let busy = $state(false);

  const client = $derived(
    signedIn ? new Api(session.baseUrl.replace(/\/$/, ""), session.apiKey, session.subject) : null
  );
  const mcpUrl = $derived(`${session.baseUrl.replace(/\/$/, "")}/mcp`);
  const oauthUrl = $derived(`${session.baseUrl.replace(/\/$/, "")}/oauth/authorize`);

  async function connect(event: SubmitEvent) {
    event.preventDefault();
    error = "";
    busy = true;
    try {
      const api = new Api(session.baseUrl.replace(/\/$/, ""), session.apiKey, session.subject);
      await api.health();
      memories = await api.list();
      saveSession(session);
      signedIn = true;
    } catch (caught) {
      error = caught instanceof ApiError ? caught.message : "Connexion impossible";
    } finally {
      busy = false;
    }
  }

  function disconnect() {
    clearSession();
    signedIn = false;
    memories = [];
  }

  async function addMemory(event: SubmitEvent) {
    event.preventDefault();
    if (!client || !draft.trim()) return;
    busy = true;
    error = "";
    try {
      await client.add(draft.trim());
      draft = "";
      memories = await client.list();
    } catch (caught) {
      error = caught instanceof ApiError ? caught.message : "Écriture refusée";
    } finally {
      busy = false;
    }
  }

  async function runSearch(event: SubmitEvent) {
    event.preventDefault();
    if (!client || !query.trim()) return;
    busy = true;
    error = "";
    try {
      memories = await client.search(query.trim());
    } catch (caught) {
      error = caught instanceof ApiError ? caught.message : "Recherche impossible";
    } finally {
      busy = false;
    }
  }

  async function exportUser() {
    if (!client) return;
    busy = true;
    error = "";
    try {
      const data = await client.exportSubject();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "memorypilot-export.json";
      link.click();
      URL.revokeObjectURL(url);
    } catch (caught) {
      error = caught instanceof ApiError ? caught.message : "Export impossible";
    } finally {
      busy = false;
    }
  }

  async function eraseUser() {
    if (!client || !session.subject) return;
    if (!confirm(`Effacer toutes les mémoires de ${session.subject} ? Irréversible.`)) return;
    busy = true;
    error = "";
    try {
      await client.deleteSubject();
      memories = [];
    } catch (caught) {
      error = caught instanceof ApiError ? caught.message : "Effacement impossible";
    } finally {
      busy = false;
    }
  }
</script>

{#if !signedIn}
  <form class="card" onsubmit={connect}>
    <h1>Ouvrir un espace</h1>
    <label>
      URL du serveur
      <input bind:value={session.baseUrl} placeholder="http://127.0.0.1:8080" required />
    </label>
    <label>
      Clé d'API
      <input bind:value={session.apiKey} type="password" autocomplete="off" required />
    </label>
    <label>
      Utilisateur final (optionnel)
      <input bind:value={session.subject} placeholder="alice" />
    </label>
    {#if error}<p class="alert" role="alert">{error}</p>{/if}
    <button type="submit" disabled={busy}>Entrer</button>
  </form>
{:else}
  <section class="card">
    <div class="row">
      <h1>Mémoires</h1>
      <button class="ghost" type="button" onclick={disconnect}>Quitter</button>
    </div>
    <p class="hint">
      MCP : <code>{mcpUrl}</code><br />
      OAuth : <code>{oauthUrl}</code>
    </p>
    <form class="stack" onsubmit={addMemory}>
      <textarea bind:value={draft} rows="3" placeholder="Une mémoire, autonome."></textarea>
      <button type="submit" disabled={busy || !draft.trim()}>Enregistrer</button>
    </form>
    <form class="stack" onsubmit={runSearch}>
      <input bind:value={query} placeholder="Rechercher" />
      <button class="ghost" type="submit" disabled={busy || !query.trim()}>Chercher</button>
    </form>
    {#if session.subject}
      <div class="row">
        <button class="ghost" type="button" onclick={exportUser} disabled={busy}>Exporter (RGPD)</button>
        <button class="danger" type="button" onclick={eraseUser} disabled={busy}>Effacer cet utilisateur</button>
      </div>
    {/if}
    {#if error}<p class="alert" role="alert">{error}</p>{/if}
    <ul>
      {#each memories as memory (memory.id)}
        <li>
          <strong>{memory.kind}</strong>
          <p>{memory.content}</p>
        </li>
      {:else}
        <li class="empty">Aucune mémoire pour l'instant.</li>
      {/each}
    </ul>
  </section>
{/if}

<style>
  h1 {
    margin: 0 0 1.5rem;
    font-size: 2rem;
    letter-spacing: -0.03em;
  }

  .card {
    padding: 2rem;
    border: 1px solid var(--line);
    border-radius: 1.5rem;
  }

  label,
  .stack {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
  }

  input,
  textarea,
  button {
    padding: 0.85rem 1rem;
    border-radius: 0.85rem;
    border: 1px solid var(--line);
    background: #1c1916;
    color: inherit;
  }

  button {
    background: var(--accent);
    color: #14120f;
    font-weight: 700;
    border: 0;
    cursor: pointer;
  }

  button:disabled {
    opacity: 0.5;
  }

  .ghost {
    background: transparent;
    color: var(--ink);
    border: 1px solid var(--line);
  }

  .danger {
    background: transparent;
    color: var(--danger);
    border: 1px solid var(--danger);
  }

  .row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1.25rem;
  }

  .hint,
  .empty,
  .alert {
    color: var(--muted);
  }

  .alert {
    color: var(--danger);
  }

  ul {
    list-style: none;
    padding: 0;
    margin: 2rem 0 0;
  }

  li {
    padding: 1rem 0;
    border-top: 1px solid var(--line);
  }

  li p {
    margin: 0.35rem 0 0;
    max-width: 40rem;
  }

  code {
    font-size: 0.85rem;
  }
</style>
