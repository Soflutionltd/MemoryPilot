<script lang="ts">
  import "../app.css";
  let { children } = $props();
</script>

<div class="shell">
  <header>
    <p class="brand">MemoryPilot</p>
    <p class="tag">Console</p>
  </header>
  {@render children()}
</div>

<style>
  .shell {
    max-width: 42rem;
    margin: 0 auto;
    padding: 4rem 1.5rem 6rem;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 3rem;
  }

  .brand {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
  }

  .tag {
    margin: 0;
    color: var(--muted);
  }
</style>
