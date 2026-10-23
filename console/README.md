# Console MemoryPilot

Interface SvelteKit (statique) pour un espace : mémoires, recherche, export et effacement RGPD.

```bash
cd console && npm install && npm run build
MEMORYPILOT_CONSOLE_DIR=console/build memorypilot-server serve
```

Ouvrir `http://127.0.0.1:8080/console`. La clé d'API reste dans le navigateur (localStorage), jamais envoyée ailleurs que vers l'URL du serveur saisie.
