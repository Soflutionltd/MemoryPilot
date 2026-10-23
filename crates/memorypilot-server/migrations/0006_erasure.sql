-- An app erases one of its end users (GDPR): the subject row goes, and
-- their memories, links, entities and facts with it (ON DELETE CASCADE).
GRANT DELETE ON subjects TO memorypilot_app;
