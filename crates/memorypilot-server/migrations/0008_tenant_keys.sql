-- Envelope encryption: each tenant has a 32-byte data key, stored only
-- wrapped with the server master key. NULL until a master key is configured.
ALTER TABLE tenants ADD COLUMN wrapped_dek bytea;
