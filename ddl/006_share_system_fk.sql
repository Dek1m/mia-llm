ALTER TABLE llm.provider_shares
    DROP CONSTRAINT IF EXISTS provider_shares_provider_fk;

ALTER TABLE llm.provider_shares
    ADD CONSTRAINT provider_shares_provider_fk
    FOREIGN KEY (provider_id) REFERENCES llm.llm_providers(id) ON DELETE CASCADE;
