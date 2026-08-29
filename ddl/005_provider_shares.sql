CREATE TABLE IF NOT EXISTS llm.provider_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider_id UUID NOT NULL,
    group_id UUID NOT NULL REFERENCES auth.groups(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (owner_id, provider_id, group_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_provider_shares_group
    ON llm.provider_shares (group_id);
