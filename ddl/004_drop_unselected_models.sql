-- Каталог с refresh больше не пишется в llm_models.
-- Строки enabled=false — мусор от старого INSERT всего списка.
DELETE FROM llm.llm_models WHERE enabled = FALSE;
