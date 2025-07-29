-- Gola Database Schema
-- SQLite/PostgreSQL compatible schema for the Gola system

-- Core tables for data tracking
CREATE TABLE files (
    file_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    mtime_utc TEXT NOT NULL,
    mime_type TEXT,
    language TEXT,
    pii_level INTEGER DEFAULT 0,
    added_at_utc TEXT NOT NULL
);

-- Document pages
CREATE TABLE pages (
    page_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES files(file_id),
    page_num INTEGER NOT NULL,
    width INTEGER, 
    height INTEGER,
    ocr_conf_mean REAL,
    text_chars INTEGER,
    block_count INTEGER
);

-- Content blocks
CREATE TABLE blocks (
    block_id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL REFERENCES pages(page_id),
    block_idx INTEGER NOT NULL,
    block_type TEXT CHECK(block_type IN ('title','narrative','list','table','figure','footnote')),
    bbox TEXT,
    text TEXT
);

-- Processed chunks
CREATE TABLE chunks (
    chunk_hash TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES files(file_id),
    char_start INTEGER,
    char_end INTEGER,
    text_norm TEXT NOT NULL,
    simhash64 TEXT,
    minhash_sig BLOB,
    duplicate_of TEXT REFERENCES chunks(chunk_hash)
);

-- Processing runs
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    chunk_hash TEXT NOT NULL REFERENCES chunks(chunk_hash),
    task_type TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider TEXT,
    status TEXT CHECK(status IN ('ok','retry','failed')) DEFAULT 'ok',
    started_at_utc TEXT,
    finished_at_utc TEXT,
    token_in INTEGER,
    token_out INTEGER,
    cost_usd REAL,
    job_key TEXT NOT NULL UNIQUE
);

-- Output tables
CREATE TABLE outputs_summary (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    summary_text TEXT NOT NULL,
    keypoints_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL
);

CREATE TABLE outputs_entities (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL
);

CREATE TABLE outputs_triples (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    subject TEXT,
    predicate TEXT,
    object TEXT,
    spans_json TEXT NOT NULL
);

CREATE TABLE outputs_qa_pairs (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    answer_type TEXT NOT NULL,
    evidence_json TEXT NOT NULL
);

-- Document notes
CREATE TABLE document_notes (
    file_id TEXT PRIMARY KEY REFERENCES files(file_id),
    notes_json TEXT NOT NULL
);

-- GPU telemetry
CREATE TABLE gpu_telemetry (
    ts_utc TEXT NOT NULL,
    gpu_index INTEGER NOT NULL,
    util REAL,
    mem_used_mb INTEGER,
    mem_total_mb INTEGER,
    temp_c REAL,
    power_w REAL
);

-- Hub integration tables
CREATE TABLE hub_sync (
    sync_id TEXT PRIMARY KEY,
    dataset_slug TEXT NOT NULL,
    project_id TEXT,
    content_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    rest_or_mcp TEXT CHECK(rest_or_mcp IN ('rest','mcp')) DEFAULT 'rest',
    status TEXT CHECK(status IN ('queued','pushed','failed')) DEFAULT 'queued',
    retries INTEGER DEFAULT 0,
    pushed_at TEXT,
    hub_content_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE plan_archive (
    plan_id TEXT PRIMARY KEY,
    dataset_slug TEXT NOT NULL,
    mode TEXT NOT NULL,
    plan_yaml TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    result TEXT CHECK(result IN ('ok','revise')) DEFAULT 'ok'
);

-- Web crawling tables
CREATE TABLE sites (
    site_id TEXT PRIMARY KEY,
    root_url TEXT NOT NULL,
    robots_txt TEXT,
    sitemap_urls_json TEXT,
    policy_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE web_pages (
    page_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(site_id),
    url TEXT NOT NULL,
    canonical_url TEXT,
    status TEXT,
    fetched_at TEXT,
    mime TEXT,
    lang TEXT,
    content_sha256 TEXT NOT NULL,
    html_path TEXT,
    md_clean_path TEXT,
    md_fit_path TEXT,
    source_kind TEXT CHECK(source_kind IN ('llms','md','plugin','html'))
);

CREATE TABLE web_notes (
    page_id TEXT PRIMARY KEY REFERENCES web_pages(page_id),
    notes_json TEXT NOT NULL
);

CREATE TABLE web_chunks (
    chunk_hash TEXT PRIMARY KEY,
    page_id TEXT REFERENCES web_pages(page_id),
    char_start INTEGER,
    char_end INTEGER,
    text_norm TEXT NOT NULL,
    simhash64 TEXT,
    minhash_sig BLOB,
    duplicate_of TEXT REFERENCES web_chunks(chunk_hash)
);

-- LLM-ready assets
CREATE TABLE llm_ready_assets (
    asset_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(site_id),
    kind TEXT CHECK(kind IN ('llms.txt','llms-full.txt','md_variant','plugin_manifest','ai.txt')),
    url TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    status TEXT CHECK(status IN ('ok','missing','error')) DEFAULT 'ok',
    notes_json TEXT
);

CREATE TABLE llms_txt_links (
    asset_id TEXT REFERENCES llm_ready_assets(asset_id),
    link_text TEXT NOT NULL,
    link_url TEXT NOT NULL,
    optional BOOLEAN DEFAULT FALSE
);

-- Indexes for performance
CREATE INDEX idx_files_sha256 ON files(file_sha256);
CREATE INDEX idx_chunks_file_id ON chunks(file_id);
CREATE INDEX idx_runs_chunk_hash ON runs(chunk_hash);
CREATE INDEX idx_runs_job_key ON runs(job_key);
CREATE INDEX idx_hub_sync_dataset ON hub_sync(dataset_slug);
CREATE INDEX idx_hub_sync_status ON hub_sync(status);
CREATE INDEX idx_plan_archive_dataset ON plan_archive(dataset_slug);
CREATE INDEX idx_web_pages_site_id ON web_pages(site_id);
CREATE INDEX idx_web_pages_sha256 ON web_pages(content_sha256);
CREATE INDEX idx_gpu_telemetry_ts ON gpu_telemetry(ts_utc);

-- Views for common queries
CREATE VIEW dataset_summary AS
SELECT 
    dataset_slug,
    COUNT(DISTINCT file_id) as file_count,
    COUNT(DISTINCT chunk_hash) as chunk_count,
    COUNT(DISTINCT run_id) as run_count,
    SUM(cost_usd) as total_cost,
    AVG(token_in) as avg_tokens_in,
    AVG(token_out) as avg_tokens_out
FROM (
    SELECT 
        'dataset_' || strftime('%Y%m%d', created_at) as dataset_slug,
        f.file_id,
        c.chunk_hash,
        r.run_id,
        r.cost_usd,
        r.token_in,
        r.token_out,
        r.created_at
    FROM files f
    JOIN chunks c ON f.file_id = c.file_id
    JOIN runs r ON c.chunk_hash = r.chunk_hash
) sub
GROUP BY dataset_slug;

CREATE VIEW hub_sync_summary AS
SELECT 
    dataset_slug,
    content_type,
    COUNT(*) as total_items,
    COUNT(CASE WHEN status = 'pushed' THEN 1 END) as pushed_items,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_items
FROM hub_sync
GROUP BY dataset_slug, content_type; 