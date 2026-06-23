-- Database initialization for ML pipeline metadata

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    source VARCHAR(500),
    num_samples INTEGER,
    split VARCHAR(50),
    schema_hash VARCHAR(64),
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    training_type VARCHAR(50) NOT NULL,  -- 'cpt' or 'sft'
    dataset_id INTEGER REFERENCES datasets(id),
    config JSONB,
    metrics JSONB,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES training_runs(run_id),
    benchmark VARCHAR(255) NOT NULL,
    metrics JSONB NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_quality_logs (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    check_name VARCHAR(255) NOT NULL,
    passed BOOLEAN NOT NULL,
    details JSONB,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_evaluations_benchmark ON evaluations(benchmark);
CREATE INDEX idx_data_quality_dataset ON data_quality_logs(dataset_id);
