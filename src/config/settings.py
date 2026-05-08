from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    primary_llm_model: str = "claude-sonnet-4-6"
    fallback_llm_model: str = "gpt-4o"

    # LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "systematic-compliance-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "compliance_agent"
    postgres_user: str = "compliance"
    postgres_password: str = "compliance_secret"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_complaints_topic: str = "raw-complaints"
    kafka_dlq_topic: str = "complaints-dlq"
    kafka_consumer_group: str = "compliance-agent-group"

    # ChromaDB — use "local" (no server) or "server" (Docker HTTP)
    chroma_mode: str = "local"
    chroma_local_path: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_complaints: str = "cfpb_complaints"
    chroma_collection_regulatory: str = "regulatory_corpus"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "compliance-classifier"

    # Application
    log_level: str = "INFO"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    confidence_threshold: float = 0.72
    regulatory_review_pass_score: int = 80
    max_tokens_per_complaint: int = 512

    # Mock
    jira_base_url: str = "http://localhost:9090/jira"
    jira_project_key: str = "COMPLIANCE"
    mock_external_apis: bool = True

    # Embeddings — "local" (no API key) or "openai" (requires quota)
    embedding_provider: str = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 512

    # Classifier
    distilbert_model_name: str = "distilbert-base-uncased"
    classifier_checkpoint_path: str = "models/checkpoints/distilbert_cfpb"
    num_product_classes: int = 12
    num_issue_classes: int = 47

    # Fairness
    ffiec_data_path: str = "data/ffiec_demographic.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
