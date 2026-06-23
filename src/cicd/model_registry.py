"""Model Registry: version, promote, and deploy models."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    name: str
    version: int
    stage: ModelStage
    metrics: dict
    artifact_path: str
    created_at: str
    promoted_at: str = None


class ModelRegistry:
    """Manage model lifecycle: register, validate, promote."""

    MODEL_NAME = "qwen3-vl-math"

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def register_model(self, run_id: str, artifact_path: str = "model") -> int:
        """Register a new model version from a training run."""
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(model_uri, self.MODEL_NAME)
        version = int(result.version)

        self.client.set_model_version_tag(self.MODEL_NAME, str(version), "stage", ModelStage.DEVELOPMENT.value)

        logger.info(f"Registered model version {version}")
        return version

    def validate_for_promotion(self, version: int, min_accuracy: float = 0.5) -> dict:
        """Validate model meets criteria for promotion."""
        mv = self.client.get_model_version(self.MODEL_NAME, str(version))
        run = self.client.get_run(mv.run_id)
        metrics = run.data.metrics

        checks = {
            "has_eval_metrics": "gsm8k_accuracy" in metrics or "math_accuracy" in metrics,
            "meets_accuracy_threshold": any(
                v >= min_accuracy
                for k, v in metrics.items()
                if "accuracy" in k
            ),
            "training_completed": metrics.get("final_train_loss", float("inf")) < 5.0,
            "no_nan_metrics": not any(
                v != v for v in metrics.values()  # NaN check
            ),
        }

        all_passed = all(checks.values())

        result = {
            "version": version,
            "passed": all_passed,
            "checks": checks,
            "metrics": dict(metrics),
        }

        if not all_passed:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Validation failed for version {version}: {failed}")
        else:
            logger.info(f"Version {version} passed all validation checks")

        return result

    def promote(self, version: int, target_stage: ModelStage):
        """Promote model to a new stage."""
        validation = self.validate_for_promotion(version)

        if not validation["passed"] and target_stage == ModelStage.PRODUCTION:
            raise ValueError(f"Model version {version} failed validation, cannot promote to production")

        self.client.set_model_version_tag(self.MODEL_NAME, str(version), "stage", target_stage.value)
        self.client.set_model_version_tag(
            self.MODEL_NAME, str(version), "promoted_at", datetime.now().isoformat()
        )

        logger.info(f"Promoted version {version} to {target_stage.value}")

    def get_production_model(self) -> ModelVersion | None:
        """Get current production model."""
        versions = self.client.search_model_versions(f"name='{self.MODEL_NAME}'")

        for v in versions:
            tags = v.tags or {}
            if tags.get("stage") == ModelStage.PRODUCTION.value:
                run = self.client.get_run(v.run_id)
                return ModelVersion(
                    name=self.MODEL_NAME,
                    version=int(v.version),
                    stage=ModelStage.PRODUCTION,
                    metrics=dict(run.data.metrics),
                    artifact_path=v.source,
                    created_at=str(v.creation_timestamp),
                    promoted_at=tags.get("promoted_at"),
                )

        return None

    def compare_versions(self, version_a: int, version_b: int) -> dict:
        """Compare two model versions."""
        mv_a = self.client.get_model_version(self.MODEL_NAME, str(version_a))
        mv_b = self.client.get_model_version(self.MODEL_NAME, str(version_b))

        metrics_a = dict(self.client.get_run(mv_a.run_id).data.metrics)
        metrics_b = dict(self.client.get_run(mv_b.run_id).data.metrics)

        comparison = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())

        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            if val_a is not None and val_b is not None:
                comparison[key] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "diff": val_b - val_a,
                    "improved": val_b > val_a if "loss" not in key else val_b < val_a,
                }

        return comparison
