"""Configuration dataclass for experiment parameters."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # ── Task type ──
    task_type: str = "classification"  # "classification" or "regression"

    # Paths
    data_path: str = ""
    test_data_path: str = ""  # Optional separate test folder
    output_path: str = ""
    custom_script_path: str = ""  # Optional custom training script

    # Vast.ai
    api_key: str = ""
    ssh_key_path: str = ""

    # Model
    model_name: str = "resnet50"

    # Hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "adamw"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    pre_split_data: bool = False  # data_path contains train/val/test folders

    # Regression-specific
    target_column: str = ""  # Name of the column to predict
    feature_columns: str = ""  # Comma-separated feature column names (empty = all)

    # Augmentation flags (classification only)
    random_rotation: bool = False
    horizontal_flip: bool = False
    random_erasing: bool = False

    # Test mode
    use_builtin: bool = False  # Use CIFAR-100 instead of user dataset

    # Feature flags
    early_stopping: bool = True
    early_stopping_patience: int = 7
    grad_cam: bool = True
    lr_scheduler: bool = False
    mixup: bool = False
    label_smoothing: bool = False

    # Metrics to compute and plot
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "loss", "precision", "recall", "f1_score",
        "auc_roc", "confusion_matrix",
    ])

    # Dataset subsampling (0 = no limit)
    max_samples_per_class: int = 500

    # Vast.ai instance filters
    min_gpu_ram: float = 8.0
    max_price: float = 1.0

    TASK_TYPE_CHOICES = ["Classification", "Regression"]

    MODEL_CHOICES = {
        "ResNet-50": "resnet50",
        "DenseNet-121": "densenet121",
        "EfficientNet-B0": "efficientnet_b0",
        "ConvNeXt": "convnext",
    }

    OPTIMIZER_CHOICES = ["AdamW", "SGD"]

    CLASSIFICATION_METRICS = [
        "accuracy", "loss", "precision", "recall",
        "f1_score", "auc_roc", "confusion_matrix",
        "specificity", "sensitivity", "cohen_kappa",
    ]

    REGRESSION_METRICS = [
        "mse", "rmse", "mae", "r2", "loss",
        "explained_variance", "mape",
    ]

    def build_train_command(self) -> str:
        """Build the CLI command to run train.py with configured args."""
        parts = [
            "cd /workspace && python train.py",
            f"--task {self.task_type}",
            f"--model {self.model_name}",
            f"--lr {self.learning_rate}",
            f"--batch_size {self.batch_size}",
            f"--epochs {self.epochs}",
            f"--optimizer {self.optimizer.lower()}",
            f"--data_dir /workspace/data",
            f"--output_dir /workspace/output",
            f"--train_split {self.train_split}",
            f"--val_split {self.val_split}",
            f"--test_split {self.test_split}",
            f"--seed {self.seed}",
        ]

        if self.pre_split_data and not self.use_builtin:
            parts.append("--pre_split_data")

        # Built-in test dataset
        if self.use_builtin:
            parts.append("--use_builtin")

        # Test data
        if self.test_data_path and not self.use_builtin and not self.pre_split_data:
            parts.append("--test_dir /workspace/test_data")

        # Regression columns
        if self.task_type == "regression":
            if self.target_column:
                parts.append(f"--target_column {self.target_column}")
            if self.feature_columns:
                parts.append(f"--feature_columns {self.feature_columns}")

        # Augmentation (classification only)
        if self.task_type == "classification":
            if self.random_rotation:
                parts.append("--random_rotation")
            if self.horizontal_flip:
                parts.append("--horizontal_flip")
            if self.random_erasing:
                parts.append("--random_erasing")

        # Feature flags
        if self.early_stopping:
            parts.append(f"--early_stopping --patience {self.early_stopping_patience}")
        if self.grad_cam and self.task_type == "classification":
            parts.append("--grad_cam")
        if self.lr_scheduler:
            parts.append("--lr_scheduler")
        if self.mixup and self.task_type == "classification":
            parts.append("--mixup")
        if self.label_smoothing and self.task_type == "classification":
            parts.append("--label_smoothing")

        # Metrics
        if self.metrics:
            parts.append(f"--metrics {','.join(self.metrics)}")

        return " ".join(parts)
