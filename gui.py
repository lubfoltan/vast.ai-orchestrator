"""SCOUT — Desktop GUI for Vast.ai Training Orchestrator."""

import csv
import json
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from config import ExperimentConfig
from orchestrator import Orchestrator
from vast_api import VastAPI

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_USER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "user_config.json")


class App(ctk.CTk):
    """Main application window."""

    WIDTH = 1150
    HEIGHT = 900

    def __init__(self) -> None:
        super().__init__()
        self.title("SCOUT — Vast.ai Training Orchestrator")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(1050, 800)

        self._orchestrator: Optional[Orchestrator] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._ssh_console_active = False

        self._build_ui()
        self._load_user_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Top: scrollable config panel ---
        config_scroll = ctk.CTkScrollableFrame(self, height=420)
        config_scroll.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        config_scroll.grid_columnconfigure(0, weight=1)
        self._build_config_panel(config_scroll)

        # --- Middle: log console + telemetry chart ---
        mid_frame = ctk.CTkFrame(self)
        mid_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        mid_frame.grid_rowconfigure(0, weight=1)
        mid_frame.grid_columnconfigure(0, weight=1)

        self.log_box = ctk.CTkTextbox(mid_frame, state="disabled", font=("Consolas", 12))
        self.log_box.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # SSH input bar (hidden by default)
        self.ssh_input_frame = ctk.CTkFrame(mid_frame)
        self.ssh_input_var = ctk.StringVar()
        ctk.CTkLabel(self.ssh_input_frame, text="SSH >", font=("Consolas", 12)).pack(side="left", padx=(5, 2))
        self.ssh_input_entry = ctk.CTkEntry(
            self.ssh_input_frame, textvariable=self.ssh_input_var,
            font=("Consolas", 12), placeholder_text="Type command and press Enter…"
        )
        self.ssh_input_entry.pack(side="left", fill="x", expand=True, padx=2, pady=4)
        self.ssh_input_entry.bind("<Return>", self._on_ssh_send)
        ctk.CTkButton(self.ssh_input_frame, text="Send", width=60, command=self._on_ssh_send).pack(side="left", padx=2)
        ctk.CTkButton(self.ssh_input_frame, text="Exit Console", width=90,
                       fg_color="gray", command=self._on_ssh_exit).pack(side="left", padx=(2, 5))
        # NOT gridded yet — shown only when SSH Console is activated

        # Live telemetry chart (hidden by default)
        self._chart_visible = False
        self._chart_frame = ctk.CTkFrame(mid_frame)
        if _HAS_MATPLOTLIB:
            self._fig = Figure(figsize=(8, 2.5), dpi=80, facecolor="#1a1a2e")
            self._ax_loss = self._fig.add_subplot(121)
            self._ax_acc = self._fig.add_subplot(122)
            for ax in (self._ax_loss, self._ax_acc):
                ax.set_facecolor("#1a1a2e")
                ax.tick_params(colors="white", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#333")
            self._ax_loss.set_title("Loss", color="white", fontsize=9)
            self._ax_acc.set_title("Accuracy / Metric", color="white", fontsize=9)
            self._fig.tight_layout(pad=1.5)
            self._canvas = FigureCanvasTkAgg(self._fig, master=self._chart_frame)
            self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Bottom: action buttons ---
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        self._build_buttons(btn_frame)

    # ------------------------------------------------------------------
    def _build_config_panel(self, parent) -> None:
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", expand=True)
        f.grid_columnconfigure((1, 4), weight=1)

        row = 0
        # ── Task type ──
        ctk.CTkLabel(f, text="Task Type:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.task_type_var = ctk.StringVar(value="Classification")
        task_menu = ctk.CTkOptionMenu(
            f, variable=self.task_type_var,
            values=ExperimentConfig.TASK_TYPE_CHOICES,
            command=self._on_task_type_changed,
        )
        task_menu.grid(row=row, column=1, padx=5, pady=4, sticky="w")

        # ── Custom script ──
        ctk.CTkLabel(f, text="Custom Script:").grid(row=row, column=3, padx=5, pady=4, sticky="e")
        self.custom_script_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.custom_script_var,
                      placeholder_text="(Optional — uses built-in train.py if empty)").grid(
            row=row, column=4, padx=5, pady=4, sticky="ew"
        )
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_custom_script).grid(row=row, column=5, padx=2)

        row += 1
        # ── Paths ──
        ctk.CTkLabel(f, text="Train Data:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.data_path_var = ctk.StringVar()
        self._data_entry = ctk.CTkEntry(f, textvariable=self.data_path_var)
        self._data_entry.grid(row=row, column=1, columnspan=2, padx=5, pady=4, sticky="ew")
        self._data_browse_btn = ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_data)
        self._data_browse_btn.grid(row=row, column=3, padx=2)

        ctk.CTkLabel(f, text="Output Path:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.output_path_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.output_path_var).grid(row=row, column=5, padx=5, pady=4, sticky="ew")
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_output).grid(row=row, column=6, padx=2)

        row += 1
        self.test_data_label = ctk.CTkLabel(f, text="Test Data:")
        self.test_data_label.grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.test_data_path_var = ctk.StringVar()
        self._test_data_entry = ctk.CTkEntry(
            f,
            textvariable=self.test_data_path_var,
            placeholder_text="(Regression only; classification uses train/val/test split ratios)",
        )
        self._test_data_entry.grid(row=row, column=1, columnspan=2, padx=5, pady=4, sticky="ew")
        self._test_data_browse_btn = ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_test_data)
        self._test_data_browse_btn.grid(row=row, column=3, padx=2)

        ctk.CTkLabel(f, text="Seed:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.seed_var = ctk.StringVar(value="42")
        ctk.CTkEntry(f, textvariable=self.seed_var, width=70).grid(row=row, column=5, padx=5, pady=4, sticky="w")

        row += 1
        self.pre_split_data_var = ctk.BooleanVar(value=False)
        self.pre_split_data_cb = ctk.CTkCheckBox(
            f,
            text="Data already split into train/val/test",
            variable=self.pre_split_data_var,
            command=self._on_pre_split_toggled,
        )
        self.pre_split_data_cb.grid(row=row, column=1, columnspan=3, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(f, text="Split Ratio:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.ratio_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.ratio_frame.grid(row=row, column=5, columnspan=2, padx=5, pady=4, sticky="w")
        self.split_var = ctk.StringVar(value="0.8")
        self.val_split_var = ctk.StringVar(value="0.1")
        self.test_split_var = ctk.StringVar(value="0.1")
        ctk.CTkLabel(self.ratio_frame, text="train").pack(side="left", padx=(0, 2))
        self._train_split_entry = ctk.CTkEntry(self.ratio_frame, textvariable=self.split_var, width=52)
        self._train_split_entry.pack(side="left", padx=(0, 6))
        ctk.CTkLabel(self.ratio_frame, text="val").pack(side="left", padx=(0, 2))
        self._val_split_entry = ctk.CTkEntry(self.ratio_frame, textvariable=self.val_split_var, width=52)
        self._val_split_entry.pack(side="left", padx=(0, 6))
        ctk.CTkLabel(self.ratio_frame, text="test").pack(side="left", padx=(0, 2))
        self._test_split_entry = ctk.CTkEntry(self.ratio_frame, textvariable=self.test_split_var, width=52)
        self._test_split_entry.pack(side="left")

        row += 1
        # ── Test Mode (CIFAR-100) ──
        self.use_builtin_var = ctk.BooleanVar(value=False)
        self.use_builtin_cb = ctk.CTkCheckBox(
            f, text="Use Built-in Test Dataset (CIFAR-100)",
            variable=self.use_builtin_var,
            command=self._on_use_builtin_toggled,
        )
        self.use_builtin_cb.grid(row=row, column=1, columnspan=3, padx=5, pady=4, sticky="w")

        row += 1
        # ── API Key ──
        ctk.CTkLabel(f, text="API Key:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.api_key_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.api_key_var, show="•").grid(row=row, column=1, columnspan=6, padx=5, pady=4, sticky="ew")

        row += 1
        # ── SSH Key ──
        ctk.CTkLabel(f, text="SSH Key:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.ssh_key_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.ssh_key_var, placeholder_text="Path to private key").grid(
            row=row, column=1, columnspan=5, padx=5, pady=4, sticky="ew"
        )
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_ssh_key).grid(row=row, column=6, padx=2)

        row += 1
        # ── Regression fields (hidden by default) ──
        self.regression_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.regression_row = row  # remember for show/hide
        self.regression_parent = f

        ctk.CTkLabel(self.regression_frame, text="Target Column (predict):").pack(side="left", padx=5)
        self.target_col_var = ctk.StringVar()
        ctk.CTkEntry(self.regression_frame, textvariable=self.target_col_var, width=150,
                      placeholder_text="e.g. price").pack(side="left", padx=5)
        ctk.CTkLabel(self.regression_frame, text="Feature Columns:").pack(side="left", padx=5)
        self.feature_cols_var = ctk.StringVar()
        ctk.CTkEntry(self.regression_frame, textvariable=self.feature_cols_var, width=300,
                      placeholder_text="(comma-sep, empty = all except target)").pack(side="left", padx=5, fill="x", expand=True)
        # Initially hidden — will be gridded when task_type=Regression

        row += 1
        # ── Model & Optimizer ──
        ctk.CTkLabel(f, text="Model:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.model_var = ctk.StringVar(value="ResNet-50")
        ctk.CTkOptionMenu(f, variable=self.model_var, values=list(ExperimentConfig.MODEL_CHOICES.keys())).grid(
            row=row, column=1, padx=5, pady=4, sticky="ew"
        )
        ctk.CTkLabel(f, text="Optimizer:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.optimizer_var = ctk.StringVar(value="AdamW")
        ctk.CTkOptionMenu(f, variable=self.optimizer_var, values=ExperimentConfig.OPTIMIZER_CHOICES).grid(
            row=row, column=5, padx=5, pady=4, sticky="ew"
        )

        row += 1
        # ── Hyperparameters ──
        ctk.CTkLabel(f, text="Learning Rate:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.lr_var = ctk.StringVar(value="0.001")
        ctk.CTkEntry(f, textvariable=self.lr_var, width=100).grid(row=row, column=1, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(f, text="Batch Size:").grid(row=row, column=2, padx=5, pady=4, sticky="e")
        self.batch_var = ctk.IntVar(value=32)
        ctk.CTkSlider(f, from_=4, to=128, number_of_steps=31, variable=self.batch_var).grid(row=row, column=3, columnspan=2, padx=5, pady=4, sticky="ew")
        self.batch_label = ctk.CTkLabel(f, text="32")
        self.batch_label.grid(row=row, column=5, padx=5, sticky="w")
        self.batch_var.trace_add("write", lambda *_: self.batch_label.configure(text=str(self.batch_var.get())))

        ctk.CTkLabel(f, text="Epochs:").grid(row=row, column=6, padx=5, pady=4, sticky="e")
        self.epochs_var = ctk.IntVar(value=50)

        row += 1
        ctk.CTkLabel(f, text="Epochs:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        ctk.CTkEntry(f, textvariable=self.epochs_var, width=80).grid(row=row, column=1, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(f, text="Min GPU RAM (GB):").grid(row=row, column=2, padx=5, pady=4, sticky="e")
        self.min_gpu_ram_var = ctk.StringVar(value="8")
        ctk.CTkEntry(f, textvariable=self.min_gpu_ram_var, width=60).grid(row=row, column=3, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(f, text="Max $/hr:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.max_price_var = ctk.StringVar(value="1.0")
        ctk.CTkEntry(f, textvariable=self.max_price_var, width=60).grid(row=row, column=5, padx=5, pady=4, sticky="w")

        row += 1
        # ── Dataset subsampling ──
        ctk.CTkLabel(f, text="Max Samples/Class:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.max_samples_var = ctk.StringVar(value="0")
        ctk.CTkEntry(f, textvariable=self.max_samples_var, width=80).grid(row=row, column=1, padx=5, pady=4, sticky="w")
        ctk.CTkLabel(f, text="(0 = use full dataset; e.g. 250 → 250 NORMAL + 250 PNEUMONIA)").grid(
            row=row, column=2, columnspan=5, padx=5, pady=4, sticky="w"
        )

        row += 1
        # ── Resize (classification only) ──
        ctk.CTkLabel(f, text="Resize:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.resize_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.resize_frame.grid(row=row, column=1, columnspan=6, padx=5, pady=4, sticky="w")
        self.resize_enabled_var = ctk.BooleanVar(value=True)
        self.resize_cb = ctk.CTkCheckBox(
            self.resize_frame, text="Enable", variable=self.resize_enabled_var,
            command=self._on_resize_toggled,
        )
        self.resize_cb.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(self.resize_frame, text="width").pack(side="left", padx=(0, 2))
        self.resize_width_var = ctk.StringVar(value="224")
        self.resize_width_entry = ctk.CTkEntry(self.resize_frame, textvariable=self.resize_width_var, width=60)
        self.resize_width_entry.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(self.resize_frame, text="height").pack(side="left", padx=(0, 2))
        self.resize_height_var = ctk.StringVar(value="224")
        self.resize_height_entry = ctk.CTkEntry(self.resize_frame, textvariable=self.resize_height_var, width=60)
        self.resize_height_entry.pack(side="left", padx=(0, 8))

        row += 1
        # ── Augmentation (classification only) ──
        self.aug_label = ctk.CTkLabel(f, text="Augmentation:")
        self.aug_label.grid(row=row, column=0, padx=5, pady=4, sticky="ne")
        self.aug_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.aug_frame.grid(row=row, column=1, columnspan=6, padx=5, pady=4, sticky="w")
        self.aug_row = row

        self.aug_rotation_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.aug_frame, text="Random Rotation", variable=self.aug_rotation_var).pack(side="left", padx=6)
        self.aug_hflip_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.aug_frame, text="Horizontal Flip", variable=self.aug_hflip_var).pack(side="left", padx=6)
        self.aug_erasing_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.aug_frame, text="Random Erasing", variable=self.aug_erasing_var).pack(side="left", padx=6)

        row += 1
        # ── Feature Flags ──
        ctk.CTkLabel(f, text="Features:").grid(row=row, column=0, padx=5, pady=4, sticky="ne")
        feat_frame = ctk.CTkFrame(f, fg_color="transparent")
        feat_frame.grid(row=row, column=1, columnspan=6, padx=5, pady=4, sticky="w")

        self.early_stop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(feat_frame, text="Early Stopping", variable=self.early_stop_var).pack(side="left", padx=6)
        self.patience_var = ctk.IntVar(value=7)
        ctk.CTkLabel(feat_frame, text="patience:").pack(side="left", padx=(2, 0))
        ctk.CTkEntry(feat_frame, textvariable=self.patience_var, width=40).pack(side="left", padx=(2, 10))

        self.grad_cam_var = ctk.BooleanVar(value=True)
        self.grad_cam_cb = ctk.CTkCheckBox(feat_frame, text="Grad-CAM", variable=self.grad_cam_var)
        self.grad_cam_cb.pack(side="left", padx=6)
        self.lr_sched_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(feat_frame, text="LR Scheduler", variable=self.lr_sched_var).pack(side="left", padx=6)
        self.mixup_var = ctk.BooleanVar(value=False)
        self.mixup_cb = ctk.CTkCheckBox(feat_frame, text="Mixup", variable=self.mixup_var)
        self.mixup_cb.pack(side="left", padx=6)
        self.label_smooth_var = ctk.BooleanVar(value=False)
        self.label_smooth_cb = ctk.CTkCheckBox(feat_frame, text="Label Smoothing", variable=self.label_smooth_var)
        self.label_smooth_cb.pack(side="left", padx=6)

        row += 1
        # ── Metrics (dynamically rebuilt based on task type) ──
        ctk.CTkLabel(f, text="Metrics:").grid(row=row, column=0, padx=5, pady=4, sticky="ne")
        self.met_frame = ctk.CTkFrame(f, fg_color="transparent")
        self.met_frame.grid(row=row, column=1, columnspan=6, padx=5, pady=4, sticky="w")
        self.metrics_row = row

        self.metric_vars = {}
        self._rebuild_metrics("Classification")

    # ------------------------------------------------------------------
    def _rebuild_metrics(self, task_type: str) -> None:
        """Destroy and rebuild metric checkboxes based on task type."""
        for w in self.met_frame.winfo_children():
            w.destroy()
        self.metric_vars.clear()

        if task_type == "Classification":
            choices = ExperimentConfig.CLASSIFICATION_METRICS
            defaults_on = {"accuracy", "loss", "precision", "recall", "f1_score", "auc_roc", "confusion_matrix"}
        else:
            choices = ExperimentConfig.REGRESSION_METRICS
            defaults_on = {"mse", "rmse", "mae", "r2", "loss"}

        for m in choices:
            var = ctk.BooleanVar(value=(m in defaults_on))
            self.metric_vars[m] = var
            label = m.replace("_", " ").upper() if m in ("mse", "rmse", "mae", "r2", "mape") else m.replace("_", " ").title()
            ctk.CTkCheckBox(self.met_frame, text=label, variable=var).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    def _on_use_builtin_toggled(self) -> None:
        """Enable/disable data path inputs when test mode is toggled."""
        builtin = self.use_builtin_var.get()
        state = "disabled" if builtin else "normal"
        self._data_entry.configure(state=state)
        self._data_browse_btn.configure(state=state)
        self.pre_split_data_cb.configure(state=state)
        self._on_pre_split_toggled()

    # ------------------------------------------------------------------
    def _on_pre_split_toggled(self) -> None:
        """Toggle fields that only apply when SCOUT creates the split."""
        builtin = self.use_builtin_var.get()
        pre_split = self.pre_split_data_var.get()
        is_regression = self.task_type_var.get() == "Regression"
        data_state = "disabled" if builtin else "normal"
        test_state = "normal" if is_regression and not builtin and not pre_split else "disabled"
        ratio_state = "disabled" if pre_split else "normal"

        self._data_entry.configure(state=data_state)
        self._data_browse_btn.configure(state=data_state)
        self.pre_split_data_cb.configure(state=data_state)
        self._test_data_entry.configure(state=test_state)
        self._test_data_browse_btn.configure(state=test_state)
        for child in self.ratio_frame.winfo_children():
            try:
                child.configure(state=ratio_state)
            except Exception:
                pass

    def _on_resize_toggled(self) -> None:
        """Enable/disable resize dimension inputs."""
        is_classification = self.task_type_var.get() == "Classification"
        state = "normal" if is_classification and self.resize_enabled_var.get() else "disabled"
        self.resize_cb.configure(state="normal" if is_classification else "disabled")
        self.resize_width_entry.configure(state=state)
        self.resize_height_entry.configure(state=state)

    # ------------------------------------------------------------------
    def _on_task_type_changed(self, choice: str) -> None:
        """Show/hide fields based on classification vs regression."""
        self._rebuild_metrics(choice)

        if choice == "Regression":
            # Show regression fields
            self.regression_frame.grid(
                row=self.regression_row, column=0, columnspan=7, padx=5, pady=4, sticky="ew",
                in_=self.regression_parent,
            )
            # Hide classification-only widgets
            self.aug_label.grid_remove()
            self.aug_frame.grid_remove()
            self.grad_cam_cb.configure(state="disabled")
            self.mixup_cb.configure(state="disabled")
            self.label_smooth_cb.configure(state="disabled")
            self.test_data_label.configure(text="Test Data:")
        else:
            # Hide regression fields
            self.regression_frame.grid_remove()
            # Show classification widgets
            self.aug_label.grid()
            self.aug_frame.grid()
            self.grad_cam_cb.configure(state="normal")
            self.mixup_cb.configure(state="normal")
            self.label_smooth_cb.configure(state="normal")
            self.test_data_label.configure(text="Test Data:")
        self._on_pre_split_toggled()
        self._on_resize_toggled()

    # ------------------------------------------------------------------
    def _build_buttons(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)

        self.btn_start = ctk.CTkButton(parent, text="▶  Start Pipeline", command=self._on_start, fg_color="green")
        self.btn_start.grid(row=0, column=0, padx=6, pady=8, sticky="ew")

        self.btn_attach = ctk.CTkButton(
            parent, text="🔗  Connect Existing", command=self._on_attach, fg_color="#6366f1"
        )
        self.btn_attach.grid(row=0, column=1, padx=6, pady=8, sticky="ew")

        self.btn_cancel = ctk.CTkButton(parent, text="⏹  Stop Instance", command=self._on_cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=2, padx=6, pady=8, sticky="ew")

        self.btn_destroy = ctk.CTkButton(
            parent, text="🗑  Terminate", command=self._on_destroy, fg_color="red", state="disabled"
        )
        self.btn_destroy.grid(row=0, column=3, padx=6, pady=8, sticky="ew")

        self.btn_ssh_console = ctk.CTkButton(
            parent, text="💻  SSH Console", command=self._on_ssh_console, state="disabled"
        )
        self.btn_ssh_console.grid(row=0, column=4, padx=6, pady=8, sticky="ew")

        self.btn_files = ctk.CTkButton(
            parent, text="📂  Remote Files", command=self._on_toggle_file_browser, state="disabled"
        )
        self.btn_files.grid(row=0, column=5, padx=6, pady=8, sticky="ew")

        self.btn_download = ctk.CTkButton(
            parent, text="⬇  Download Results", command=self._on_download_results, state="disabled"
        )
        self.btn_download.grid(row=0, column=6, padx=6, pady=8, sticky="ew")

        self.btn_chart = ctk.CTkButton(
            parent, text="📈  Live Chart", command=self._on_toggle_chart,
            state="normal" if _HAS_MATPLOTLIB else "disabled"
        )
        self.btn_chart.grid(row=0, column=7, padx=6, pady=8, sticky="ew")

        self.btn_clear = ctk.CTkButton(parent, text="Clear Log", command=self._clear_log, fg_color="gray")
        self.btn_clear.grid(row=1, column=0, padx=6, pady=(0, 6), sticky="ew")

        # Status label for attach mode
        self._status_label = ctk.CTkLabel(parent, text="", font=("Consolas", 11))
        self._status_label.grid(row=1, column=1, columnspan=7, padx=6, pady=(0, 6), sticky="w")

    # ==================================================================
    # Helpers
    # ==================================================================
    def _browse_data(self) -> None:
        path = filedialog.askdirectory(title="Select Train Dataset Folder")
        if path:
            self.data_path_var.set(path)

    def _browse_test_data(self) -> None:
        path = filedialog.askdirectory(title="Select Test Dataset Folder")
        if path:
            self.test_data_path_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_path_var.set(path)

    def _browse_ssh_key(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SSH Private Key",
            filetypes=[("All files", "*.*"), ("PEM files", "*.pem")],
        )
        if path:
            self.ssh_key_var.set(path)

    def _browse_custom_script(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Custom Training Script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.custom_script_var.set(path)

    def _append_log(self, msg: str) -> None:
        """Thread-safe log append."""
        self.after(0, self._insert_log_line, msg)

    def _insert_log_line(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Persistent user config
    # ------------------------------------------------------------------
    def _load_user_config(self) -> None:
        """Load saved settings from user_config.json and populate GUI fields."""
        if not os.path.isfile(_USER_CONFIG_PATH):
            return
        try:
            with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        self.api_key_var.set(cfg.get("api_key", ""))
        self.ssh_key_var.set(cfg.get("ssh_key_path", ""))
        self.data_path_var.set(cfg.get("data_path", ""))
        self.test_data_path_var.set(cfg.get("test_data_path", ""))
        self.output_path_var.set(cfg.get("output_path", ""))
        self.custom_script_var.set(cfg.get("custom_script_path", ""))
        self.task_type_var.set(cfg.get("task_type", "Classification"))
        self.use_builtin_var.set(bool(cfg.get("use_builtin", False)))
        self.pre_split_data_var.set(bool(cfg.get("pre_split_data", False)))
        self.model_var.set(cfg.get("model", "ResNet-50"))
        self.optimizer_var.set(cfg.get("optimizer", "AdamW"))
        self.lr_var.set(cfg.get("learning_rate", "0.001"))
        self.batch_var.set(int(cfg.get("batch_size", 32)))
        self.epochs_var.set(int(cfg.get("epochs", 50)))
        self.split_var.set(cfg.get("train_split", "0.8"))
        self.val_split_var.set(cfg.get("val_split", "0.1"))
        self.test_split_var.set(cfg.get("test_split", "0.1"))
        self.seed_var.set(str(cfg.get("seed", 42)))
        self.min_gpu_ram_var.set(cfg.get("min_gpu_ram", "8"))
        self.max_price_var.set(cfg.get("max_price", "1.0"))
        self.max_samples_var.set(str(cfg.get("max_samples_per_class", 0)))
        self.resize_enabled_var.set(bool(cfg.get("resize_enabled", True)))
        self.resize_width_var.set(str(cfg.get("resize_width", 224)))
        self.resize_height_var.set(str(cfg.get("resize_height", 224)))
        self._on_task_type_changed(self.task_type_var.get())
        self._on_use_builtin_toggled()
        self._on_resize_toggled()

    def _save_user_config(self) -> None:
        """Persist current GUI settings to user_config.json."""
        cfg = {
            "api_key": self.api_key_var.get(),
            "ssh_key_path": self.ssh_key_var.get(),
            "data_path": self.data_path_var.get(),
            "test_data_path": self.test_data_path_var.get(),
            "output_path": self.output_path_var.get(),
            "custom_script_path": self.custom_script_var.get(),
            "task_type": self.task_type_var.get(),
            "use_builtin": self.use_builtin_var.get(),
            "pre_split_data": self.pre_split_data_var.get(),
            "model": self.model_var.get(),
            "optimizer": self.optimizer_var.get(),
            "learning_rate": self.lr_var.get(),
            "batch_size": self.batch_var.get(),
            "epochs": self.epochs_var.get(),
            "train_split": self.split_var.get(),
            "val_split": self.val_split_var.get(),
            "test_split": self.test_split_var.get(),
            "seed": self.seed_var.get(),
            "min_gpu_ram": self.min_gpu_ram_var.get(),
            "max_price": self.max_price_var.get(),
            "max_samples_per_class": int(self.max_samples_var.get() or 0),
            "resize_enabled": self.resize_enabled_var.get(),
            "resize_width": self.resize_width_var.get(),
            "resize_height": self.resize_height_var.get(),
        }
        try:
            with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
        except OSError:
            pass

    def _on_close(self) -> None:
        """Save settings and exit."""
        self._save_user_config()
        self.destroy()

    def _build_config(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.task_type = self.task_type_var.get().lower()
        cfg.use_builtin = self.use_builtin_var.get()
        cfg.pre_split_data = self.pre_split_data_var.get()
        cfg.data_path = self.data_path_var.get().strip()
        cfg.test_data_path = self.test_data_path_var.get().strip()
        cfg.output_path = self.output_path_var.get().strip()
        cfg.custom_script_path = self.custom_script_var.get().strip()
        cfg.api_key = self.api_key_var.get().strip()
        cfg.ssh_key_path = self.ssh_key_var.get().strip()
        cfg.model_name = ExperimentConfig.MODEL_CHOICES[self.model_var.get()]
        cfg.optimizer = self.optimizer_var.get()
        try:
            cfg.learning_rate = float(self.lr_var.get())
        except ValueError:
            cfg.learning_rate = 0.001
        cfg.batch_size = self.batch_var.get()
        cfg.epochs = self.epochs_var.get()
        try:
            cfg.train_split = float(self.split_var.get())
        except ValueError:
            cfg.train_split = 0.8
        try:
            cfg.val_split = float(self.val_split_var.get())
        except ValueError:
            cfg.val_split = 0.1
        try:
            cfg.test_split = float(self.test_split_var.get())
        except ValueError:
            cfg.test_split = 0.1
        try:
            cfg.seed = int(self.seed_var.get())
        except ValueError:
            cfg.seed = 42
        # Regression fields
        cfg.target_column = self.target_col_var.get().strip()
        cfg.feature_columns = self.feature_cols_var.get().strip()
        # Augmentation
        cfg.random_rotation = self.aug_rotation_var.get()
        cfg.horizontal_flip = self.aug_hflip_var.get()
        cfg.random_erasing = self.aug_erasing_var.get()
        cfg.resize_enabled = self.resize_enabled_var.get()
        try:
            cfg.resize_width = int(self.resize_width_var.get())
        except ValueError:
            cfg.resize_width = 224
        try:
            cfg.resize_height = int(self.resize_height_var.get())
        except ValueError:
            cfg.resize_height = 224
        # Features
        cfg.early_stopping = self.early_stop_var.get()
        cfg.early_stopping_patience = self.patience_var.get()
        cfg.grad_cam = self.grad_cam_var.get()
        cfg.lr_scheduler = self.lr_sched_var.get()
        cfg.mixup = self.mixup_var.get()
        cfg.label_smoothing = self.label_smooth_var.get()
        cfg.metrics = [m for m, var in self.metric_vars.items() if var.get()]
        try:
            cfg.min_gpu_ram = float(self.min_gpu_ram_var.get())
        except ValueError:
            cfg.min_gpu_ram = 8.0
        try:
            cfg.max_price = float(self.max_price_var.get())
        except ValueError:
            cfg.max_price = 1.0
        try:
            cfg.max_samples_per_class = int(self.max_samples_var.get() or 0)
        except ValueError:
            cfg.max_samples_per_class = 0
        return cfg

    # ==================================================================
    # Button callbacks
    # ==================================================================
    def _validate_inputs(self, attach_mode: bool = False) -> Optional[str]:
        use_builtin = self.use_builtin_var.get()
        pre_split = self.pre_split_data_var.get()
        data_path = self.data_path_var.get().strip()
        if not attach_mode:
            if not use_builtin and not data_path:
                return "Dataset path is required (or enable built-in CIFAR-100)."
            if not use_builtin and not os.path.isdir(data_path):
                return f"Dataset folder not found: {data_path}"
            if not use_builtin and pre_split:
                train_dir = os.path.join(data_path, "train")
                val_dir = os.path.join(data_path, "val")
                validation_dir = os.path.join(data_path, "validation")
                test_dir = os.path.join(data_path, "test")
                if not os.path.isdir(train_dir) or not (os.path.isdir(val_dir) or os.path.isdir(validation_dir)) or not os.path.isdir(test_dir):
                    return "Pre-split dataset must contain train, val (or validation), and test folders."
            if not self.api_key_var.get().strip():
                return "Vast.ai API Key is required."
        if not pre_split:
            try:
                train_ratio = float(self.split_var.get())
                val_ratio = float(self.val_split_var.get())
                test_ratio = float(self.test_split_var.get())
            except ValueError:
                return "Train/val/test ratios must be numbers."
            if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
                return "Train/val/test ratios must be greater than 0."
            if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-4:
                return "Train + val + test ratios must equal 1.0."
        try:
            int(self.seed_var.get())
        except ValueError:
            return "Seed must be an integer."
        test_path = self.test_data_path_var.get().strip()
        if self.task_type_var.get() == "Regression" and test_path and not pre_split and not os.path.isdir(test_path):
            return f"Test Data folder not found: {test_path}"
        if not self.output_path_var.get().strip():
            return "Output Path is required."
        if self.task_type_var.get() == "Classification" and self.resize_enabled_var.get():
            try:
                resize_width = int(self.resize_width_var.get())
                resize_height = int(self.resize_height_var.get())
            except ValueError:
                return "Resize width and height must be integers."
            if resize_width <= 0 or resize_height <= 0:
                return "Resize width and height must be greater than 0, or disable resize."
        ssh_key = self.ssh_key_var.get().strip()
        if not ssh_key:
            return "SSH Key path is required."
        if not os.path.isfile(ssh_key):
            return f"SSH Key file not found: {ssh_key}"
        metrics = [m for m, var in self.metric_vars.items() if var.get()]
        if not metrics:
            return "Select at least one metric."
        # Regression requires target column
        if self.task_type_var.get() == "Regression" and not self.target_col_var.get().strip():
            return "Target Column is required for regression."
        # Custom script must exist if specified
        cs = self.custom_script_var.get().strip()
        if cs and not os.path.isfile(cs):
            return f"Custom script not found: {cs}"
        return None

    def _on_start(self) -> None:
        # If already attached to an existing instance, run in attached mode
        if self._orchestrator and self._orchestrator._attached and self._orchestrator.ssh and self._orchestrator.ssh.is_connected:
            self._on_start_attached()
            return

        err = self._validate_inputs()
        if err:
            self._append_log(f"⚠  Validation error: {err}")
            return

        cfg = self._build_config()
        self._orchestrator = Orchestrator(
            cfg, log_cb=self._append_log,
            telemetry_cb=self._on_telemetry_data,
        )

        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_destroy.configure(state="disabled")
        self.btn_ssh_console.configure(state="disabled")
        self.btn_download.configure(state="disabled")

        self._worker_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._worker_thread.start()

    def _run_pipeline(self) -> None:
        try:
            self._orchestrator.run()
        finally:
            self.after(0, self._pipeline_finished)

    def _pipeline_finished(self) -> None:
        if self._orchestrator and self._orchestrator._attached:
            self.btn_start.configure(text="▶  Start Training", state="normal")
        else:
            self.btn_start.configure(text="▶  Start Pipeline", state="normal")
        self.btn_attach.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        if self._orchestrator and self._orchestrator.instance_id:
            self.btn_destroy.configure(state="normal")
            # Enable SSH console if we have a live SSH connection
            if self._orchestrator.ssh and self._orchestrator.ssh.is_connected:
                self.btn_ssh_console.configure(state="normal")
                self.btn_files.configure(state="normal")
                self.btn_download.configure(state="normal")
        elif self._orchestrator and self._orchestrator._attached:
            if self._orchestrator.ssh and self._orchestrator.ssh.is_connected:
                self.btn_ssh_console.configure(state="normal")
                self.btn_files.configure(state="normal")
                self.btn_download.configure(state="normal")

    def _on_cancel(self) -> None:
        if self._orchestrator:
            self._orchestrator.cancel()
            self._append_log("Cancellation requested — stopping instance…")
            threading.Thread(target=self._do_stop, daemon=True).start()
        self.btn_cancel.configure(state="disabled")

    def _do_stop(self) -> None:
        self._orchestrator.stop_instance()

    def _on_destroy(self) -> None:
        if self._orchestrator:
            threading.Thread(target=self._do_destroy, daemon=True).start()

    def _do_destroy(self) -> None:
        self._orchestrator.destroy_instance()
        self.after(0, lambda: self.btn_destroy.configure(state="disabled"))
        self.after(0, lambda: self.btn_ssh_console.configure(state="disabled"))
        self.after(0, lambda: self.btn_download.configure(state="disabled"))

    # ==================================================================
    # Attach to Existing Instance
    # ==================================================================
    def _on_attach(self) -> None:
        """Open a popup to input SSH host/port and connect to existing instance."""
        popup = ctk.CTkToplevel(self)
        popup.title("Connect to Existing Instance")
        popup.geometry("420x220")
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(popup, text="SSH Host:", font=("Consolas", 12)).grid(
            row=0, column=0, padx=10, pady=(15, 5), sticky="e"
        )
        host_var = ctk.StringVar(value="ssh5.vast.ai")
        ctk.CTkEntry(popup, textvariable=host_var, width=240).grid(
            row=0, column=1, padx=10, pady=(15, 5), sticky="ew"
        )

        ctk.CTkLabel(popup, text="SSH Port:", font=("Consolas", 12)).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        port_var = ctk.StringVar(value="22")
        ctk.CTkEntry(popup, textvariable=port_var, width=240).grid(
            row=1, column=1, padx=10, pady=5, sticky="ew"
        )

        ctk.CTkLabel(popup, text="Instance ID:", font=("Consolas", 12)).grid(
            row=2, column=0, padx=10, pady=5, sticky="e"
        )
        iid_var = ctk.StringVar()
        ctk.CTkEntry(popup, textvariable=iid_var, width=240,
                      placeholder_text="(optional — for stop/destroy)").grid(
            row=2, column=1, padx=10, pady=5, sticky="ew"
        )

        status_lbl = ctk.CTkLabel(popup, text="", text_color="red")
        status_lbl.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        def do_connect():
            host = host_var.get().strip()
            port_str = port_var.get().strip()
            if not host:
                status_lbl.configure(text="Host is required.")
                return
            try:
                port = int(port_str)
            except ValueError:
                status_lbl.configure(text="Port must be a number.")
                return

            ssh_key = self.ssh_key_var.get().strip()
            if not ssh_key or not os.path.isfile(ssh_key):
                status_lbl.configure(text="SSH Key path is missing or invalid.")
                return

            status_lbl.configure(text="Connecting…", text_color="yellow")
            popup.update()

            cfg = self._build_config()
            self._orchestrator = Orchestrator(
                cfg, log_cb=self._append_log,
                telemetry_cb=self._on_telemetry_data,
            )

            # Try to parse instance ID
            iid = iid_var.get().strip()
            if iid:
                try:
                    self._orchestrator.instance_id = int(iid)
                    if cfg.api_key:
                        self._orchestrator.vast = VastAPI(cfg.api_key)
                except ValueError:
                    pass

            # Connect in background
            def _do():
                try:
                    self._orchestrator.attach(host, port)
                    self.after(0, _on_success)
                except Exception as exc:
                    self.after(0, lambda: status_lbl.configure(
                        text=f"Failed: {exc}", text_color="red"))

            def _on_success():
                popup.destroy()
                self._status_label.configure(text=f"🟢 Attached: {host}:{port}")
                self.btn_start.configure(text="▶  Start Training", state="normal")
                self.btn_ssh_console.configure(state="normal")
                self.btn_files.configure(state="normal")
                self.btn_download.configure(state="normal")
                self.btn_cancel.configure(state="normal")
                if self._orchestrator.instance_id:
                    self.btn_destroy.configure(state="normal")
                self._append_log(f"Attached to instance at {host}:{port}")

            threading.Thread(target=_do, daemon=True).start()

        ctk.CTkButton(popup, text="Connect", fg_color="green", command=do_connect).grid(
            row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

    def _on_start_attached(self) -> None:
        """Run the pipeline on the attached instance."""
        err = self._validate_inputs(attach_mode=True)
        if err:
            self._append_log(f"⚠  Validation error: {err}")
            return

        cfg = self._build_config()
        # Update the existing orchestrator's config
        self._orchestrator.config = cfg

        self.btn_start.configure(text="▶  Training…", state="disabled")
        self.btn_attach.configure(state="disabled")

        def _run():
            try:
                self._orchestrator.run_attached()
            finally:
                self.after(0, self._pipeline_finished)

        self._worker_thread = threading.Thread(target=_run, daemon=True)
        self._worker_thread.start()

    def _on_download_results(self) -> None:
        """Download /workspace/output from the active SSH session."""
        if not self._orchestrator or not self._orchestrator.ssh or not self._orchestrator.ssh.is_connected:
            self._append_log("⚠  No active SSH connection for result download.")
            return

        cfg = self._build_config()
        self._orchestrator.config = cfg
        self.btn_download.configure(state="disabled")

        def _do():
            try:
                self._orchestrator.download_results()
            except Exception as exc:
                self._append_log(f"⚠  Download failed: {exc}")
            finally:
                self.after(0, lambda: self.btn_download.configure(state="normal"))

        threading.Thread(target=_do, daemon=True).start()

    # ==================================================================
    # Remote File Browser
    # ==================================================================
    def _on_toggle_file_browser(self) -> None:
        """Open a remote file browser window."""
        if not self._orchestrator or not self._orchestrator.ssh or not self._orchestrator.ssh.is_connected:
            self._append_log("⚠  No active SSH connection.")
            return

        fb = ctk.CTkToplevel(self)
        fb.title("Remote File Browser — /workspace")
        fb.geometry("600x500")
        fb.transient(self)

        # Toolbar
        toolbar = ctk.CTkFrame(fb)
        toolbar.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(toolbar, text="Refresh", width=80,
                       command=lambda: self._refresh_file_browser(fb, tree_frame)).pack(side="left", padx=3)
        ctk.CTkButton(toolbar, text="Delete /output", width=120, fg_color="red",
                       command=lambda: self._delete_remote_output(fb, tree_frame)).pack(side="left", padx=3)
        ctk.CTkButton(toolbar, text="Check Data", width=100,
                       command=self._check_remote_data).pack(side="left", padx=3)

        # Tree area (scrollable)
        tree_frame = ctk.CTkScrollableFrame(fb)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Initial load
        self._refresh_file_browser(fb, tree_frame)

    def _refresh_file_browser(self, window, tree_frame) -> None:
        """Reload the remote file tree into the scrollable frame."""
        # Clear existing
        for w in tree_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(tree_frame, text="Loading…", font=("Consolas", 11)).pack(anchor="w")
        window.update()

        def _load():
            try:
                tree = self._orchestrator.ssh.get_remote_file_structure(
                    "/workspace", max_depth=3
                )
                self.after(0, lambda: self._render_file_tree(tree_frame, tree, 0))
            except Exception as exc:
                self.after(0, lambda: self._render_file_error(tree_frame, str(exc)))

        threading.Thread(target=_load, daemon=True).start()

    def _render_file_tree(self, parent, nodes: list, depth: int) -> None:
        """Render the file tree as indented labels."""
        for w in parent.winfo_children():
            w.destroy()
        self._render_nodes(parent, nodes, depth)

    def _render_nodes(self, parent, nodes: list, depth: int) -> None:
        for node in nodes:
            indent = "    " * depth
            icon = "📁" if node["is_dir"] else "📄"
            size_str = ""
            if not node["is_dir"] and node["size"] > 0:
                if node["size"] > 1024 * 1024:
                    size_str = f"  ({node['size'] / (1024*1024):.1f} MB)"
                elif node["size"] > 1024:
                    size_str = f"  ({node['size'] / 1024:.0f} KB)"
                else:
                    size_str = f"  ({node['size']} B)"
            text = f"{indent}{icon} {node['name']}{size_str}"
            ctk.CTkLabel(parent, text=text, font=("Consolas", 11), anchor="w").pack(
                fill="x", padx=2, pady=0
            )
            if node.get("children"):
                self._render_nodes(parent, node["children"], depth + 1)

    def _render_file_error(self, parent, msg: str) -> None:
        for w in parent.winfo_children():
            w.destroy()
        ctk.CTkLabel(parent, text=f"Error: {msg}", text_color="red").pack(anchor="w")

    def _delete_remote_output(self, window, tree_frame) -> None:
        """Delete /workspace/output on the remote server and refresh."""
        if not self._orchestrator or not self._orchestrator.ssh:
            return

        def _do():
            try:
                self._orchestrator.ssh.delete_remote_path("/workspace/output", log_cb=self._append_log)
                self._orchestrator.ssh.exec_command("mkdir -p /workspace/output", log_cb=self._append_log)
                self._append_log("Remote /workspace/output cleared.")
                self.after(0, lambda: self._refresh_file_browser(window, tree_frame))
            except Exception as exc:
                self._append_log(f"⚠  Delete failed: {exc}")

        threading.Thread(target=_do, daemon=True).start()

    def _check_remote_data(self) -> None:
        """Check if /workspace/data exists and report."""
        if not self._orchestrator or not self._orchestrator.ssh:
            return

        def _do():
            try:
                status = self._orchestrator.ssh.verify_remote_data(log_cb=self._append_log)
                summary = ", ".join(
                    f"{k}: {'✓' if v else '✗'}" for k, v in status.items()
                )
                self._append_log(f"Remote status: {summary}")
            except Exception as exc:
                self._append_log(f"⚠  Check failed: {exc}")

        threading.Thread(target=_do, daemon=True).start()

    # ==================================================================
    # Live Telemetry Chart
    # ==================================================================
    def _on_toggle_chart(self) -> None:
        """Show/hide the live training chart."""
        if not _HAS_MATPLOTLIB:
            return
        if self._chart_visible:
            self._chart_frame.grid_remove()
            self._chart_visible = False
            self.btn_chart.configure(text="📈  Live Chart")
        else:
            self._chart_frame.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="nsew")
            self._chart_visible = True
            self.btn_chart.configure(text="📈  Hide Chart")

    def _on_telemetry_data(self, rows: list) -> None:
        """Called by orchestrator with parsed telemetry CSV rows. Thread-safe via after()."""
        self.after(0, self._update_chart, rows)

    def _update_chart(self, rows: list) -> None:
        """Redraw the live chart with latest telemetry data."""
        if not _HAS_MATPLOTLIB or not rows:
            return

        epochs = [int(r.get("epoch", 0)) for r in rows]
        train_loss = [float(r.get("train_loss", 0)) for r in rows]
        val_loss = [float(r.get("val_loss", 0)) for r in rows]
        test_loss = [float(r.get("test_loss", 0)) for r in rows] if "test_loss" in rows[0] else []

        self._ax_loss.clear()
        self._ax_loss.plot(epochs, train_loss, "c-", linewidth=1.2, label="Train")
        self._ax_loss.plot(epochs, val_loss, "r-", linewidth=1.2, label="Val")
        if test_loss:
            self._ax_loss.plot(epochs, test_loss, color="#fbbf24", linewidth=1.2, label="Test")
        self._ax_loss.set_title("Loss", color="white", fontsize=9)
        loss_values = train_loss + val_loss + test_loss
        if loss_values:
            self._ax_loss.set_ylim(0, max(max(loss_values) * 1.08, 1e-6))
        self._ax_loss.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        self._ax_loss.tick_params(colors="white", labelsize=7)

        self._ax_acc.clear()
        # Plot accuracy if available, otherwise first non-loss metric
        if "val_acc" in rows[0]:
            train_acc = [float(r.get("train_acc", 0)) for r in rows]
            val_acc = [float(r.get("val_acc", 0)) for r in rows]
            test_acc = [float(r.get("test_acc", 0)) for r in rows] if "test_acc" in rows[0] else []
            self._ax_acc.plot(epochs, train_acc, "c-", linewidth=1.2, label="Train Acc")
            self._ax_acc.plot(epochs, val_acc, "r-", linewidth=1.2, label="Val Acc")
            if test_acc:
                self._ax_acc.plot(epochs, test_acc, color="#fbbf24", linewidth=1.2, label="Test Acc")
            self._ax_acc.set_ylim(0, 1)
            self._ax_acc.set_title("Accuracy", color="white", fontsize=9)
        else:
            # Plot whatever metric columns exist (skip epoch, timestamp, losses)
            skip = {"epoch", "timestamp", "train_loss", "val_loss", "test_loss"}
            plotted = 0
            colors = ["#22d3ee", "#f472b6", "#a78bfa", "#34d399", "#fbbf24"]
            for key in rows[0]:
                if key in skip or plotted >= 3:
                    continue
                try:
                    vals = [float(r.get(key, 0)) for r in rows]
                    self._ax_acc.plot(epochs, vals, color=colors[plotted % len(colors)],
                                     linewidth=1.2, label=key.upper())
                    plotted += 1
                except (ValueError, TypeError):
                    pass
            self._ax_acc.set_title("Metrics", color="white", fontsize=9)

        self._ax_acc.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        self._ax_acc.tick_params(colors="white", labelsize=7)
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

        # Auto-show chart on first data
        if not self._chart_visible:
            self._on_toggle_chart()

    # ==================================================================
    # Interactive SSH Console
    # ==================================================================
    def _on_ssh_console(self) -> None:
        """Toggle the interactive SSH console bar."""
        if not self._orchestrator or not self._orchestrator.ssh or not self._orchestrator.ssh.is_connected:
            self._append_log("⚠  No active SSH connection. Run the pipeline first.")
            return

        self._ssh_console_active = True
        self.ssh_input_frame.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")
        self.ssh_input_entry.focus_set()
        self._append_log("═══ SSH Console opened. Type commands below. Click 'Exit Console' to close. ═══")

    def _on_ssh_send(self, event=None) -> None:
        """Execute the typed command on the remote server."""
        cmd = self.ssh_input_var.get().strip()
        if not cmd:
            return
        self.ssh_input_var.set("")

        if not self._orchestrator or not self._orchestrator.ssh or not self._orchestrator.ssh.is_connected:
            self._append_log("⚠  SSH connection lost.")
            return

        self._append_log(f">>> {cmd}")
        threading.Thread(target=self._ssh_exec_command, args=(cmd,), daemon=True).start()

    def _ssh_exec_command(self, cmd: str) -> None:
        """Run a command via SSH and stream output to log."""
        try:
            rc = self._orchestrator.ssh.exec_command(cmd, log_cb=self._append_log)
            if rc == 0 and self._looks_like_training_command(cmd):
                self._append_log("Training command finished — downloading /workspace/output automatically…")
                self._orchestrator.config = self._build_config()
                self._orchestrator.download_results()
        except Exception as exc:
            self._append_log(f"⚠  SSH error: {exc}")

    @staticmethod
    def _looks_like_training_command(cmd: str) -> bool:
        lowered = cmd.lower()
        return "train.py" in lowered and "python" in lowered

    def _on_ssh_exit(self) -> None:
        """Close the SSH console bar."""
        self._ssh_console_active = False
        self.ssh_input_frame.grid_remove()
        self._append_log("═══ SSH Console closed. ═══")


def run_app() -> None:
    app = App()
    app.mainloop()
