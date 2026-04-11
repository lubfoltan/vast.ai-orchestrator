"""SCOUT — Desktop GUI for Vast.ai Training Orchestrator."""

import csv
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from config import ExperimentConfig
from orchestrator import Orchestrator

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
        ctk.CTkEntry(f, textvariable=self.data_path_var).grid(row=row, column=1, columnspan=2, padx=5, pady=4, sticky="ew")
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_data).grid(row=row, column=3, padx=2)

        ctk.CTkLabel(f, text="Output Path:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.output_path_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.output_path_var).grid(row=row, column=5, padx=5, pady=4, sticky="ew")
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_output).grid(row=row, column=6, padx=2)

        row += 1
        ctk.CTkLabel(f, text="Test Data:").grid(row=row, column=0, padx=5, pady=4, sticky="e")
        self.test_data_path_var = ctk.StringVar()
        ctk.CTkEntry(f, textvariable=self.test_data_path_var, placeholder_text="(Optional — leave empty to split from Train)").grid(
            row=row, column=1, columnspan=2, padx=5, pady=4, sticky="ew"
        )
        ctk.CTkButton(f, text="Browse…", width=70, command=self._browse_test_data).grid(row=row, column=3, padx=2)

        ctk.CTkLabel(f, text="Train/Test Split:").grid(row=row, column=4, padx=5, pady=4, sticky="e")
        self.split_var = ctk.StringVar(value="0.8")
        ctk.CTkEntry(f, textvariable=self.split_var, width=60).grid(row=row, column=5, padx=5, pady=4, sticky="w")
        ctk.CTkLabel(f, text="(train ratio)").grid(row=row, column=6, padx=2, sticky="w")

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
        else:
            # Hide regression fields
            self.regression_frame.grid_remove()
            # Show classification widgets
            self.aug_label.grid()
            self.aug_frame.grid()
            self.grad_cam_cb.configure(state="normal")
            self.mixup_cb.configure(state="normal")
            self.label_smooth_cb.configure(state="normal")

    # ------------------------------------------------------------------
    def _build_buttons(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        self.btn_start = ctk.CTkButton(parent, text="▶  Start Pipeline", command=self._on_start, fg_color="green")
        self.btn_start.grid(row=0, column=0, padx=8, pady=8, sticky="ew")

        self.btn_cancel = ctk.CTkButton(parent, text="⏹  Stop Instance", command=self._on_cancel, state="disabled")
        self.btn_cancel.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        self.btn_destroy = ctk.CTkButton(
            parent, text="🗑  Terminate Instance", command=self._on_destroy, fg_color="red", state="disabled"
        )
        self.btn_destroy.grid(row=0, column=2, padx=8, pady=8, sticky="ew")

        self.btn_ssh_console = ctk.CTkButton(
            parent, text="💻  SSH Console", command=self._on_ssh_console, state="disabled"
        )
        self.btn_ssh_console.grid(row=0, column=3, padx=8, pady=8, sticky="ew")

        self.btn_chart = ctk.CTkButton(
            parent, text="📈  Live Chart", command=self._on_toggle_chart,
            state="normal" if _HAS_MATPLOTLIB else "disabled"
        )
        self.btn_chart.grid(row=0, column=4, padx=8, pady=8, sticky="ew")

        self.btn_clear = ctk.CTkButton(parent, text="Clear Log", command=self._clear_log)
        self.btn_clear.grid(row=0, column=5, padx=8, pady=8, sticky="ew")

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

    def _build_config(self) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.task_type = self.task_type_var.get().lower()
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
        # Regression fields
        cfg.target_column = self.target_col_var.get().strip()
        cfg.feature_columns = self.feature_cols_var.get().strip()
        # Augmentation
        cfg.random_rotation = self.aug_rotation_var.get()
        cfg.horizontal_flip = self.aug_hflip_var.get()
        cfg.random_erasing = self.aug_erasing_var.get()
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
        return cfg

    # ==================================================================
    # Button callbacks
    # ==================================================================
    def _validate_inputs(self) -> Optional[str]:
        if not self.data_path_var.get().strip():
            return "Train Data path is required."
        if not self.output_path_var.get().strip():
            return "Output Path is required."
        if not self.api_key_var.get().strip():
            return "Vast.ai API Key is required."
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

        self._worker_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._worker_thread.start()

    def _run_pipeline(self) -> None:
        try:
            self._orchestrator.run()
        finally:
            self.after(0, self._pipeline_finished)

    def _pipeline_finished(self) -> None:
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        if self._orchestrator and self._orchestrator.instance_id:
            self.btn_destroy.configure(state="normal")
            # Enable SSH console if we have a live SSH connection
            if self._orchestrator.ssh and self._orchestrator.ssh.is_connected:
                self.btn_ssh_console.configure(state="normal")

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

        self._ax_loss.clear()
        self._ax_loss.plot(epochs, train_loss, "c-", linewidth=1.2, label="Train")
        self._ax_loss.plot(epochs, val_loss, "r-", linewidth=1.2, label="Val")
        self._ax_loss.set_title("Loss", color="white", fontsize=9)
        self._ax_loss.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        self._ax_loss.tick_params(colors="white", labelsize=7)

        self._ax_acc.clear()
        # Plot accuracy if available, otherwise first non-loss metric
        if "val_acc" in rows[0]:
            train_acc = [float(r.get("train_acc", 0)) for r in rows]
            val_acc = [float(r.get("val_acc", 0)) for r in rows]
            self._ax_acc.plot(epochs, train_acc, "c-", linewidth=1.2, label="Train Acc")
            self._ax_acc.plot(epochs, val_acc, "r-", linewidth=1.2, label="Val Acc")
            self._ax_acc.set_title("Accuracy", color="white", fontsize=9)
        else:
            # Plot whatever metric columns exist (skip epoch, timestamp, losses)
            skip = {"epoch", "timestamp", "train_loss", "val_loss"}
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
            self._orchestrator.ssh.exec_command(cmd, log_cb=self._append_log)
        except Exception as exc:
            self._append_log(f"⚠  SSH error: {exc}")

    def _on_ssh_exit(self) -> None:
        """Close the SSH console bar."""
        self._ssh_console_active = False
        self.ssh_input_frame.grid_remove()
        self._append_log("═══ SSH Console closed. ═══")


def run_app() -> None:
    app = App()
    app.mainloop()
