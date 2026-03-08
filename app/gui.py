from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import diffusers
import torch

from app.config import AppConfig, INFERENCE_PRESETS, get_preset
from app.docx_reader import read_docx_text
from app.image_generator import ImageGenerator
from app.logger import attach_gui_handler
from app.manifest import RunManifestWriter
from app.prompt_intelligence import resolve_prompt_plan
from app.runtime_state import RuntimeResolver, build_export_payload
from app.repo_updater import pull_current_branch
from app.scanner import scan_docx_files
from app.types import DocumentManifest
from app.utils import open_path

VISUAL_STRATEGY_OPTIONS = [
    "auto",
    "editorial_photo",
    "conceptual",
    "infographic_like",
    "industrial",
    "institutional",
    "documentary_wide",
]


class AppGUI(ctk.CTk):
    def __init__(self, config: AppConfig, logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.title("Local Note Illustrator")
        self.geometry("1120x760")

        self.selected_root = Path.cwd()
        self.docx_jobs: list[Path] = []
        self.is_running = False
        self._queue: queue.Queue[str] = queue.Queue()

        self.gui_log_handler = attach_gui_handler(self.logger, self._enqueue_log)
        self.image_generator = ImageGenerator(config, logger)
        self.runtime_resolver = RuntimeResolver(config)
        self.manifest_writer = RunManifestWriter(config.log_dir)
        self.last_run_runtime = None
        self.last_document_exports: list[dict[str, object]] = []
        self.last_document_runtime_status = "doc_runtime=pendiente"

        self._build_ui()
        self.after(100, self._flush_logs)
        self._set_status("Listo")
        self._set_root_label(self.selected_root)
        self._refresh_runtime_panel()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        title = ctk.CTkLabel(self, text="Local Note Illustrator", font=("Segoe UI", 24, "bold"))
        title.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        top_frame.grid_columnconfigure(0, weight=1)

        self.root_label = ctk.CTkLabel(top_frame, text="Carpeta raíz: -")
        self.root_label.grid(row=0, column=0, padx=10, pady=8, sticky="w")

        self.runtime_label = ctk.CTkLabel(top_frame, text="Runtime efectivo: -", justify="left")
        self.runtime_label.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")

        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        for i in range(8):
            controls.grid_columnconfigure(i, weight=1)

        self.btn_update = ctk.CTkButton(controls, text="Actualizar repo", command=self.update_repo)
        self.btn_select = ctk.CTkButton(
            controls, text="Seleccionar carpeta raíz", command=self.select_root_folder
        )
        self.btn_scan = ctk.CTkButton(controls, text="Escanear documentos", command=self.scan_documents)
        self.btn_generate = ctk.CTkButton(controls, text="Generar imágenes", command=self.generate_images)
        self.btn_export_params = ctk.CTkButton(
            controls,
            text="Exportar parámetros",
            command=self.export_parameters,
        )
        self.btn_open_folder = ctk.CTkButton(
            controls, text="Abrir carpeta", command=self.open_selected_folder
        )
        self.btn_open_outputs = ctk.CTkButton(controls, text="Abrir outputs", command=self.open_outputs_folder)
        self.btn_open_logs = ctk.CTkButton(controls, text="Abrir logs", command=self.open_logs_folder)

        self.btn_update.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.btn_select.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.btn_scan.grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        self.btn_generate.grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        self.btn_export_params.grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        self.btn_open_folder.grid(row=0, column=5, padx=6, pady=6, sticky="ew")
        self.btn_open_outputs.grid(row=0, column=6, padx=6, pady=6, sticky="ew")
        self.btn_open_logs.grid(row=0, column=7, padx=6, pady=6, sticky="ew")

        self.num_images_var = ctk.IntVar(value=self.config.default_num_images)
        self.include_subfolders_var = ctk.BooleanVar(value=True)
        self.preset_var = ctk.StringVar(value=self.config.default_preset)
        self.strategy_var = ctk.StringVar(value="auto")

        ctk.CTkLabel(controls, text="Imágenes por documento:").grid(
            row=1, column=0, padx=6, pady=6, sticky="e"
        )
        self.option_images = ctk.CTkOptionMenu(
            controls,
            values=["1", "2"],
            command=self._on_images_per_doc_changed,
        )
        self.option_images.set(str(self.config.default_num_images))
        self.option_images.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        self.chk_subfolders = ctk.CTkCheckBox(
            controls,
            text="Incluir subcarpetas",
            variable=self.include_subfolders_var,
            onvalue=True,
            offvalue=False,
        )
        self.chk_subfolders.grid(row=1, column=2, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(controls, text="Preset:").grid(row=1, column=3, padx=6, pady=6, sticky="e")
        self.option_preset = ctk.CTkOptionMenu(
            controls,
            values=list(INFERENCE_PRESETS.keys()),
            command=lambda value: self.preset_var.set(value),
        )
        self.option_preset.set(self.config.default_preset)
        self.option_preset.grid(row=1, column=4, padx=6, pady=6, sticky="w")
        self.option_preset.configure(command=self._on_preset_changed)

        ctk.CTkLabel(controls, text="Estrategia visual:").grid(row=1, column=5, padx=6, pady=6, sticky="e")
        self.option_strategy = ctk.CTkOptionMenu(
            controls,
            values=VISUAL_STRATEGY_OPTIONS,
            command=lambda value: self.strategy_var.set(value),
        )
        self.option_strategy.set("auto")
        self.option_strategy.grid(row=1, column=6, padx=6, pady=6, sticky="w")
        self.option_strategy.configure(command=self._on_strategy_changed)

        ctk.CTkLabel(controls, text="Seed (vacío=aleatorio):").grid(
            row=2, column=0, padx=6, pady=6, sticky="e"
        )
        self.seed_entry = ctk.CTkEntry(controls, placeholder_text="Ej: 12345")
        self.seed_entry.grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        self.seed_entry.bind("<KeyRelease>", lambda _event: self._refresh_runtime_panel())

        self.progress = ctk.CTkProgressBar(self)
        self.progress.set(0)
        self.progress.grid(row=4, column=0, padx=12, pady=(6, 4), sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="Estado: -")
        self.status_label.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="w")

        self.log_box = ctk.CTkTextbox(self, wrap="word")
        self.log_box.grid(row=3, column=0, padx=12, pady=6, sticky="nsew")
        self.log_box.configure(state="disabled")

    def _parse_seed(self) -> int | None:
        raw = self.seed_entry.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            raise ValueError("Seed inválida. Debe ser un número entero o vacío.")

    def _build_runtime_snapshot(self):
        return self.runtime_resolver.resolve_run_runtime(
            preset_name=self.preset_var.get(),
            strategy_override=self.strategy_var.get(),
            seed=self._parse_seed(),
            images_per_document=self.num_images_var.get(),
            backend_state=self.image_generator.get_backend_state(),
        )

    def _on_preset_changed(self, value: str) -> None:
        self.preset_var.set(value)
        self._refresh_runtime_panel()

    def _on_images_per_doc_changed(self, value: str) -> None:
        self.num_images_var.set(int(value))
        self._refresh_runtime_panel()

    def _on_strategy_changed(self, value: str) -> None:
        self.strategy_var.set(value)
        self._refresh_runtime_panel()

    def _runtime_summary(self, runtime) -> str:
        return (
            f"Runtime efectivo | model={runtime.model_id.value} | pipeline={runtime.pipeline_class.value} | "
            f"device={runtime.device.value}/{runtime.dtype.value} | preset={runtime.preset.value} -> "
            f"{runtime.width.value}x{runtime.height.value} steps={runtime.steps.value} guidance={runtime.guidance_scale.value} | "
            f"seed={runtime.seed.value} | strategy={runtime.strategy_override.value} | images/doc={runtime.images_per_document.value} | "
            f"output={runtime.output_format.value} (jpg q={runtime.jpeg_quality.value} subsampling={runtime.jpeg_subsampling.value}) | "
            f"openai={runtime.openai_enable.value} mode={runtime.openai_mode.value} | cuda_fallback={runtime.cuda_fallback_triggered.value}"
            f" | {self.last_document_runtime_status}"
        )

    def _refresh_runtime_panel(self) -> None:
        try:
            runtime = self._build_runtime_snapshot()
        except ValueError:
            return
        self.last_run_runtime = runtime
        self.runtime_label.configure(text=self._runtime_summary(runtime))

    def _enqueue_log(self, msg: str) -> None:
        self._queue.put(msg)

    def _flush_logs(self) -> None:
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"{msg}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(100, self._flush_logs)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=f"Estado: {text}")

    def _set_root_label(self, path: Path) -> None:
        self.root_label.configure(text=f"Carpeta raíz: {path}")

    def _set_running(self, running: bool) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.btn_update.configure(state=state)
        self.btn_select.configure(state=state)
        self.btn_scan.configure(state=state)
        self.btn_generate.configure(state=state)

    def _run_background(self, target) -> None:
        if self.is_running:
            messagebox.showwarning("Proceso en ejecución", "Esperá a que termine la tarea actual.")
            return

        self._set_running(True)

        def worker():
            try:
                target()
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Error inesperado: %s", exc)
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def select_root_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.selected_root))
        if not selected:
            return
        self.selected_root = Path(selected)
        self._set_root_label(self.selected_root)
        self.logger.info("Carpeta seleccionada: %s", self.selected_root)

    def update_repo(self) -> None:
        def job():
            self.after(0, lambda: self._set_status("Actualizando repo..."))
            msg = pull_current_branch(Path.cwd())
            self.logger.info(msg)
            self.after(0, lambda: self._set_status("Repo actualizado"))

        self._run_background(job)

    def scan_documents(self) -> None:
        def job():
            self.after(0, lambda: self._set_status("Escaneando documentos..."))
            self.after(0, lambda: self.progress.set(0.1))
            docs = scan_docx_files(
                self.selected_root,
                include_subfolders=self.include_subfolders_var.get(),
            )
            self.docx_jobs = docs
            self.logger.info("Documentos encontrados: %s", len(docs))
            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self._set_status(f"Escaneo listo ({len(docs)} docx)"))

        self._run_background(job)

    def generate_images(self) -> None:
        def job():
            run_final_status = "failed"
            manifest_started = False
            manifest_closed = False
            try:
                selected_preset = self.preset_var.get()
                selected_strategy = self.strategy_var.get()
                preset = get_preset(selected_preset)
                seed = self._parse_seed()
                run_runtime = self.runtime_resolver.resolve_run_runtime(
                    preset_name=selected_preset,
                    strategy_override=selected_strategy,
                    seed=seed,
                    images_per_document=self.num_images_var.get(),
                    backend_state=self.image_generator.get_backend_state(),
                )
                self.last_run_runtime = run_runtime
                self.last_document_exports = []
                self.last_document_runtime_status = "doc_runtime=pendiente"
                self.after(0, self._refresh_runtime_panel)

                config_snapshot = {
                    "model_id": self.config.model_id,
                    "output_format": self.config.output_format,
                    "default_negative_prompt": self.config.default_negative_prompt,
                    "default_preset": self.config.default_preset,
                    "openai_enable": self.config.openai_enable,
                    "openai_prompt_intelligence_mode": self.config.openai_prompt_intelligence_mode,
                    "openai_model": self.config.openai_model,
                    "openai_timeout_seconds": self.config.openai_timeout_seconds,
                    "openai_max_retries": self.config.openai_max_retries,
                    "openai_strict_schema": self.config.openai_strict_schema,
                }
                self.manifest_writer.start(
                    selected_root_folder=self.selected_root,
                    include_subfolders=self.include_subfolders_var.get(),
                    images_per_document=self.num_images_var.get(),
                    config_snapshot=config_snapshot,
                    runtime_effective=run_runtime.to_dict(),
                )
                manifest_started = True

                if not self.docx_jobs:
                    self.logger.info("No hay documentos cargados, se ejecuta escaneo previo.")
                    self.docx_jobs = scan_docx_files(
                        self.selected_root,
                        include_subfolders=self.include_subfolders_var.get(),
                    )

                total = len(self.docx_jobs)
                if total == 0:
                    self.after(0, lambda: self.progress.set(0))
                    self.after(0, lambda: self._set_status("No se encontraron .docx"))
                    run_manifest_path = self.manifest_writer.finish(status="empty")
                    manifest_closed = True
                    self.logger.info("Manifest de corrida generado: %s", run_manifest_path)
                    return

                self.after(0, lambda: self._set_status("Generando imágenes..."))
                self.after(0, lambda: self.progress.set(0))

                done = 0
                succeeded_docs = 0
                failed_docs: list[Path] = []

                self.logger.info(
                    "%s | defaults individuales DEFAULT_* quedan eclipsados por preset en GUI",
                    self._runtime_summary(run_runtime),
                )

                for docx_path in self.docx_jobs:
                    document_manifest = DocumentManifest(
                        document_path=str(docx_path),
                        source="local_fallback",
                        strategy_override=selected_strategy,
                        strategy_suggested="pending",
                        strategy_effective="pending",
                        domain="pending",
                        preset=selected_preset,
                        seed=seed,
                        width=preset.width,
                        height=preset.height,
                        steps=preset.steps,
                        guidance_scale=preset.guidance_scale,
                        openai_status="pending",
                        prompt_source="local_prompt_builder",
                        strategy_adjustment_reason=None,
                        semantic_adjustment_reason=None,
                        semantic_validation_status=None,
                        openai_raw_payload=None,
                        validated_prompt_main=None,
                        final_positive_prompt=None,
                        final_negative_prompt=None,
                        sanitation_flags=None,
                        runtime_effective=run_runtime.to_dict(),
                    )
                    self.manifest_writer.add_document(document_manifest)
                    try:
                        self.after(0, lambda p=docx_path.name: self._set_status(f"Procesando: {p}"))
                        text = read_docx_text(docx_path)
                        resolution = resolve_prompt_plan(
                            text=text,
                            strategy_override=selected_strategy,
                            config=self.config,
                            logger=self.logger,
                            variants=self.num_images_var.get(),
                        )
                        plan = resolution.prompt_plan
                        intelligence = resolution.intelligence
                        self.logger.info(
                            "Documento=%s | source=%s | openai_status=%s | domain=%s | strategy_effective=%s | fallback_reason=%s | strategy_adjustment=%s",
                            docx_path.name,
                            plan.source,
                            resolution.openai_status,
                            plan.domain,
                            plan.strategy_effective,
                            intelligence.fallback_reason or "none",
                            intelligence.strategy_adjustment_reason or "none",
                        )
                        document_manifest.source = plan.source
                        document_manifest.openai_status = resolution.openai_status
                        document_manifest.fallback_reason = intelligence.fallback_reason
                        document_manifest.openai_model = self.config.openai_model
                        document_manifest.prompt_source = plan.source
                        document_manifest.strategy_suggested = intelligence.visual_strategy
                        document_manifest.strategy_effective = plan.strategy_effective
                        document_manifest.domain = plan.domain
                        document_manifest.strategy_adjustment_reason = plan.strategy_adjustment_reason
                        document_manifest.semantic_adjustment_reason = plan.semantic_adjustment_reason
                        document_manifest.semantic_validation_status = plan.semantic_validation_status
                        document_manifest.openai_raw_payload = intelligence.openai_raw_payload
                        document_manifest.validated_prompt_main = intelligence.prompt_main
                        document_manifest.final_positive_prompt = " || ".join(plan.positive_prompts)
                        document_manifest.final_negative_prompt = plan.negative_prompt
                        document_manifest.sanitation_flags = plan.sanitation_flags
                        doc_runtime = self.runtime_resolver.resolve_document_runtime(
                            run_runtime=run_runtime,
                            strategy_effective=plan.strategy_effective,
                            prompt_source=plan.source,
                            openai_status=resolution.openai_status,
                            fallback_reason=intelligence.fallback_reason,
                        )
                        document_manifest.runtime_effective = doc_runtime.to_dict()
                        self.last_document_exports.append(
                            {
                                "document_path": str(docx_path),
                                "prompt_source": plan.source,
                                "openai_status": resolution.openai_status,
                                "strategy_effective": plan.strategy_effective,
                                "domain": plan.domain,
                                "openai_raw_payload": intelligence.openai_raw_payload,
                                "validated_prompt_main": intelligence.prompt_main,
                                "final_positive_prompt": " || ".join(plan.positive_prompts),
                                "final_negative_prompt": plan.negative_prompt,
                                "sanitation_flags": plan.sanitation_flags,
                                "width": preset.width,
                                "height": preset.height,
                                "steps": preset.steps,
                                "guidance_scale": preset.guidance_scale,
                                "outputs": [],
                                "runtime_effective": doc_runtime.to_dict(),
                            }
                        )
                        doc_export_ref = self.last_document_exports[-1]
                        self.last_document_runtime_status = (
                            f"doc={docx_path.name} source={plan.source} openai_status={resolution.openai_status} "
                            f"strategy_effective={plan.strategy_effective}"
                        )
                        self.after(0, self._refresh_runtime_panel)
                        for idx, prompt in enumerate(plan.positive_prompts, start=1):
                            output_path = self.image_generator.generate(
                                docx_path=docx_path,
                                positive_prompt=prompt,
                                negative_prompt=plan.negative_prompt,
                                image_index=idx,
                                steps=preset.steps,
                                guidance_scale=preset.guidance_scale,
                                width=preset.width,
                                height=preset.height,
                                seed=seed,
                                preset_name=selected_preset,
                                strategy_name=plan.strategy_effective,
                            )
                            backend_state = self.image_generator.get_backend_state()
                            run_runtime = self.runtime_resolver.resolve_run_runtime(
                                preset_name=selected_preset,
                                strategy_override=selected_strategy,
                                seed=seed,
                                images_per_document=self.num_images_var.get(),
                                backend_state=backend_state,
                            )
                            self.last_run_runtime = run_runtime
                            self.after(0, self._refresh_runtime_panel)
                            generation_meta = self.image_generator.last_generation_metadata
                            self.manifest_writer.append_output(
                                document_manifest,
                                idx,
                                output_path,
                                file_size_bytes=int(generation_meta.get("file_size_bytes", 0)),
                                device_at_generation=str(generation_meta.get("device", backend_state.get("device", "cpu"))),
                                dtype_at_generation=str(generation_meta.get("dtype", backend_state.get("dtype", "float32"))),
                                cuda_fallback_triggered=bool(
                                    generation_meta.get(
                                        "cuda_fallback_triggered",
                                        backend_state.get("cuda_fallback_triggered", False),
                                    )
                                ),
                            )
                            self.logger.info(
                                "Output idx=%s | doc=%s | prompt_source=%s | strategy=%s | size_bytes=%s",
                                idx,
                                docx_path.name,
                                plan.source,
                                plan.strategy_effective,
                                generation_meta.get("file_size_bytes", "n/a"),
                            )
                            doc_export_ref["outputs"].append(
                                {
                                    "image_index": idx,
                                    "output_path": str(output_path),
                                    "file_size_bytes": int(generation_meta.get("file_size_bytes", 0)),
                                }
                            )
                        succeeded_docs += 1
                        self.logger.info("Documento procesado: %s", docx_path)
                    except Exception as exc:  # noqa: BLE001
                        failed_docs.append(docx_path)
                        document_manifest.openai_status = "error"
                        self.manifest_writer.mark_document_error(document_manifest, str(exc))
                        self.logger.exception("Falló el documento %s: %s", docx_path, exc)
                    finally:
                        done += 1
                        progress = done / total
                        self.after(0, lambda p=progress: self.progress.set(p))

                failed_count = len(failed_docs)
                if failed_count == 0:
                    final_status = "Generación completada"
                    run_final_status = "success"
                    self.logger.info(
                        "Resultado final: éxito total (%s/%s documentos).", succeeded_docs, total
                    )
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Generación completada",
                            f"Se procesaron correctamente {succeeded_docs} de {total} documentos.",
                        ),
                    )
                elif succeeded_docs > 0:
                    final_status = "Generación completada con errores"
                    run_final_status = "partial_success"
                    self.logger.warning(
                        "Resultado final: éxito parcial (%s ok, %s con error).",
                        succeeded_docs,
                        failed_count,
                    )
                    self.after(
                        0,
                        lambda: messagebox.showwarning(
                            "Generación completada con errores",
                            f"Se procesaron {succeeded_docs} de {total} documentos. "
                            f"Fallaron {failed_count}. Revisá logs.",
                        ),
                    )
                else:
                    final_status = "Generación fallida"
                    run_final_status = "failed"
                    self.logger.error(
                        "Resultado final: fallo total (%s/%s con error).", failed_count, total
                    )
                    self.after(
                        0,
                        lambda: messagebox.showerror(
                            "Generación fallida",
                            "No se pudo generar imágenes para ningún documento. Revisá logs.",
                        ),
                    )

                self.after(0, lambda s=final_status: self._set_status(s))
                run_manifest_path = self.manifest_writer.finish(status=run_final_status)
                manifest_closed = True
                self.logger.info("Manifest de corrida generado: %s", run_manifest_path)
            finally:
                if manifest_started and not manifest_closed:
                    run_manifest_path = self.manifest_writer.finish(status=run_final_status)
                    self.logger.info("Manifest de corrida generado (cierre de seguridad): %s", run_manifest_path)

        self._run_background(job)

    def export_parameters(self) -> None:
        timestamp = datetime.now()
        runtime = self._build_runtime_snapshot()
        data = build_export_payload(
            runtime=runtime,
            selected_root_folder=str(self.selected_root),
            per_document=self.last_document_exports,
            torch_version=torch.__version__,
            diffusers_version=diffusers.__version__,
            generated_at=timestamp.isoformat(timespec="seconds"),
        )

        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.config.log_dir / f"run_config_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"

        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        self.logger.info("Parámetros exportados: %s", file_path)
        self._set_status("Parámetros exportados")
        messagebox.showinfo("Exportar parámetros", f"Archivo guardado en:\n{file_path}")

    def open_selected_folder(self) -> None:
        try:
            open_path(self.selected_root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    def open_outputs_folder(self) -> None:
        self.open_selected_folder()

    def open_logs_folder(self) -> None:
        try:
            open_path(self.config.log_dir)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))


def launch_gui(config: AppConfig, logger) -> None:
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = AppGUI(config, logger)
    app.mainloop()
