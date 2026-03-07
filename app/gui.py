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

from app.config import AppConfig
from app.docx_reader import read_docx_text
from app.image_generator import ImageGenerator
from app.logger import attach_gui_handler
from app.prompt_builder import build_prompts
from app.repo_updater import pull_current_branch
from app.scanner import scan_docx_files
from app.utils import open_path


class AppGUI(ctk.CTk):
    def __init__(self, config: AppConfig, logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.title("Local Note Illustrator")
        self.geometry("980x720")

        self.selected_root = Path.cwd()
        self.docx_jobs: list[Path] = []
        self.is_running = False
        self._queue: queue.Queue[str] = queue.Queue()

        self.gui_log_handler = attach_gui_handler(self.logger, self._enqueue_log)
        self.image_generator = ImageGenerator(config, logger)

        self._build_ui()
        self.after(100, self._flush_logs)
        self._set_status("Listo")
        self._set_root_label(self.selected_root)

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

        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        for i in range(7):
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
        self.btn_open_logs = ctk.CTkButton(controls, text="Abrir logs", command=self.open_logs_folder)

        self.btn_update.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.btn_select.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.btn_scan.grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        self.btn_generate.grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        self.btn_export_params.grid(row=0, column=4, padx=6, pady=6, sticky="ew")
        self.btn_open_folder.grid(row=0, column=5, padx=6, pady=6, sticky="ew")
        self.btn_open_logs.grid(row=0, column=6, padx=6, pady=6, sticky="ew")

        self.num_images_var = ctk.IntVar(value=self.config.default_num_images)
        self.include_subfolders_var = ctk.BooleanVar(value=True)

        ctk.CTkLabel(controls, text="Imágenes por documento:").grid(
            row=1, column=0, padx=6, pady=6, sticky="e"
        )
        self.option_images = ctk.CTkOptionMenu(
            controls,
            values=["1", "2"],
            command=lambda value: self.num_images_var.set(int(value)),
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

        self.progress = ctk.CTkProgressBar(self)
        self.progress.set(0)
        self.progress.grid(row=4, column=0, padx=12, pady=(6, 4), sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="Estado: -")
        self.status_label.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="w")

        self.log_box = ctk.CTkTextbox(self, wrap="word")
        self.log_box.grid(row=3, column=0, padx=12, pady=6, sticky="nsew")
        self.log_box.configure(state="disabled")

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
                return

            self.after(0, lambda: self._set_status("Generando imágenes..."))
            self.after(0, lambda: self.progress.set(0))

            done = 0
            succeeded_docs = 0
            failed_docs: list[Path] = []

            for docx_path in self.docx_jobs:
                try:
                    text = read_docx_text(docx_path)
                    prompts = build_prompts(
                        text=text,
                        negative_prompt=self.config.default_negative_prompt,
                        variants=self.num_images_var.get(),
                    )
                    for idx, prompt in enumerate(prompts.positive_prompts, start=1):
                        self.image_generator.generate(
                            docx_path=docx_path,
                            positive_prompt=prompt,
                            negative_prompt=prompts.negative_prompt,
                            image_index=idx,
                        )
                    succeeded_docs += 1
                    self.logger.info("Documento procesado: %s", docx_path)
                except Exception as exc:  # noqa: BLE001
                    failed_docs.append(docx_path)
                    self.logger.exception("Falló el documento %s: %s", docx_path, exc)
                finally:
                    done += 1
                    progress = done / total
                    self.after(0, lambda p=progress: self.progress.set(p))

            failed_count = len(failed_docs)
            if failed_count == 0:
                final_status = "Generación completada"
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
                self.logger.error("Resultado final: fallo total (%s/%s con error).", failed_count, total)
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Generación fallida",
                        "No se pudo generar imágenes para ningún documento. Revisá logs.",
                    ),
                )

            self.after(0, lambda s=final_status: self._set_status(s))

        self._run_background(job)

    def export_parameters(self) -> None:
        timestamp = datetime.now()
        runtime = self.image_generator.get_runtime_parameters()
        data = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "model_id": runtime["model_id"],
            "device": runtime["device"],
            "force_cpu": runtime["force_cpu"],
            "width": runtime["width"],
            "height": runtime["height"],
            "steps": runtime["steps"],
            "guidance_scale": runtime["guidance_scale"],
            "negative_prompt": runtime["negative_prompt"],
            "torch_version": torch.__version__,
            "diffusers_version": diffusers.__version__,
            "selected_root_folder": str(self.selected_root),
            "images_per_document": self.num_images_var.get(),
        }

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
