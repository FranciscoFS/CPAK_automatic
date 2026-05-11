import importlib.util
import os
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = BASE_DIR / "scripts" / "13_final_inference.py"


def load_pipeline_module():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"No se encontro el pipeline: {PIPELINE_PATH}")

    spec = importlib.util.spec_from_file_location("cpak_final_inference_pipeline", str(PIPELINE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el modulo de pipeline final")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CPAK - Inferencia Final GUI")
        self.root.geometry("1280x860")

        self.pipeline = load_pipeline_module()

        self.source_var = tk.StringVar(value="")
        self.det_model_var = tk.StringVar(value=str(self.pipeline.DEFAULT_DET_MODEL))
        self.pose_model_var = tk.StringVar(value=str(self.pipeline.DEFAULT_POSE_MODEL))
        self.output_dir_var = tk.StringVar(value=str(self.pipeline.DEFAULT_OUTPUT_DIR))
        self.det_conf_var = tk.StringVar(value="0.20")
        self.pose_conf_var = tk.StringVar(value="0.20")
        self.padding_var = tk.StringVar(value="0.15")
        self.show_boxes_var = tk.BooleanVar(value=True)
        self.show_axes_var = tk.BooleanVar(value=True)

        self.det_model = None
        self.pose_model = None
        self.loaded_det_path = None
        self.loaded_pose_path = None

        self.preview_image = None
        self.latest_overlay_path = None
        self.latest_json_path = None
        self.latest_payload = None
        self.latest_source_path = None
        self.der_metrics_var = tk.StringVar(value="Der: sin datos")
        self.izq_metrics_var = tk.StringVar(value="Izq: sin datos")

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        form = ttk.Frame(self.root, padding=12)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        self._row_file_picker(form, 0, "Imagen Rx", self.source_var, self.pick_source_file)
        self._row_file_picker(form, 1, "Modelo deteccion", self.det_model_var, self.pick_det_model)
        self._row_file_picker(form, 2, "Modelo pose", self.pose_model_var, self.pick_pose_model)
        self._row_folder_picker(form, 3, "Directorio salida", self.output_dir_var, self.pick_output_dir)

        ttk.Label(form, text="det_conf").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.det_conf_var, width=10).grid(row=4, column=1, sticky="w", pady=(8, 0))

        ttk.Label(form, text="pose_conf").grid(row=4, column=2, sticky="w", pady=(8, 0), padx=(12, 0))
        ttk.Entry(form, textvariable=self.pose_conf_var, width=10).grid(row=4, column=3, sticky="w", pady=(8, 0))

        ttk.Label(form, text="padding (% BB)").grid(row=4, column=4, sticky="w", pady=(8, 0), padx=(12, 0))
        ttk.Entry(form, textvariable=self.padding_var, width=10).grid(row=4, column=5, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(form)
        buttons.grid(row=5, column=0, columnspan=6, sticky="w", pady=(12, 0))
        ttk.Button(buttons, text="Ejecutar inferencia", command=self.on_run).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Abrir overlay", command=self.open_overlay).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Abrir JSON", command=self.open_json).grid(row=0, column=2)

        layers = ttk.Frame(form)
        layers.grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(layers, text="Visualizador:").grid(row=0, column=0, padx=(0, 8))
        ttk.Checkbutton(layers, text="Ver Boxes", variable=self.show_boxes_var, command=self.on_view_toggle).grid(row=0, column=1, padx=(0, 8))
        ttk.Checkbutton(layers, text="Ejes (proyectados)", variable=self.show_axes_var, command=self.on_view_toggle).grid(row=0, column=2)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=12)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Metrica y estado").grid(row=0, column=0, sticky="w")
        metrics_panel = ttk.Frame(left)
        metrics_panel.grid(row=1, column=0, sticky="ew", pady=(6, 2))
        ttk.Label(metrics_panel, textvariable=self.der_metrics_var).grid(row=0, column=0, sticky="w")
        ttk.Label(metrics_panel, textvariable=self.izq_metrics_var).grid(row=1, column=0, sticky="w")
        self.log_text = tk.Text(left, wrap="word", height=20)
        self.log_text.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

        right = ttk.Frame(body, padding=12)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Preview overlay").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(right, text="Sin resultado aun", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        body.add(left, weight=1)
        body.add(right, weight=2)

        self.log("GUI lista. Selecciona una Rx y ejecuta.")

    def _row_file_picker(self, parent, row, label, var, callback):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, columnspan=4, sticky="ew", pady=2)
        ttk.Button(parent, text="...", width=4, command=callback).grid(row=row, column=5, sticky="e", pady=2)

    def _row_folder_picker(self, parent, row, label, var, callback):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, columnspan=4, sticky="ew", pady=2)
        ttk.Button(parent, text="...", width=4, command=callback).grid(row=row, column=5, sticky="e", pady=2)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def pick_source_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen Rx",
            filetypes=[("Imagen", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")],
            initialdir=str(BASE_DIR / "data"),
        )
        if path:
            self.source_var.set(path)

    def pick_det_model(self):
        path = filedialog.askopenfilename(title="Seleccionar modelo deteccion", filetypes=[("Modelos PyTorch", "*.pt")])
        if path:
            self.det_model_var.set(path)

    def pick_pose_model(self):
        path = filedialog.askopenfilename(title="Seleccionar modelo pose", filetypes=[("Modelos PyTorch", "*.pt")])
        if path:
            self.pose_model_var.set(path)

    def pick_output_dir(self):
        path = filedialog.askdirectory(title="Seleccionar directorio de salida", initialdir=str(BASE_DIR / "visualizations"))
        if path:
            self.output_dir_var.set(path)

    def _load_models_if_needed(self, det_model_path, pose_model_path):
        if self.det_model is None or self.loaded_det_path != det_model_path:
            self.log(f"Cargando modelo deteccion: {det_model_path}")
            self.det_model = self.pipeline.YOLO(str(det_model_path))
            self.loaded_det_path = det_model_path

        if self.pose_model is None or self.loaded_pose_path != pose_model_path:
            self.log(f"Cargando modelo pose: {pose_model_path}")
            self.pose_model = self.pipeline.YOLO(str(pose_model_path))
            self.loaded_pose_path = pose_model_path

    def on_run(self):
        worker = threading.Thread(target=self._run_pipeline_safe, daemon=True)
        worker.start()

    def _run_pipeline_safe(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self.root.after(0, lambda: self.log(f"ERROR: {exc}"))

    def _run_pipeline(self):
        source = Path(self.source_var.get().strip())
        det_model_path = Path(self.det_model_var.get().strip())
        pose_model_path = Path(self.pose_model_var.get().strip())
        output_dir = Path(self.output_dir_var.get().strip())

        if not source.exists():
            raise FileNotFoundError(f"No existe la imagen: {source}")
        if not det_model_path.exists():
            raise FileNotFoundError(f"No existe el modelo deteccion: {det_model_path}")
        if not pose_model_path.exists():
            raise FileNotFoundError(f"No existe el modelo pose: {pose_model_path}")

        det_conf = float(self.det_conf_var.get().strip())
        pose_conf = float(self.pose_conf_var.get().strip())
        padding = float(self.padding_var.get().strip())

        self.root.after(0, lambda: self.log("Iniciando inferencia..."))
        self._load_models_if_needed(det_model_path, pose_model_path)

        overlay_path, json_path, payload = self.pipeline.process_image(
            det_model=self.det_model,
            pose_model=self.pose_model,
            image_path=source,
            output_dir=output_dir,
            det_conf=det_conf,
            pose_conf=pose_conf,
            padding=padding,
        )

        self.latest_overlay_path = Path(overlay_path)
        self.latest_json_path = Path(json_path)
        self.latest_payload = payload
        self.latest_source_path = source

        self.root.after(0, lambda: self._show_result(payload))

    def _show_result(self, payload):
        self.log("Inferencia completada")
        self.log(f"Overlay: {self.latest_overlay_path}")
        self.log(f"JSON: {self.latest_json_path}")

        for side_name in ["Der", "Izq"]:
            side_info = payload["sides"].get(side_name, {})
            metrics = side_info.get("metrics", {})
            status = metrics.get("status", "sin_datos")
            if status == "ok":
                self.log(
                    f"{side_name}: LDFA={metrics['LDFA']:.2f} MPTA={metrics['MPTA']:.2f} "
                    f"aHKA={metrics['aHKA']:.2f} JLO={metrics['JLO']:.2f} | {metrics['CPAK']}"
                )
            else:
                self.log(f"{side_name}: {metrics}")

        self._update_metric_labels(payload)
        self._update_preview_image()

    def _update_metric_labels(self, payload):
        for side_name, target_var in [("Der", self.der_metrics_var), ("Izq", self.izq_metrics_var)]:
            side_info = payload.get("sides", {}).get(side_name, {})
            metrics = side_info.get("metrics", {})
            if metrics.get("status") == "ok":
                target_var.set(
                    f"{side_name}: LDFA={metrics['LDFA']:.2f}  MPTA={metrics['MPTA']:.2f}  "
                    f"aHKA={metrics['aHKA']:.2f}  JLO={metrics['JLO']:.2f}  {metrics['CPAK']}"
                )
            else:
                target_var.set(f"{side_name}: {metrics}")

    def on_view_toggle(self):
        if self.latest_payload is None or self.latest_source_path is None:
            return
        self._update_preview_image()

    def _update_preview_image(self):
        if self.latest_payload is None or self.latest_source_path is None:
            self.preview_label.configure(text="No hay overlay disponible", image="")
            return

        try:
            img = self.pipeline.render_overlay_from_payload(
                image_path=self.latest_source_path,
                payload=self.latest_payload,
                draw_boxes=bool(self.show_boxes_var.get()),
                draw_axes=bool(self.show_axes_var.get()),
            )
        except Exception as exc:
            self.preview_label.configure(text=f"Error en preview: {exc}", image="")
            return

        if img is None:
            self.preview_label.configure(text="No se pudo cargar preview")
            return

        if self.latest_overlay_path is not None:
            self.pipeline.cv2.imwrite(str(self.latest_overlay_path), img, [self.pipeline.cv2.IMWRITE_JPEG_QUALITY, 92])

        max_w = 850
        max_h = 760
        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = self.pipeline.cv2.resize(img, (new_w, new_h), interpolation=self.pipeline.cv2.INTER_AREA)

        preview_path = Path(tempfile.gettempdir()) / "cpak_overlay_preview.png"
        self.pipeline.cv2.imwrite(str(preview_path), img)

        photo = tk.PhotoImage(file=str(preview_path))
        self.preview_label.configure(image=photo, text="")
        self.preview_image = photo

    def open_overlay(self):
        if self.latest_overlay_path and self.latest_overlay_path.exists():
            os.startfile(str(self.latest_overlay_path))
        else:
            messagebox.showinfo("Info", "Aun no hay overlay generado")

    def open_json(self):
        if self.latest_json_path and self.latest_json_path.exists():
            os.startfile(str(self.latest_json_path))
        else:
            messagebox.showinfo("Info", "Aun no hay JSON generado")


def main():
    root = tk.Tk()
    app = FinalInferenceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
