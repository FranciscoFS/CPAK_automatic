import argparse
import cv2
import os
from pathlib import Path
import numpy as np

# Configuración de Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
CROP_DIR = BASE_DIR / "crops_for_keypoints"
OUTPUT_DIR = CROP_DIR

# Definición del Esqueleto Único (8 Puntos - Ahora con Notch Femoral)
JOINT_CONFIG = {
    "Cadera": {
        "id": 0, # Hip_Crop
        "visible_indices": [0],
        "points": ["0: Centro Cabeza Femoral"]
    },
    "Rodilla": {
        "id": 1, # Knee_Crop
        "visible_indices": [1, 3, 4, 5, 6, 7],
        "points": [
            "1: Centro Espinas Tibiales",
            "3: Cóndilo Medial (Vértice)",
            "4: Cóndilo Lateral (Vértice)",
            "5: Plateau Medial (Centro)",
            "6: Plateau Lateral (Centro)",
            "7: Centro Notch Femoral (Apex)"
        ]
    },
    "Tobillo": {
        "id": 2, # Ankle_Crop
        "visible_indices": [2],
        "points": ["2: Centro Domo del Talo (Astrágalo)"]
    }
}

MAX_POINTS = 8 # p0 a p7
current_points = []
predicted_points = []   # pre-relleno desde YOLO (puede modificarse antes de guardar)
display_scale = 1.0

# Variables para el Modo Círculo
drawing_mode = "point" # "point" o "circle"
circle_state = "none" # "none", "drawing", "placed", "moving"
temp_circle_center = None
temp_circle_radius = 0
drag_offset = (0, 0)

# Drag de puntos verdes
dragging_idx = -1       # índice del punto que se está arrastrando (-1 = ninguno)
DRAG_RADIUS_PX = 18     # píxeles de pantalla para detectar "cerca del punto"

def dummy_callback(val):
    pass

def click_event(event, x, y, flags, params):
    global current_points, predicted_points, display_scale, drawing_mode
    global circle_state, temp_circle_center, temp_circle_radius, drag_offset
    global dragging_idx

    orig_x, orig_y = int(x / display_scale), int(y / display_scale)

    if drawing_mode == "circle":
        if event == cv2.EVENT_LBUTTONDOWN:
            if circle_state == "none":
                circle_state = "drawing"
                temp_circle_center = (orig_x, orig_y)
                temp_circle_radius = 0
            elif circle_state == "placed":
                if temp_circle_center is not None:
                    dx = orig_x - temp_circle_center[0]
                    dy = orig_y - temp_circle_center[1]
                    if (dx**2 + dy**2)**0.5 <= temp_circle_radius:
                        circle_state = "moving"
                        drag_offset = (dx, dy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if circle_state == "drawing" and temp_circle_center is not None:
                dx = orig_x - temp_circle_center[0]
                dy = orig_y - temp_circle_center[1]
                temp_circle_radius = int((dx**2 + dy**2)**0.5)
            elif circle_state == "moving" and temp_circle_center is not None:
                temp_circle_center = (orig_x - drag_offset[0], orig_y - drag_offset[1])
        elif event == cv2.EVENT_LBUTTONUP:
            if circle_state == "drawing" or circle_state == "moving":
                circle_state = "placed"
    else:
        # Modo punto: soporta arrastrar puntos existentes
        if event == cv2.EVENT_LBUTTONDOWN:
            snap_thresh = DRAG_RADIUS_PX / display_scale
            for i, (px, py) in enumerate(current_points):
                if ((orig_x - px)**2 + (orig_y - py)**2)**0.5 <= snap_thresh:
                    dragging_idx = i
                    return
            # Click en espacio vacío → agregar nuevo punto
            dragging_idx = -1
            current_points.append((orig_x, orig_y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx >= 0 and (flags & cv2.EVENT_FLAG_LBUTTON):
                current_points[dragging_idx] = (orig_x, orig_y)

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Click derecho: eliminar el punto más cercano
            snap_thresh = DRAG_RADIUS_PX / display_scale
            for i, (px, py) in enumerate(current_points):
                if ((orig_x - px)**2 + (orig_y - py)**2)**0.5 <= snap_thresh:
                    current_points.pop(i)
                    break

def run_tagger():
    global current_points, predicted_points, display_scale, drawing_mode, circle_state, temp_circle_center, temp_circle_radius

    parser = argparse.ArgumentParser(description="Tagueador CPAK con predicción YOLO")
    parser.add_argument("--model", type=str,
                        default=str(BASE_DIR / "training_runs/telerx_pose_s_rebuild1_ft_full_b16_flipfix/weights/best.pt"),
                        help="Ruta al modelo pose .pt para pre-rellenar con [p]")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Confianza mínima para la predicción YOLO")
    parser.add_argument("--crop-dir", type=str, default=str(CROP_DIR),
                        help="Directorio de crops a taguear")
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)

    # Cargar modelo YOLO si existe
    yolo_model = None
    model_path = Path(args.model)
    if model_path.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(model_path))
            print(f"Modelo YOLO cargado: {model_path}")
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo YOLO: {e}")
    else:
        print(f"[INFO] Modelo no encontrado ({model_path}). Predicción deshabilitada.")

    all_crops = [f for f in os.listdir(crop_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not all_crops:
        print(f"No hay imágenes en {crop_dir}")
        return

    print("--- Tagueador CPAK Pro v6 (Predicción YOLO) ---")
    print("  [p]  Predecir con YOLO y pre-rellenar puntos")
    print("  [s]  Guardar / Saltar   [r]  Reset   [q]  Salir")
    print("  [c]  Cambiar modo punto/círculo   [SPACE] Confirmar círculo")

    cv2.namedWindow("Tagueador CPAK")
    cv2.createTrackbar("Brillo", "Tagueador CPAK", 100, 200, dummy_callback)
    cv2.createTrackbar("Contraste", "Tagueador CPAK", 100, 300, dummy_callback)
    cv2.setMouseCallback("Tagueador CPAK", click_event)

    for img_name in all_crops:
        joint_type = None
        for key in JOINT_CONFIG.keys():
            if key in img_name:
                joint_type = key
                break
        
        if not joint_type: continue
            
        lbl_path = crop_dir / f"{Path(img_name).stem}.txt"
        
        if lbl_path.exists(): continue

        config = JOINT_CONFIG[joint_type]
        img_path = crop_dir / img_name
        img_orig = cv2.imread(str(img_path))
        if img_orig is None: continue
        
        h_orig, w_orig, _ = img_orig.shape
        target_h = 650
        display_scale = target_h / h_orig
        img_resized_base = cv2.resize(img_orig, (int(w_orig * display_scale), target_h))
            
        current_points = []
        predicted_points = []
        circle_state = "none"
        temp_circle_center = None
        temp_circle_radius = 0
        
        if joint_type == "Cadera": drawing_mode = "circle"
        else: drawing_mode = "point"
        
        while True:
            b_val = cv2.getTrackbarPos("Brillo", "Tagueador CPAK") - 100
            c_val = cv2.getTrackbarPos("Contraste", "Tagueador CPAK") / 100.0
            
            info_img = cv2.convertScaleAbs(img_resized_base, alpha=c_val, beta=b_val)
            h_disp, w_disp, _ = info_img.shape

            # Mostrar predicciones YOLO en naranja solo si el verde aún no llegó a ese punto
            for i, (orig_x, orig_y) in enumerate(predicted_points):
                if i < len(current_points):
                    continue  # ya hay punto verde confirmado en esta posición, no mostrar naranja
                disp_x, disp_y = int(orig_x * display_scale), int(orig_y * display_scale)
                cv2.circle(info_img, (disp_x, disp_y), 8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.circle(info_img, (disp_x, disp_y), 3, (0, 165, 255), -1, cv2.LINE_AA)
                idx_label = config["visible_indices"][i] if i < len(config["visible_indices"]) else i
                cv2.putText(info_img, f"p{idx_label}?", (disp_x + 6, disp_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            # Mostrar puntos confirmados (verde) con nombre del punto
            for i, (orig_x, orig_y) in enumerate(current_points):
                disp_x, disp_y = int(orig_x * display_scale), int(orig_y * display_scale)
                color = (0, 255, 0)
                cv2.circle(info_img, (disp_x, disp_y), 5, color, -1)
                # Etiqueta: nombre corto del punto (ej. "Cóndilo Medial")
                if i < len(config["points"]):
                    # tomar solo la parte después del "N: "
                    full_name = config["points"][i]
                    short = full_name.split(": ", 1)[-1] if ": " in full_name else full_name
                else:
                    short = str(i + 1)
                cv2.putText(info_img, short, (disp_x + 6, disp_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if drawing_mode == "circle" and circle_state != "none" and temp_circle_center is not None:
                disp_cx = int(temp_circle_center[0] * display_scale)
                disp_cy = int(temp_circle_center[1] * display_scale)
                disp_r = int(temp_circle_radius * display_scale)
                circle_color = (0, 165, 255) if circle_state == "moving" else (0, 255, 255)
                cv2.circle(info_img, (disp_cx, disp_cy), disp_r, circle_color, 2)
                cv2.circle(info_img, (disp_cx, disp_cy), 2, (0, 0, 255), -1)
                if circle_state == "placed":
                    cv2.putText(info_img, "Presiona ESPACIO para confirmar", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            p_idx = len(current_points)
            cv2.rectangle(info_img, (0, 0), (w_disp, 50), (0, 0, 0), -1)
            
            if p_idx < len(config["points"]):
                target_name = config["points"][p_idx]
                cv2.putText(info_img, f"{joint_type}: {target_name} ({p_idx+1}/{len(config['points'])})", 
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(info_img, "LISTO - Presiona 's' para Guardar", 
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(info_img, (0, h_disp - 40), (w_disp, h_disp), (0, 0, 0), -1)
            mode_text = "CIRCULO" if drawing_mode == "circle" else "PUNTO"
            color_mode = (0, 255, 255) if drawing_mode == "circle" else (255, 255, 255)
            yolo_hint = "  [p] YOLO" if yolo_model else ""
            yolo_hint = "  [p] YOLO" if yolo_model else ""
            cv2.putText(info_img,
                        f"[s] Guardar  [r] Reset  [q] Salir  [c] {mode_text}{yolo_hint}  | drag=mover  RClick=borrar",
                        (10, h_disp - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_mode, 1)

            cv2.imshow("Tagueador CPAK", info_img)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): 
                cv2.destroyAllWindows(); return
            elif key == ord('c'):
                drawing_mode = "circle" if drawing_mode == "point" else "point"
            elif key == ord('r'):
                current_points = []; predicted_points = []; circle_state = "none"
                if joint_type == "Cadera": drawing_mode = "circle"
            elif key == ord('p'):
                # Predicción YOLO
                if yolo_model is None:
                    print("[WARN] No hay modelo YOLO cargado.")
                else:
                    try:
                        result = yolo_model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]
                        if result.keypoints is not None and len(result.keypoints.xy) > 0:
                            pred_xy = result.keypoints.xy[0].cpu().numpy()  # (8, 2) en px orig
                            new_pts = []
                            for vis_idx in config["visible_indices"]:
                                px, py = float(pred_xy[vis_idx][0]), float(pred_xy[vis_idx][1])
                                # Si el punto es (0,0) el modelo no lo detectó
                                if px > 1 or py > 1:
                                    new_pts.append((px, py))
                                else:
                                    new_pts.append(None)

                            # Para Cadera usamos el centro del círculo (índice 0)
                            if joint_type == "Cadera":
                                px, py = float(pred_xy[0][0]), float(pred_xy[0][1])
                                if px > 1 or py > 1:
                                    temp_circle_center = (int(px), int(py))
                                    temp_circle_radius = 0
                                    circle_state = "placed"
                                    drawing_mode = "circle"
                                    print(f"  YOLO → centro cadera: ({px:.0f}, {py:.0f}). Ajusta el radio y presiona ESPACIO.")
                            else:
                                # Filtrar Nones para current_points
                                valid = [(p[0], p[1]) for p in new_pts if p is not None]
                                predicted_points = list(valid)   # naranja (referencia)
                                current_points = list(valid)     # verde (editable)
                                print(f"  YOLO → {len(valid)}/{len(config['visible_indices'])} puntos pre-rellenados.")
                        else:
                            print("  YOLO → sin detección.")
                    except Exception as e:
                        print(f"  Error al predecir: {e}")
            elif key == 32: # SPACE
                if drawing_mode == "circle" and circle_state == "placed":
                    current_points.append(temp_circle_center); circle_state = "none"; drawing_mode = "point"
            elif key == ord('s'):
                if len(current_points) == len(config["points"]):
                    # Mapeo a formato YOLO Pose 29 columnas (5 base + 8*3=24 keypoints)
                    global_points = {i: (0.0, 0.0, 0) for i in range(MAX_POINTS)}
                    for i, idx in enumerate(config["visible_indices"]):
                        px, py = current_points[i]
                        global_points[idx] = (px/w_orig, py/h_orig, 2)
                    
                    line = f"{config['id']} 0.5 0.5 0.9 0.9"
                    for i in range(MAX_POINTS):
                        x, y, v = global_points[i]
                        line += f" {x:.6f} {y:.6f} {v}"
                    
                    with open(lbl_path, 'w') as f: f.write(line + "\n")
                    print(f"Guardado: {img_name}")
                    break
                else:
                    print(f"Saltada: {img_name} (Incompleta)"); break

    cv2.destroyAllWindows()
    print("\nProceso terminado.")

if __name__ == "__main__":
    run_tagger()
