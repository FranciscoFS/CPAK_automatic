import cv2
import os
from pathlib import Path
import numpy as np

# Configuración de Rutas
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
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
display_scale = 1.0

# Variables para el Modo Círculo
drawing_mode = "point" # "point" o "circle"
circle_state = "none" # "none", "drawing", "placed", "moving"
temp_circle_center = None
temp_circle_radius = 0
drag_offset = (0,0)

def dummy_callback(val):
    pass

def click_event(event, x, y, flags, params):
    global current_points, display_scale, drawing_mode, circle_state, temp_circle_center, temp_circle_radius, drag_offset
    
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
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((orig_x, orig_y))

def run_tagger():
    global current_points, display_scale, drawing_mode, circle_state, temp_circle_center, temp_circle_radius
    
    all_crops = [f for f in os.listdir(CROP_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not all_crops: 
        print(f"No hay imágenes en {CROP_DIR}")
        return

    print("--- Tagueador CPAK Pro v5 (Círculo Interactivo) ---")

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
            
        lbl_path = CROP_DIR / f"{Path(img_name).stem}.txt"
        
        if lbl_path.exists(): continue

        config = JOINT_CONFIG[joint_type]
        img_path = CROP_DIR / img_name
        img_orig = cv2.imread(str(img_path))
        if img_orig is None: continue
        
        h_orig, w_orig, _ = img_orig.shape
        target_h = 650
        display_scale = target_h / h_orig
        img_resized_base = cv2.resize(img_orig, (int(w_orig * display_scale), target_h))
            
        current_points = []
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
            
            for i, (orig_x, orig_y) in enumerate(current_points):
                disp_x, disp_y = int(orig_x * display_scale), int(orig_y * display_scale)
                color = (0, 255, 0)
                cv2.circle(info_img, (disp_x, disp_y), 4, color, -1)
                cv2.putText(info_img, str(i+1), (disp_x + 5, disp_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
            cv2.putText(info_img, f"[s] Guardar/Saltar  [r] Reset  [q] Salir  [c] Modo: {mode_text}", 
                        (10, h_disp - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mode, 1)

            cv2.imshow("Tagueador CPAK", info_img)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): 
                cv2.destroyAllWindows(); return
            elif key == ord('c'):
                drawing_mode = "circle" if drawing_mode == "point" else "point"
            elif key == ord('r'):
                current_points = []; circle_state = "none"
                if joint_type == "Cadera": drawing_mode = "circle"
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
