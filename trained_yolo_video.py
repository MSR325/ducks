import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('runs/detect/train6/weights/best.pt')

# Cargar matriz de homografía
H = np.load("RETO_PATOS/homography_matrix/homography_matrix.npy")

def pixel_to_world(pt, H):
    pt_h = np.array([pt[0], pt[1], 1.0])
    world = H @ pt_h
    world /= world[2]
    return world[:2]

# Abrir video
cap = cv2.VideoCapture("VIDEOS_yolo_&_original/video_patos_original.mp4")

# Obtener propiedades del video original
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar VideoWriter
out = cv2.VideoWriter('VIDEOS_yolo_&_original/patos_detectados_yolo_entrenado.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con YOLO
    results = model(frame, verbose=False, show=False)[0]

    for i, box in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, conf, cls = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Coordenadas reales
        real_coords = pixel_to_world((cx, cy), H)
        print(f"🟢 Patito {i}: {real_coords}")

        # Dibujar caja y centro
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{i}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Guardar frame procesado
    out.write(frame)

    # Opcional: mostrar mientras graba
    # cv2.imshow("YOLOv8 Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
