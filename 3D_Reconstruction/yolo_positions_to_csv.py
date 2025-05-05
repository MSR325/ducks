import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo entrenado
model = YOLO('Yolo_models/best_trained_yolo_model.pt')

# Abrir video
cap = cv2.VideoCapture("VIDEOS_yolo_&_original/video_patos_original.mp4")

# Obtener propiedades del video original
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar VideoWriter
out = cv2.VideoWriter('VIDEOS_yolo_&_original/patos_detectados_yolo_entrenado2.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# Crear archivo de salida
output_path = "3D_Reconstruction/duck_world_positions.csv"
f_out = open(output_path, "w")
f_out.write("frame,duck_id,x_world,y_world\n")

frame_id = 0

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

        # Map from pixel to world coordinates (example)
        x_world = cx / width * 4.5     # Assuming 10m width in world
        y_world = (1 - cy / height) * 3.0  # Assuming 5m height, invert y

        # Write to CSV
        f_out.write(f"{frame_id},{i},{x_world:.4f},{y_world:.4f}\n")

        print(f"🟢 Frame {frame_id} ")

        # Dibujar caja y centro
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"{i}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        

    # Guardar frame procesado
    out.write(frame)

    # Mostrar opcional
    # cv2.imshow("YOLOv8 Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Liberar recursos
cap.release()
out.release()
f_out.close()
cv2.destroyAllWindows()
