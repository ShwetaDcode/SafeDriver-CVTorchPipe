import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T
from drowsiness_model import mobilenet_v3_small # import model definition if needed


# Load quantized MobileNetV3
model = mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load('drowsiness_quantized.pth', map_location='cpu'))
model.eval()


mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
alert_counter = 0


while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Convert BGR frame to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    prediction_made = False

    if result.detections:
        for det in result.detections:
            # 1. Get bounding box coordinates
            box = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

            # 2. Extract Face ROI
            face_roi = frame[max(0,y1):y2, max(0,x1):x2]
            if face_roi.size == 0: 
                continue
            
            # 3. Preprocess and Predict
            inp = transform(face_roi).unsqueeze(0)

            with torch.no_grad():
                pred = model(inp).argmax(dim=1).item()
            
            prediction_made = True

            # 4. Draw box and label
            label = 'Drowsy' if pred == 1 else 'Awake'
            color = (0,0,255) if pred == 1 else (0,255,0) # Red for Drowsy, Green for Awake

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.7, color, 2)

            # 5. Alert Logic (simplified)
            if pred == 1:
                alert_counter += 1
                if alert_counter > 30: # If drowsy for 30 consecutive frames
                    cv2.putText(frame, "!!! WAKE UP !!!", (50, 50), font, 1, (0, 0, 255), 3)
            else:
                alert_counter = 0

    
    # 6. Display the frame
    cv2.imshow('Driver Drowsiness Detection', frame)
    
    # 7. Check for exit key ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows() # Moved outside the loop