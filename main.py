import cv2
from ultralytics import YOLO
import math

# ====================================================
# TRAFFIC SIGN RECOGNITION SYSTEM
# ====================================================

print("⏳ STATUS: Loading Model... Please wait.")

# 1. Load the trained model
# Ensure 'best.pt' is in the same folder as this script
try:
    model = YOLO('best.pt')
except:
    print("❌ ERROR: 'best.pt' file not found. Please check the folder.")
    exit()

# 2. Initialize the Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Set Width
cap.set(4, 480) # Set Height

# 3. Correction Function
# This function fixes the wrong labels from the dataset.
def get_corrected_name(class_id, default_name):
    
    # FIX 1: If Model detects ID 1 (Speed Limit 50/20/30 mix), show "Speed Limit 30"
    if class_id == 1:
        return "Speed Limit 30"
    
    # FIX 2: If Model detects ID 11 (Keep Left), show "Turn Right"
    elif class_id == 11:
        return "Turn Right"
        
    # For all other signs, use the default name from the model
    else:
        return default_name

print("✅ SYSTEM READY: Camera is ON. Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ ERROR: Could not read from webcam.")
        break

    # Run AI prediction on the frame
    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Get Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 2. Get Confidence and Class ID
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0]) 
            
            # 3. Get the Correct Name using our fix function
            original_name = model.names[cls_id]
            final_name = get_corrected_name(cls_id, original_name)

            # Only show if confidence is above 45%
            if conf > 0.45:
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Prepare the label text
                label = f"{final_name}" 
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show the final image
    cv2.imshow('Traffic Sign Recognition System', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 System Closed.")
