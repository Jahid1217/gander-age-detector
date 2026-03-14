import cv2
from deepface import DeepFace

def main():
    # Attempt to open the webcam
    camera_index = 0
    video = None

    print("Looking for webcam...")
    while camera_index < 3:
        video = cv2.VideoCapture(camera_index)
        if video.isOpened():
            print(f"✓ Camera {camera_index} opened successfully")
            break
        camera_index += 1

    if not video or not video.isOpened():
        print("Error: Could not open any camera!")
        exit(1)

    # Set camera resolution (optional)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting video stream. Press 'q' or 'ESC' to exit.")

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        
        # We don't need to analyze every single frame. Analyzing every N frames
        # provides smoother video feedback while still updating attributes quickly.
        if frame_count % 5 == 0:
            try:
                # Analyze the frame for age and gender using DeepFace
                # enforce_detection=False prevents crashes if no face is found
                results = DeepFace.analyze(
                    frame, 
                    actions=['age', 'gender'], 
                    enforce_detection=False,
                    silent=True
                )
                
                # DeepFace.analyze can return a list of dictionaries if multiple faces are found,
                # or a single dictionary. We handle both:
                if not isinstance(results, list):
                    results = [results]

                for result in results:
                    # DeepFace returns 'region' dict with x, y, w, h
                    region = result.get('region', {})
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)

                    age = result.get('age', 'Unknown')
                    
                    # 'gender' is typically a dict probabilities. Pick the highest.
                    gender_dict = result.get('gender', {})
                    if isinstance(gender_dict, dict):
                        # Extract the key with the highest confidence
                        dominant_gender = max(gender_dict, key=gender_dict.get)
                    else:
                        dominant_gender = str(gender_dict)
                    
                    # Draw bounding box entirely on the frame
                    if w > 0 and h > 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        label = f"{dominant_gender}, {age}"
                        cv2.putText(frame, label, (x, max(10, y - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                # If there is any exception (like face detection issues not caught by enforce_detection), ignore it
                pass

        # Show the frame
        cv2.imshow("DeepFace Age & Gender Detection", frame)

        # Press 'q' to quit or ESC
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Program ended")

if __name__ == "__main__":
    main()
