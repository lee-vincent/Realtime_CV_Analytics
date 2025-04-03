import cv2

def find_epoccam_index():
    """
    Iterates through camera indices to find the EpocCam feed.
    Returns the index if found, otherwise returns -1.
    """
    for index in range(10):  # Check indices 0 through 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to read a frame.  This helps confirm it's a working camera.
            ret, frame = cap.read()
            if ret:
                print(f"Found a working camera at index: {index}")
                # Display the frame for a short time to visually confirm
                cv2.imshow(f"Camera Index {index}", frame)
                cv2.waitKey(1000)  # Display for 1 second
                cv2.destroyAllWindows()
                cap.release()

                user_input = input(f"Is this the EpocCam feed? (y/n): ").lower()
                if user_input == 'y':
                    return index
            else:
                print(f"Camera at index {index} is open but could not read a frame.")
                cap.release()
        else:
            print(f"Camera index {index} is not available.")
    return -1  # Not found

epoccam_index = find_epoccam_index()

if epoccam_index == -1:
    print("Error: Could not find the EpocCam video feed.")
    exit()

# Now use the correct index
cap = cv2.VideoCapture(epoccam_index)

if not cap.isOpened():
    print(f"Error: Could not open video device at index {epoccam_index}.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    cv2.imshow("EpocCam Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
