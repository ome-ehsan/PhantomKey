import cv2
import config
from core.perception import PerceptionEngine

def main():


    # setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(3, config.CAM_WIDTH)
    cap.set(4, config.CAM_HEIGHT)

    # init perception layer 
    perception = PerceptionEngine()

    print("[PhantomKey Perception Layer Initialized]")

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # PROCESS FRAME: Get Hand & Face Data
        hand_results, face_results = perception.process_frame(frame)

        # VISUALIZE: Draw the skeleton to prove it works
        perception.draw_debug(frame, hand_results, face_results)
        
        # Display Status
        status_text = "[System: OK]"
        if not face_results.multi_face_landmarks:
             # Feature 4: Liveness Check Fail
            status_text = "LOCKED: No Face Detected" 
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
             cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('PhantomKey - Perception Layer', frame)

        if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()