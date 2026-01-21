import cv2
import config
from core.perception import PerceptionEngine
from core.state_machine import StateMachine, AppState

def main():
    # setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(3, config.CAM_WIDTH)
    cap.set(4, config.CAM_HEIGHT)

    # Initialize Layers 1 & 2
    perception = PerceptionEngine()
    brain = StateMachine()

    print("PhantomKey: Perception & State Machine Initialized.")

    while True:
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)

        # perception layer
        hand_results, face_results = perception.process_frame(frame)

        #state machine state 
        current_state = brain.update(hand_results, face_results)

        # debugging visualization
        perception.draw_debug(frame, hand_results, face_results)
        
        # colors represents states 
        color = (0, 255, 0) # Green for Good
        if current_state == AppState.LOCKED:
            color = (0, 0, 255) # Red for Locked
        elif current_state == AppState.IDLE:
            color = (255, 255, 0) # Cyan for Idle

        cv2.putText(frame, f"STATE: {current_state.value}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('PhantomKey Alpha', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()