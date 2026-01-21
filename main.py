import cv2
import config
from core.perception import PerceptionEngine
from core.state_machine import StateMachine, AppState
from core.logic import SecurityLogic
from core.ui import UserInterface

def get_index_finger_pos(hand_results, width, height):
    """Extracts Index Finger Tip (Landmark 8) coordinates."""
    if hand_results.multi_hand_landmarks:
        # Get the first hand detected
        lm = hand_results.multi_hand_landmarks[0].landmark[8] # 8 = Index Tip
        return int(lm.x * width), int(lm.y * height)
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, config.CAM_WIDTH)
    cap.set(4, config.CAM_HEIGHT)

    # Initialize All Layers
    perception = PerceptionEngine()
    brain = StateMachine()
    logic = SecurityLogic()
    ui = UserInterface()

    print("PhantomKey: All Systems Nominal.")

    while True:
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)

        # 1. PERCEPTION
        hand_results, face_results = perception.process_frame(frame)
        cursor_x, cursor_y = get_index_finger_pos(hand_results, config.CAM_WIDTH, config.CAM_HEIGHT)

        # 2. STATE MACHINE
        current_state = brain.update(hand_results, face_results)

        # 3. LOGIC (Hit Testing)
        hovered_btn = None
        if current_state == AppState.TRACKING:
            # Check if finger is over a button
            hovered_btn = logic.get_hovered_button(cursor_x, cursor_y)
            if hovered_btn:
                current_state = AppState.HOVER

            # SIMULATED CLICK (Temporary for testing)
            # If you hover for a long time, or press 'c' on keyboard, we scramble
            # This is just to test Feature 2 before we build the complex click gesture
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("Click Detected! Scrambling...")
                logic.scramble_keypad() # Test Feature 2

        # 4. RENDERING
        ui.draw_keypad(frame, logic.buttons, hovered_btn, current_state.value)
        ui.draw_cursor(frame, cursor_x, cursor_y)
        
        # Draw status text
        cv2.putText(frame, f"State: {current_state.value}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('PhantomKey Alpha', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()