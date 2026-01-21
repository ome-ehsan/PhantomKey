import cv2
import config
from core.perception import PerceptionEngine
from core.state_machine import StateMachine, AppState
from core.logic import SecurityLogic
from core.ui import UserInterface

def get_index_finger_pos(hand_results, width, height):
    # get index finger coordinates 
    if hand_results.multi_hand_landmarks:
        # Get the first hand detected
        lm = hand_results.multi_hand_landmarks[0].landmark[8] # 8 = index finger tip in openCV
        return int(lm.x * width), int(lm.y * height)
    return None, None

def main():
    #setting up the camera
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

        # 2. LOGIC PRE-CHECK (Ask: Is the finger over a button?)
        # We need this *before* the state machine to inform it
        hovered_btn = logic.get_hovered_button(cursor_x, cursor_y)
        is_hovering = (hovered_btn is not None)

        # 3. STATE MACHINE (The Brain decides the state)
        current_state = brain.update(hand_results, face_results, is_hovering)

        # 4. INTERACTION (Neural Net / Pinch Check)
        # Only check for clicks if we are legally allowed to (HOVER state)
        if current_state == AppState.HOVER:
            if hand_results.multi_hand_landmarks:
                # --- THIS IS WHERE THE NEURAL NET WILL GO LATER ---
                is_pinching = logic.detect_pinch(hand_results.multi_hand_landmarks[0])
                
                if is_pinching:
                    print(f"CLICKED Number: {hovered_btn.value}")
                    brain.trigger_click()   
                    logic.scramble_keypad() 
        
        # 5. RENDERING
        if current_state == AppState.DEBOUNCE:
             cv2.circle(frame, (cursor_x, cursor_y), 20, (0, 255, 0), -1)
             
        ui.draw_keypad(frame, logic.buttons, hovered_btn, current_state.value)
        ui.draw_cursor(frame, cursor_x, cursor_y)
        
        cv2.putText(frame, f"State: {current_state.value}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('PhantomKey MVP', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()