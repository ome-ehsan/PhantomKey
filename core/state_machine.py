import time
from enum import Enum

class AppState(Enum):
    IDLE = "IDLE"           # No hand detected
    TRACKING = "TRACKING"   # Hand detected, moving
    HOVER = "HOVER"         # Cursor over a button 
    CLICKED = "CLICKED"     # Trigger registered
    DEBOUNCE = "DEBOUNCE"   # Cool-down period
    LOCKED = "LOCKED"       # Security Violation 

class StateMachine:
    def __init__(self):
        self.current_state = AppState.IDLE
        self.last_click_time = 0
        self.debounce_duration = 0.5  # 0.5s lockout

    def update(self, hand_results, face_results, is_hovering=False):
        current_time = time.time()
        #no face detected m immediate lockdown 
        if not face_results.multi_face_landmarks:
            self.current_state = AppState.LOCKED
            return self.current_state
        
        # ignore everything if recently cliked 
        if self.current_state == AppState.DEBOUNCE:
            if (current_time - self.last_click_time) > self.debounce_duration:
                self.current_state = AppState.IDLE # reset to IDLE after cooldown
            else:
                return AppState.DEBOUNCE

        # hand Presence & hover Interaction
        if hand_results.multi_hand_landmarks:
            if is_hovering:
                self.current_state = AppState.HOVER
            else:
                self.current_state = AppState.TRACKING
        else:
            self.current_state = AppState.IDLE

        return self.current_state

    def trigger_click(self):
        """Call this when a click is detected to start the cooldown"""
        self.current_state = AppState.DEBOUNCE
        self.last_click_time = time.time()