import cv2
import numpy as np

class UserInterface:
    def __init__(self):
        # Colors (BGR)
        self.COLOR_NORMAL = (255, 0, 0)   # Blue
        self.COLOR_HOVER = (0, 255, 0)    # Green
        self.COLOR_TEXT = (255, 255, 255) # White
        self.COLOR_LOCKED = (0, 0, 255)   # Red

    def draw_keypad(self, frame, buttons, hovered_btn, state_name):
        """
        Draws the grid. Handles the 'Secure Hover Masking' logic.
        """
        # If system is LOCKED, draw a warning overlay
        if state_name == "LOCKED":
            cv2.putText(frame, "LOCKED - USER MISSING", (200, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_LOCKED, 3)
            return

        for btn in buttons:
            x, y, s, _ = btn.rect
            
            # 1. Determine Color (Hover vs Normal)
            is_hovered = (hovered_btn and hovered_btn.id == btn.id)
            color = self.COLOR_HOVER if is_hovered else self.COLOR_NORMAL
            thickness = -1 if is_hovered else 2 # Fill if hovered

            # 2. Draw Box
            cv2.rectangle(frame, (x, y), (x + s, y + s), color, thickness)

            # 3. Secure Hover Masking (Feature 5) [cite: 34-35]
            # Only show the real value if hovered. Otherwise show '*'
            display_text = btn.value if is_hovered else "*"
            
            # Center the text
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = x + (s - text_size[0]) // 2
            text_y = y + (s + text_size[1]) // 2

            cv2.putText(frame, display_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_TEXT, 3)

    def draw_cursor(self, frame, x, y):
        if x is not None and y is not None:
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)