import random
import config

class Button:
    def __init__(self, x, y, size, id):
        self.x = x
        self.y = y
        self.size = size
        self.rect = (x, y, size, size) # Used for drawing
        self.id = id      # Unique ID for the button slot (0-9)
        self.value = str(id) # The actual number displayed (scrambled)

class SecurityLogic:
    def __init__(self):
        self.buttons = []
        self.create_grid()
        self.scramble_keypad() # Initial scramble

    def create_grid(self):
        """Define the 3x4 grid layout centered on screen."""
        cols = 3
        gap = 20
        btn_size = 100
        
        # Calculate starting X to center the grid
        total_width = (cols * btn_size) + ((cols - 1) * gap)
        start_x = (config.CAM_WIDTH - total_width) // 2
        start_y = 100

        # Create 10 buttons (0-9)
        for i in range(9):
            row = i // 3
            col = i % 3
            x = start_x + (col * (btn_size + gap))
            y = start_y + (row * (btn_size + gap))
            self.buttons.append(Button(x, y, btn_size, i))
        
        # Add the '0' button at the bottom center
        zero_x = start_x + (1 * (btn_size + gap))
        zero_y = start_y + (3 * (btn_size + gap))
        self.buttons.append(Button(zero_x, zero_y, btn_size, 9))

    def scramble_keypad(self):
        """Feature 2: Dynamic Scrambling Algorithm"""
        values = list("1234567890")
        random.shuffle(values)
        for i, btn in enumerate(self.buttons):
            btn.value = values[i]

    def get_hovered_button(self, cursor_x, cursor_y):
        """Checks if the cursor (x,y) is inside any button."""
        if cursor_x is None: return None
        
        for btn in self.buttons:
            if (btn.x < cursor_x < btn.x + btn.size) and \
               (btn.y < cursor_y < btn.y + btn.size):
                return btn
        return None

    def detect_pinch(self, hand_landmarks):
        """
        Feature 3: Pinch-Distance Algorithm.
        Returns True if Thumb (4) and Index (8) are touching.
        """
        if not hand_landmarks:
            return False
            
        # Get coordinates of Thumb Tip (4) and Index Tip (8)
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        
        # Calculate Euclidean Distance
        # We don't need sqrt() for simple thresholding
        distance = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
        
        return distance < config.PINCH_THRESHOLD