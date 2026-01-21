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
        cols = 3
        gap = 20
        btn_size = 100
        
        # postiton math 
        total_width = (cols* btn_size)+ ((cols - 1)* gap)
        start_x = (config.CAM_WIDTH - total_width) // 2
        start_y = 100

        # create the buttons,  1-9 in grid
        for i in range(9):
            row = i // 3
            col = i % 3
            x = start_x + (col * (btn_size + gap))
            y = start_y + (row * (btn_size + gap))
            self.buttons.append(Button(x, y, btn_size, i))
        
        # 0' button at the bottom center
        zero_x = start_x + (1 * (btn_size + gap))
        zero_y = start_y + (3 * (btn_size + gap))
        self.buttons.append(Button(zero_x, zero_y, btn_size, 9))

    def scramble_keypad(self):
        values = list("1234567890")
        random.shuffle(values)
        for i, btn in enumerate(self.buttons):
            btn.value = values[i]

    def get_hovered_button(self, cursor_x, cursor_y):
        if cursor_x is None: return None
        
        for btn in self.buttons:
            if (btn.x < cursor_x < btn.x + btn.size) and (btn.y < cursor_y < btn.y + btn.size):
                return btn
        return None