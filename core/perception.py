import cv2
import mediapipe as mp
import numpy as np
import config

class PerceptionEngine:
    def __init__(self):
        # mediapipe hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

        # media pipe face
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # needed for eye tracking
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )

        # for debugging
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):

        # brg to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # to improve performance, image kept read only 
        frame_rgb.flags.writeable = False
        
        # running inference 
        hand_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        
        # restore writeable flag
        frame_rgb.flags.writeable = True

        return hand_results, face_results

    def draw_debug(self, frame, hand_results, face_results):
        # Draw Hand Landmarks 
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Draw Face Mesh 
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )