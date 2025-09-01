#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Recognition System with Emotion Detection
ระบบตรวจจับและระบุใบหน้าแบบ Real-time พร้อมระบุชื่อและอารมณ์

Author: Wachirawit Raksa
Modified: 2025
Requirements: Python 3.11.9, OpenCV, numpy, mediapipe, deepface, face_recognition
"""

# Standard library imports
import os
import time
import pickle
import threading
from queue import Queue
from typing import List, Tuple, Dict, Optional
from collections import deque

# Third-party imports
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import face_recognition


class FaceRecognitionSystem:
    """Face Recognition System with Emotion Detection and PKL Dataset Support"""

    def __init__(self, dataset_path: str = "dataset/", pkl_path: str = "trained/face_dataset.pkl"):
        """
        Initialize Face Recognition System

        Args:
            dataset_path: Path to face dataset directory
            pkl_path: Path to save/load PKL file
        """
        # Configuration - Optimized for Real-time Performance
        self.FACE_RECOGNITION_TOLERANCE = 0.55
        self.MIN_FACE_SIZE = 40
        self.EMOTION_ANALYSIS_INTERVAL = 5
        self.FACE_PROCESSING_INTERVAL = 2
        self.MAX_EMOTION_CACHE_SIZE = 30
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.EMOTION_SMOOTHING_FRAMES = 7
        self.PROCESSING_SCALE = 0.6
        self.DISPLAY_SCALE = 1.0
        self.FRAME_BUFFER_SIZE = 2
        self.TARGET_FPS = 60
        self.EMOTION_RESIZE_SIZE = (96, 96)

        # Paths
        self.dataset_path = dataset_path
        self.pkl_path = pkl_path

        # Initialize MediaPipe
        self._initialize_mediapipe()

        # Face data storage
        self.known_face_encodings = []
        self.known_face_names = []

        # Runtime variables
        self._initialize_runtime_variables()

        # Threading for emotion analysis
        self._initialize_threading()

    def _initialize_mediapipe(self):
        """Initialize MediaPipe face detection"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def _initialize_runtime_variables(self):
        """Initialize runtime variables"""
        self.face_history = []
        self.stable_faces = {}
        self.emotion_cache = {}
        self.emotion_history = {}
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_buffer = deque(maxlen=self.FRAME_BUFFER_SIZE)
        self.last_processing_time = time.time()

    def _initialize_threading(self):
        """Initialize threading components"""
        self.emotion_queue = Queue(maxsize=5)
        self.emotion_results = {}
        self.emotion_thread = None
        self.stop_emotion_thread = False

    def save_dataset_to_pkl(self) -> bool:
        """
        Save face dataset to PKL file

        Returns:
            bool: Success status
        """
        try:
            dataset_data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_faces': len(self.known_face_encodings)
            }

            with open(self.pkl_path, 'wb') as f:
                pickle.dump(dataset_data, f)

            print(f"✓ Dataset saved to {self.pkl_path}")
            print(f"  - Total faces: {len(self.known_face_encodings)}")
            print(f"  - File size: {os.path.getsize(self.pkl_path) / 1024:.1f} KB")
            return True

        except Exception as e:
            print(f"✗ Error saving dataset to PKL: {str(e)}")
            return False

    def load_dataset_from_pkl(self) -> bool:
        """
        Load face dataset from PKL file

        Returns:
            bool: Success status
        """
        if not os.path.exists(self.pkl_path):
            print(f"PKL file not found: {self.pkl_path}")
            return False

        try:
            with open(self.pkl_path, 'rb') as f:
                dataset_data = pickle.load(f)

            self.known_face_encodings = dataset_data['encodings']
            self.known_face_names = dataset_data['names']

            print(f"✓ Dataset loaded from {self.pkl_path}")
            print(f"  - Total faces: {dataset_data['total_faces']}")
            print(f"  - Created: {dataset_data.get('created_at', 'Unknown')}")
            print(f"  - File size: {os.path.getsize(self.pkl_path) / 1024:.1f} KB")
            print("-" * 50)
            return True

        except Exception as e:
            print(f"✗ Error loading dataset from PKL: {str(e)}")
            return False

    def load_face_dataset(self, use_pkl: bool = True) -> bool:
        """
        Load face dataset from PKL file or directory

        Args:
            use_pkl: Whether to use PKL file if available

        Returns:
            bool: Success status
        """
        # Try loading from PKL first
        if use_pkl and self.load_dataset_from_pkl():
            return True

        # Fall back to loading from directory
        print("Loading face dataset from directory...")
        return self._load_from_directory()

    def _load_from_directory(self) -> bool:
        """Load face dataset from directory structure"""
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset folder '{self.dataset_path}' not found!")
            return False

        print("Scanning dataset directory...")
        total_loaded = 0

        try:
            for person_name in os.listdir(self.dataset_path):
                person_folder = os.path.join(self.dataset_path, person_name)

                if os.path.isdir(person_folder):
                    person_loaded = 0
                    for image_name in os.listdir(person_folder):
                        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(person_folder, image_name)
                            if self._load_single_face(image_path, person_name):
                                person_loaded += 1
                                total_loaded += 1

                    if person_loaded > 0:
                        print(f"  - {person_name}: {person_loaded} faces loaded")

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False

        print(f"✓ Directory loading complete! Total: {total_loaded} faces loaded")

        # Save to PKL for future use
        if total_loaded > 0:
            self.save_dataset_to_pkl()

        print("-" * 50)
        return total_loaded > 0

    def _load_single_face(self, image_path: str, person_name: str) -> bool:
        """
        Load a single face image and extract encoding

        Args:
            image_path: Path to image file
            person_name: Name of person

        Returns:
            bool: Success status
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(person_name)
                return True

        except Exception as e:
            print(f"✗ Error loading {image_path}: {str(e)}")

        return False

    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Initialize camera with optimized settings

        Returns:
            VideoCapture object or None if failed
        """
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error: Cannot open camera")
            return None

        # Optimize camera settings
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        video_capture.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        return video_capture

    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Detect faces using MediaPipe

        Args:
            frame: Input frame

        Returns:
            List of detected faces with coordinates and encodings
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.PROCESSING_SCALE, fy=self.PROCESSING_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_small_frame)
        detected_faces = []

        if results.detections:
            h, w, _ = frame.shape
            scale_factor = 1.0 / self.PROCESSING_SCALE

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box

                # Calculate bounding box coordinates
                left = max(0, int(bboxC.xmin * small_frame.shape[1] * scale_factor))
                top = max(0, int(bboxC.ymin * small_frame.shape[0] * scale_factor))
                width = int(bboxC.width * small_frame.shape[1] * scale_factor)
                height = int(bboxC.height * small_frame.shape[0] * scale_factor)
                right = min(w, left + width)
                bottom = min(h, top + height)

                # Validate bounding box
                if right > left and bottom > top and (right - left) >= self.MIN_FACE_SIZE:
                    # Extract face encoding
                    rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(
                        rgb_frame_for_encoding,
                        [(top, right, bottom, left)],
                        num_jitters=1
                    )

                    if face_encodings:
                        detected_faces.append((top, right, bottom, left, face_encodings[0]))

        return detected_faces

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize face from encoding

        Args:
            face_encoding: Face encoding to recognize

        Returns:
            Tuple of (name, confidence)
        """
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0

        # Calculate face distances
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if best_distance < self.FACE_RECOGNITION_TOLERANCE:
                confidence = (1 - best_distance) * 100
                return self.known_face_names[best_match_index], confidence

        return "Unknown", 0.0

    def start_emotion_analysis_thread(self):
        """Start background thread for emotion analysis"""
        self.emotion_thread = threading.Thread(target=self._emotion_analysis_worker, daemon=True)
        self.emotion_thread.start()

    def _emotion_analysis_worker(self):
        """Background worker for emotion analysis"""
        while not self.stop_emotion_thread:
            try:
                if not self.emotion_queue.empty():
                    face_id, face_crop = self.emotion_queue.get(timeout=0.1)

                    if face_crop.size > 0 and face_crop.shape[0] > self.MIN_FACE_SIZE and face_crop.shape[
                        1] > self.MIN_FACE_SIZE:
                        try:
                            # Resize for emotion analysis
                            face_crop_resized = cv2.resize(face_crop, self.EMOTION_RESIZE_SIZE)
                            analysis = DeepFace.analyze(
                                face_crop_resized,
                                actions=['emotion'],
                                enforce_detection=False,
                                silent=True
                            )

                            emotion_data = analysis[0] if isinstance(analysis, list) else analysis

                            if 'dominant_emotion' in emotion_data:
                                emotion = emotion_data['dominant_emotion'].capitalize()
                                confidence = emotion_data['emotion'][emotion_data['dominant_emotion']]

                                # Apply emotion smoothing
                                if face_id not in self.emotion_history:
                                    self.emotion_history[face_id] = deque(maxlen=self.EMOTION_SMOOTHING_FRAMES)

                                self.emotion_history[face_id].append((emotion, confidence))
                                smoothed_emotion = self._get_smoothed_emotion(face_id)
                                self.emotion_results[face_id] = smoothed_emotion

                        except Exception:
                            # Use cached emotion if available
                            if face_id in self.emotion_cache:
                                self.emotion_results[face_id] = self.emotion_cache[face_id]
                            else:
                                self.emotion_results[face_id] = "Neutral"
                else:
                    time.sleep(0.01)

            except Exception:
                time.sleep(0.01)

    def _get_smoothed_emotion(self, face_id: str) -> str:
        """
        Get smoothed emotion based on recent history

        Args:
            face_id: Face identifier

        Returns:
            Smoothed emotion string
        """
        if face_id not in self.emotion_history or not self.emotion_history[face_id]:
            return "Neutral"

        emotion_weights = {}
        for emotion, confidence in self.emotion_history[face_id]:
            if emotion not in emotion_weights:
                emotion_weights[emotion] = 0
            emotion_weights[emotion] += confidence

        if emotion_weights:
            return max(emotion_weights, key=emotion_weights.get)

        return "Neutral"

    def get_face_id(self, top: int, right: int, bottom: int, left: int) -> str:
        """
        Generate stable face ID

        Args:
            top, right, bottom, left: Face coordinates

        Returns:
            Stable face ID string
        """
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        size = ((right - left) + (bottom - top)) // 2

        # Apply quantization for stability
        center_x = round(center_x / 25) * 25
        center_y = round(center_y / 25) * 25
        size = round(size / 20) * 20

        return f"{center_x}_{center_y}_{size}"

    def update_stable_faces(self, detected_faces):
        """
        Update stable face tracking

        Args:
            detected_faces: List of detected faces
        """
        current_time = time.time()

        # Update or add new faces
        for face_data in detected_faces:
            top, right, bottom, left, face_encoding = face_data
            face_id = self.get_face_id(top, right, bottom, left)

            if face_id in self.stable_faces:
                # Update existing face
                self.stable_faces[face_id]['last_seen'] = current_time
                self.stable_faces[face_id]['coordinates'] = (top, right, bottom, left)
                self.stable_faces[face_id]['encoding'] = face_encoding
            else:
                # Add new face
                name, confidence = self.recognize_face(face_encoding)
                self.stable_faces[face_id] = {
                    'coordinates': (top, right, bottom, left),
                    'encoding': face_encoding,
                    'name': name,
                    'confidence': confidence,
                    'last_seen': current_time,
                    'emotion': 'Analyzing...'
                }

        # Remove old faces
        faces_to_remove = []
        for face_id, face_info in self.stable_faces.items():
            if current_time - face_info['last_seen'] > 0.01:
                faces_to_remove.append(face_id)

        for face_id in faces_to_remove:
            del self.stable_faces[face_id]

    def manage_emotion_cache(self):
        """Manage emotion cache size"""
        if len(self.emotion_cache) > self.MAX_EMOTION_CACHE_SIZE:
            oldest_keys = list(self.emotion_cache.keys())[:len(self.emotion_cache) - self.MAX_EMOTION_CACHE_SIZE]
            for key in oldest_keys:
                del self.emotion_cache[key]

    def update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def draw_face_annotations(self, frame: np.ndarray, top: int, right: int, bottom: int, left: int,
                              name: str, confidence: float, emotion: str):
        """
        Draw face rectangle and annotations

        Args:
            frame: Frame to draw on
            top, right, bottom, left: Face coordinates
            name: Person name
            confidence: Recognition confidence
            emotion: Detected emotion
        """
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA)

        # Prepare text
        if name != "Unknown":
            text = f"{name} ({confidence:.1f}%) - {emotion}"
        else:
            text = f"{name} - {emotion}"

        # Draw text with background
        text_y = max(top - 10, 20)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        padding = 5
        cv2.rectangle(frame,
                      (left - padding, text_y - text_height - padding * 2),
                      (left + text_width + padding, text_y + padding),
                      color, -1)

        cv2.putText(frame, text, (left, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_status_info(self, frame: np.ndarray):
        """
        Draw status information on frame

        Args:
            frame: Frame to draw on
        """
        h, w = frame.shape[:2]

        status_text = f"Faces: {len(self.stable_faces)} | FPS: {self.current_fps:.1f} |"
        cv2.putText(frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        self.frame_count += 1
        self.update_fps_counter()

        current_time = time.time()

        # Detect faces periodically
        if self.frame_count % self.FACE_PROCESSING_INTERVAL == 0:
            detected_faces = self.detect_faces_mediapipe(frame)
            self.update_stable_faces(detected_faces)

        # Process stable faces
        for face_id, face_info in self.stable_faces.items():
            top, right, bottom, left = face_info['coordinates']
            name = face_info['name']
            confidence = face_info['confidence']
            emotion = face_info.get('emotion', 'Analyzing...')

            # Queue emotion analysis
            if (self.frame_count % self.EMOTION_ANALYSIS_INTERVAL == 0 and
                    current_time - self.last_processing_time > 0.01):

                face_crop = frame[max(0, top):min(frame.shape[0], bottom),
                            max(0, left):min(frame.shape[1], right)]

                if face_crop.size > 0 and not self.emotion_queue.full():
                    self.emotion_queue.put((face_id, face_crop.copy()))
                    self.last_processing_time = current_time

            # Update emotion from results
            if face_id in self.emotion_results:
                emotion = self.emotion_results[face_id]
                face_info['emotion'] = emotion
                self.emotion_cache[face_id] = emotion
                del self.emotion_results[face_id]

            # Draw annotations
            self.draw_face_annotations(frame, top, right, bottom, left, name, confidence, emotion)

        # Manage cache periodically
        if self.frame_count % 100 == 0:
            self.manage_emotion_cache()

        # Draw status
        self.draw_status_info(frame)

        return frame

    def run(self):
        """Main execution loop"""
        print("=" * 60)
        print("Face Recognition System with Emotion Detection")
        print("By Wachirawit Raksa")
        print("=" * 60)

        # Load dataset
        if not self.load_face_dataset():
            print("Warning: No faces loaded from dataset. Only emotion detection will work.")

        # Initialize camera
        video_capture = self.initialize_camera()
        if video_capture is None:
            return

        # Start emotion analysis thread
        self.start_emotion_analysis_thread()

        print("\nStarting face recognition system...")
        print("Press ESC or 'q' to quit")
        print("-" * 60)

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Cannot read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display frame
                cv2.imshow("Face Recognition & Emotion Detection", processed_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break

        except KeyboardInterrupt:
            print("\nProgram interrupted by user")

        finally:
            self.cleanup(video_capture)

    def cleanup(self, video_capture: cv2.VideoCapture):
        """
        Clean up resources

        Args:
            video_capture: VideoCapture object to release
        """
        print("\nCleaning up resources...")

        # Stop emotion analysis thread
        self.stop_emotion_thread = True
        if self.emotion_thread and self.emotion_thread.is_alive():
            self.emotion_thread.join(timeout=0.001)

        # Close MediaPipe
        if hasattr(self, 'face_detection'):
            self.face_detection.close()

        # Release camera and close windows
        video_capture.release()
        cv2.destroyAllWindows()

        print("✓ Program terminated successfully")
        print("=" * 60)


def main():
    """Main function to run the face recognition system"""
    dataset_path = "dataset/"
    pkl_path = "trained/face_dataset.pkl"

    system = FaceRecognitionSystem(dataset_path, pkl_path)
    system.run()


if __name__ == "__main__":
    main()