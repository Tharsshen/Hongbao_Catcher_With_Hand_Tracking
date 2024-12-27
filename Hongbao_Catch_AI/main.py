import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time

# Initialize Pygame for sound effects and visuals
pygame.init()
pygame.mixer.init()

# Sound effects
catch_sound = pygame.mixer.Sound("Resources/catch_sound.mp3")  # Replace with your sound file path
miss_sound = pygame.mixer.Sound("Resources/miss_sound.mp3")    # Replace with your sound file path

# Load images
red_envelope_img = cv2.imread("Resources/red_envelope.png", cv2.IMREAD_UNCHANGED)  # Replace with your red envelope path
firecracker_img = cv2.imread("Resources/firecracker.png", cv2.IMREAD_UNCHANGED)    # Replace with your firecracker path

# Screen settings
WIDTH, HEIGHT = 1400, 600  # Adjusted for a wider screen

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Game variables
score = 0
game_over = False
font = cv2.FONT_HERSHEY_SIMPLEX

# Envelope and firecracker positions
red_envelope = {"x": random.randint(100, WIDTH - 100), "y": 0, "speed": random.randint(5, 10)}
firecracker = {"x": random.randint(100, WIDTH - 100), "y": 0, "speed": random.randint(5, 10)}

# Function to overlay images with scaling
def overlay_image(frame, image, x, y, scale=0.2):
    h, w, _ = image.shape
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (new_w, new_h))

    x1, y1 = max(0, x - new_w // 2), max(0, y - new_h // 2)
    x2, y2 = min(frame.shape[1], x + new_w // 2), min(frame.shape[0], y + new_h // 2)

    # Adjust dimensions of the resized image if it goes out of frame
    image_cropped = image_resized[:y2 - y1, :x2 - x1]

    overlay = frame[y1:y2, x1:x2]
    alpha_mask = image_cropped[:, :, 3] / 255.0  # Assuming the image has an alpha channel

    for c in range(3):  # Apply to each channel
        overlay[..., c] = overlay[..., c] * (1 - alpha_mask) + image_cropped[:, :, c] * alpha_mask

    frame[y1:y2, x1:x2] = overlay

# Function to detect collisions
def check_collision(hand_x, hand_y, obj):
    return obj["x"] - 50 < hand_x < obj["x"] + 50 and obj["y"] - 50 < hand_y < obj["y"] + 50

# Main game loop
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_x, hand_y = -1, -1
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            hand_x, hand_y = x, y
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Move red envelope
    red_envelope["y"] += red_envelope["speed"]
    if red_envelope["y"] > HEIGHT:
        red_envelope["y"] = 0
        red_envelope["x"] = random.randint(100, WIDTH - 100)
        score -= 1  # Penalty for missing
        pygame.mixer.Sound.play(miss_sound)

    # Move firecracker
    firecracker["y"] += firecracker["speed"]
    if firecracker["y"] > HEIGHT:
        firecracker["y"] = 0
        firecracker["x"] = random.randint(100, WIDTH - 100)

    # Check collisions
    if check_collision(hand_x, hand_y, red_envelope):
        red_envelope["y"] = 0
        red_envelope["x"] = random.randint(100, WIDTH - 100)
        score += 10  # Award points for catching
        pygame.mixer.Sound.play(catch_sound)

    if check_collision(hand_x, hand_y, firecracker):
        firecracker["y"] = 0
        firecracker["x"] = random.randint(100, WIDTH - 100)
        score -= 10  # Penalty for hitting firecracker
        pygame.mixer.Sound.play(miss_sound)

    # Draw red envelope and firecracker
    overlay_image(frame, red_envelope_img, red_envelope["x"], red_envelope["y"], scale=0.2)
    overlay_image(frame, firecracker_img, firecracker["x"], firecracker["y"], scale=0.2)

    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 50), font, 1, (255, 255, 255), 2)

    # Check for game over condition
    if score < -20:  # Example condition for game over
        cv2.putText(frame, "Game Over", (WIDTH // 2 - 100, HEIGHT // 2), font, 2, (0, 0, 255), 3)
        game_over = True
        break

    # Show the frame
    cv2.imshow("Catch the Red Envelopes", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
