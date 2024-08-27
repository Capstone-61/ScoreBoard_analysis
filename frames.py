# import cv2
# import pytesseract
# import os
# # Set the temporary directory
# os.environ['TESSDATA_PREFIX'] = 'C:\\Users\\bagar\\Documents\\TesseractTemp'
# # Path to Tesseract executable (required for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'tesseract-ocr-w64-setup-5.4.0.20240606.exe'  


# # Function to capture frames from the video
# def video_to_frames(video_path, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     vidcap = cv2.VideoCapture(video_path)
#     success, image = vidcap.read()
#     count = 0

#     while success:
#         frame_path = os.path.join(output_dir, f"frame{count}.jpg")
#         cv2.imwrite(frame_path, image)  # Save frame as JPEG file
#         success, image = vidcap.read()
#         count += 1
#     vidcap.release()
#     print(f"Extracted {count} frames from the video.")

# # Function to process each frame and extract the scoreboard region
# def detect_scoreboard_and_extract_score(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 200)

#     # Find contours in the edged image, keep only the largest ones
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     # Loop over the contours to find the possible scoreboard
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / float(h)
        
#         # Scoreboards are usually wider than tall 
#         if aspect_ratio > 2 and 200 < w < 1280 and 50 < h < 300:
#             scoreboard_image = frame[y:y+h, x:x+w]
#             score_text = pytesseract.image_to_string(scoreboard_image, config='--psm 7')
#             return score_text.strip(), (x, y, w, h)
    
#     return None, None

# # Process video and capture scores
# def process_video_for_scores(video_path, output_dir):
#     # video_to_frames(video_path, output_dir)
#     scores = []

#     for frame_file in sorted(os.listdir(output_dir)):
#         frame_path = os.path.join(output_dir, frame_file)
#         frame = cv2.imread(frame_path)

#         # Detect scoreboard and extract score
#         score, bbox = detect_scoreboard_and_extract_score(frame)
#         if score:
#             print(f"Score detected in {frame_file}: {score}")
#             scores.append((frame_file, score, bbox))
#             # Optionally, draw the bounding box on the frame
#             if bbox:
#                 x, y, w, h = bbox
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, score, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.imwrite(frame_path, frame)  # Save the frame with bounding box

#     return scores

# # Paths
# video_path = 'Brazil_vs_Belgium.mp4'  
# frames_dir = 'video_frames'

# # Process the video to capture scores
# scores_detected = process_video_for_scores(video_path, frames_dir)

# # Print all detected scores
# for frame_file, score, _ in scores_detected:
#     print(f"{frame_file}: {score}")


import cv2
import pytesseract
import re

# Path to Tesseract executable 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

# Function to detect and extract scores and country names from the scoreboard
def extract_scores_and_teams(frame):
    # Defined the region of interest (ROI) for the scoreboard
    height, width, _ = frame.shape
    scoreboard_region = frame[0:int(height / 5), :]  # Aheight 

    # Converted the region to grayscale and apply some preprocessing
    gray = cv2.cvtColor(scoreboard_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Performed OCR on the defined region
    score_text = pytesseract.image_to_string(blurred, config='--psm 6')  # Use page segmentation mode 6 for sparse text

    # Print the raw OCR output for debugging
    print("OCR Output:", score_text)

    # Used regex to find the score format (e.g., BRA 0 and BEL 2)
    # Adjusted pattern to match variations in the OCR output
    pattern = r'([A-Z]{3})\s*[|:]*\s*(\d+)'  # Matches three uppercase letters followed by ':' or '|' and a number
    matches = re.findall(pattern, score_text)

    # Clean up the matches to get just the team and score format (BRA 0)
    cleaned_matches = [(team.strip(), score.strip()) for team, score in matches]

    return cleaned_matches

# Function to process a specific frame image
def process_specific_frame(frame_path):
    # Read the specific frame image
    frame = cv2.imread(frame_path)

    if frame is None:
        print("Error: Could not read the frame.")
        return

    # Extracted scores and team names
    extracted_scores = extract_scores_and_teams(frame)
  
    # Output the results
    if extracted_scores:
        scores_output = ", ".join([f"{team} {score}" for team, score in extracted_scores])
        print("Scores Detected:", scores_output)
        
    else:
        print("No scores detected.")

    # # displaying the original frame
    # cv2.imshow("Original Frame", frame)  # Show the original frame
    # cv2.waitKey(0)  # Wait for a key press to close the image
    # cv2.destroyAllWindows()

# Path to my specific frame image
frame_path = r'video_frames/frame104759.jpg' 
process_specific_frame(frame_path)