import os
import cv2
import numpy as np
import insightface
import warnings
from tqdm import tqdm 

# --- Configuration ---
# 1. SET YOUR INPUT AND OUTPUT FOLDERS HERE
INPUT_DATA_DIR = "NDB"
OUTPUT_DB_DIR = "New_face_db"

# 2. CHOOSE YOUR EXECUTION PROVIDER ('CPU' or 'CUDA')
# Use 'CUDAExecutionProvider' if you have a compatible NVIDIA GPU and CUDA installed.
# Otherwise, use 'CPUExecutionProvider'.
PROVIDER ='CUDAExecutionProvider'

# 3. MODEL CONFIG
MODEL_NAME = "buffalo_l"

# --- End of Configuration ---


# --- Suppress Unnecessary Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """
    Main function to process the image dataset.
    """
    # --- Basic Checks ---
    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory not found at '{INPUT_DATA_DIR}'")
        print("Please update the 'INPUT_DATA_DIR' variable in the script.")
        return

    # Create the main output directory if it doesn't exist
    os.makedirs(OUTPUT_DB_DIR, exist_ok=True)

    # --- Initialize InsightFace Model (do this once) ---
    print(f"Loading InsightFace model '{MODEL_NAME}' using {PROVIDER}...")
    try:
        face_app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=[PROVIDER])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace model loaded successfully.")
    except Exception as e:
        print(f"Error loading InsightFace model: {e}")
        print("Please ensure you have the correct provider (CPU/CUDA) and all dependencies are installed.")
        return

    # --- Main Processing Loop ---
    # Get a list of person directories (e.g., ['person_a', 'person_b'])
    person_dirs = [d for d in os.listdir(INPUT_DATA_DIR) if os.path.isdir(os.path.join(INPUT_DATA_DIR, d))]
    
    print(f"\nFound {len(person_dirs)} person directories. Starting processing...")

    for person_name in person_dirs:
        input_person_dir = os.path.join(INPUT_DATA_DIR, person_name)
        output_person_dir = os.path.join(OUTPUT_DB_DIR, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        print(f"\nProcessing directory for: {person_name}")

        # Get list of images and create a progress bar
        image_files = [f for f in os.listdir(input_person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        face_idx = 0 # Counter for saved faces for this person

        for image_name in tqdm(image_files, desc=f"  Images for {person_name}"):
            image_path = os.path.join(input_person_dir, image_name)

            try:
                # Read the image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue

                # Process the frame with insightface to find faces
                faces = face_app.get(frame)

                # Loop through all faces found in the image
                for face in faces:
                    # Get the bounding box and crop the face
                    bbox = face.bbox.astype(int)
                    face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Ensure the cropped image is not empty before saving
                    if face_crop.size > 0:
                        # Construct the output file paths
                        emb_path = os.path.join(output_person_dir, f"{person_name}_{face_idx}.npy")
                        img_path = os.path.join(output_person_dir, f"{person_name}_{face_idx}.jpg")

                        # Save the embedding vector and the cropped face image
                        np.save(emb_path, face.embedding)
                        cv2.imwrite(img_path, face_crop)
                        
                        face_idx += 1

            except Exception as e:
                print(f"\nAn error occurred processing {image_path}: {e}")
                continue
        
        print(f"  -> Finished. Saved {face_idx} total faces for {person_name}.")

    print("\n---------------------------------")
    print("Dataset generation complete!")
    print(f"Output saved in: '{OUTPUT_DB_DIR}'")
    print("---------------------------------")


# --- Main Execution Guard ---
if __name__ == "__main__":
    main()