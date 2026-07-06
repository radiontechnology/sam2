import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import time

# --- ACTUAL SAM 1.0/SAM 2.0 (Segment Anything) IMPORTS ---
# NOTE: The provided example uses SAM 1.0 (segment_anything library).
# We are adapting the class to use SAM2.
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMSegmentor:
    """
    A class wrapper for the original SAM model to perform image segmentation 
    on a SINGLE image file. (Adapted from the user's SAM 2.1 wrapper template).
    """
    
    # --- Configuration Defaults (Updated for SAM 1.0 logic) ---
    _BASE_DIR = os.path.dirname(__file__)
    """
    DEFAULT_MODEL_TYPE = "hiera_b+"
    DEFAULT_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints/sam2.1_hiera_base_plus.pt")
    DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    """
    DEFAULT_MODEL_TYPE = "hiera_t"
    DEFAULT_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints/sam2.1_hiera_tiny.pt")
    DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    def __init__(
        self, 
        model_type: str = DEFAULT_MODEL_TYPE, 
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        model_cfg_path: str = DEFAULT_MODEL_CFG,
        device: Optional[str] = None
    ):
        """
        Initializes the SAMSegmentor.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint (.pth file).
            device (Optional[str]): PyTorch device ('cuda' or 'cpu'). Defaults 
                                    to 'cuda' if available, otherwise 'cpu'.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model_cfg_path = model_cfg_path
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.predictor: Optional[SAM2ImagePredictor] = None
        
        print(f"SAMSegmentor initialized. Device: {self.device}")

    # --------------------------------------------------------------------------
    # --- Helper Functions (Static methods retained) ---------------------------
    # --------------------------------------------------------------------------

    @staticmethod
    def _create_masked_images(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the mask to the image to create foreground (with alpha) and background images."""
        
        # Foreground values are set to 0
        mask = ~mask

        # 1. Create Foreground (with transparent background)
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        mask_3d = np.expand_dims(mask, axis=-1)
        
        foreground = np.where(mask_3d, image_bgra, 0)
        # Set alpha channel: opaque (255) where mask is True, transparent (0) otherwise
        foreground[..., 3] = (mask.astype(np.uint8) * 255)

        # 2. Create Background (with foreground blacked out)
        background = image.copy()
        background[mask.astype(bool)] = [0, 0, 0] # Set masked area to black
        
        return foreground, background

    @staticmethod
    def _show_masks_on_image(image_rgb: np.ndarray, masks: np.ndarray, output_path: str):
        """Overlays all provided masks on the image with random colors and saves it."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image_rgb)

        # Create a composite overlay for all masks
        # masks is an array of shape (N, H, W)
        composite = np.zeros((masks.shape[1], masks.shape[2], 4), dtype=np.float32)
        for i in range(masks.shape[0]):
            mask = masks[i]
            # Random color with 50% opacity
            color = np.concatenate([np.random.random(3), np.array([0.5])]) 
            mask_3d = np.expand_dims(mask, axis=-1)
            # Add the colored mask to the composite image
            composite += np.where(mask_3d, color, 0)

        plt.imshow(composite)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # --------------------------------------------------------------------------
    # --- Core Logic Methods ---------------------------------------------------
    # --------------------------------------------------------------------------

    def load_model(self):
        """Builds and loads the SAM model and sets up the predictor."""
        if self.predictor is not None:
            print("Model already loaded.")
            return

        if not os.path.exists(self.checkpoint_path):
            print(f"Error: Model checkpoint not found at {self.checkpoint_path}")
            print("Please download a model checkpoint and update checkpoint_path.")
            return

        print(f"Loading SAM model ({self.model_type}) from {self.checkpoint_path}...")

        # 1. Build the SAM2 model
        sam_model = build_sam2(self.model_cfg_path, self.checkpoint_path)
        sam_model.to(device=self.device)

        # 2. Create the SAM2 predictor
        self.predictor = SAM2ImagePredictor(sam_model)
        print("Model loaded successfully.")

    def _post_process_mask(self, masks: np.ndarray, scores: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Post-processes the raw SAM output masks to select the main foreground object.
        - Selects the mask with the highest score (the one SAM is most confident about).
        - No connected component logic is applied here, as the full bounding box 
          prompt often yields a good single-object mask based on the score.
        """
        
        if len(masks) == 0:
            return np.zeros((H, W), dtype=bool)

        best_mask = masks[np.argmax(scores)]

        vertical_size = H//100

        kernel = np.zeros((vertical_size, vertical_size), np.uint8)
        kernel[:, vertical_size//2] = 1
        best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

        kernel = np.zeros((vertical_size, vertical_size), np.uint8)
        kernel[vertical_size//2,:] = 1
        best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        
        return best_mask

    def segment_image(self, input_path: str, output_base_path: str, save_debug: bool = False):
        """
        Runs the segmentation process on a single image and saves the results.

        Args:
            input_path (str): The full path to the input image file.
            output_base_path (str): The base path/filename for the output files 
                                    (e.g., '/path/to/my_output'). Extensions will 
                                    be added automatically.
            save_debug (bool): Flag to save all intermediate/debug files.
        """
        if self.predictor is None:
            self.load_model()
            if self.predictor is None: # Check again if loading failed
                print("Cannot proceed: SAM model not loaded.")
                return

        print(f"\n--- Processing: {os.path.basename(input_path)} ---")

        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"Error: cannot read image {input_path}")
            return
        # SAM expects images in RGB format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = image_bgr.shape

        # Apply a light blur before segmentation (as in your example)
        image_rgb_filtered = cv2.medianBlur(image_rgb, 3)

        # --- Run SAM predictor ---
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            print("Processing image embeddings...")
            self.predictor.set_image(image_rgb_filtered)

            # Prompting with the full bounding box
            input_box = np.array([0, 0, W, H])
            print(f"Using bounding box prompt: {input_box}")

            print("Running prediction...")
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True,
            )

        if len(masks) == 0:
            print("No masks returned.")
            return

        # --- Post-process to get the best foreground mask ---
        best_mask = self._post_process_mask(masks, scores, H, W)

        
        fg, bg = self._create_masked_images(image_bgr, best_mask)
        
        
        if output_base_path:
        
            # --- Define and Save results ---
            all_masks_path = f"{output_base_path}_all_masks.png"
            fg_path = f"{output_base_path}_foreground.png"
            bg_path = f"{output_base_path}_background.jpg"
            mask_path = f"{output_base_path}_mask.png"

            
            # Save the foreground
            cv2.imwrite(fg_path, fg)
            
            print(f"Saved results to: {os.path.dirname(output_base_path) or './'}")
            print(f"- Foreground (w/ alpha): {os.path.basename(fg_path)}")
            
            if save_debug:
                # Save the composite of all masks (from your helper)
                self._show_masks_on_image(image_rgb, masks, all_masks_path)
                # Save the background (masked)
                cv2.imwrite(bg_path, bg)
                # Save the best binary mask itself
                cv2.imwrite(mask_path, (best_mask.astype(np.uint8) * 255))

                print(f"- All masks overlay: {os.path.basename(all_masks_path)}")
                print(f"- Background (masked): {os.path.basename(bg_path)}")
                print(f"- Binary Mask: {os.path.basename(mask_path)}")

        return fg, bg

# --------------------------------------------------------------------------
# --- Example Usage (Only runs when executed directly) ---------------------
# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- CUSTOM CONFIGURATION ---
    # NOTE: You MUST change these paths to your actual file/directories!
    INPUT_FILE = "./test_image.jpg" # A path to an existing image file
    # The output will be saved as: 
    # ./output/result_foreground.png
    # ...and so on.
    OUTPUT_BASE = "./output/result" 
    
    # IMPORTANT: Update these paths to match your local setup!
    # 'vit_l' is medium (approx. 1.25 GB).
    _BASE_DIR = os.path.dirname(__file__)
    """
    DEFAULT_MODEL_TYPE = "hiera_b+"
    DEFAULT_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints/sam2.1_hiera_base_plus.pt")
    DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    """
    DEFAULT_MODEL_TYPE = "hiera_t"
    DEFAULT_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints/sam2.1_hiera_tiny.pt")
    DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # ----------------------------

    # Create a mock image file for testing the logic if it doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Creating mock image at {INPUT_FILE} for demonstration.")
        mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Draw a white circle on a black background for a simple segmentation target
        cv2.circle(mock_img, (128, 128), 60, (255, 255, 255), -1) 
        os.makedirs(os.path.dirname(INPUT_FILE) or '.', exist_ok=True)
        cv2.imwrite(INPUT_FILE, mock_img)
        print("Mock image created.")

    # Ensure output directory exists for the example
    output_dir = os.path.dirname(OUTPUT_BASE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        segmentor = SAMSegmentor(
            model_type=MODEL_TYPE,
            checkpoint_path=CHECKPOINT,
            model_cfg_path=MODEL_CFG
        )
        # Process the single image with debug saving
        segmentor.segment_image(
            input_path=INPUT_FILE,
            output_base_path=OUTPUT_BASE,
            save_debug=True
        )
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required file or directory was not found.")
        print(f"Check your paths, especially CHECKPOINT and INPUT_FILE.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")