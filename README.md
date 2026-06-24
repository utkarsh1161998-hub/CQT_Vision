import cv2
import numpy as np
from typing import Dict, Any

def analyze_quartz_clarity(image_path: str) -> Dict[str, Any]:
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Segment the stone from the background (Assuming bright backlight)
    # Adjust the threshold value depending on your background brightness
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours to isolate the largest object (the stone)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "Error", "message": "No stone detected"}
        
    stone_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the stone
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [stone_contour], -1, 255, -1)
    
    # Erode the mask slightly to ignore outer boundary edges during internal analysis
    kernel = np.ones((5, 5), np.uint8)
    internal_mask = cv2.erode(mask, kernel, iterations=2)
    
    # 3. Analyze Cloudiness / Transparency (Standard Deviation of Intensity)
    # High-quality clear quartz has a highly uniform light distribution (low variance)
    mean_val, std_dev = cv2.meanStdDev(gray, mask=internal_mask)
    cloudiness_score = std_dev[0][0]  # Higher standard deviation = more uneven/cloudy
    
    # 4. Analyze Inclusions / Cracks (Edge Density)
    # Canny edge detection finds sharp cracks or dark internal inclusion borders
    edges = cv2.Canny(gray, 30, 100)
    # Isolate edges only *inside* the stone
    internal_edges = cv2.bitwise_and(edges, internal_mask)
    
    total_stone_pixels = np.sum(internal_mask == 255)
    edge_pixels = np.sum(internal_edges == 255)
    
    # Prevent division by zero
    edge_density = (edge_pixels / total_stone_pixels) * 100 if total_stone_pixels > 0 else 0
    
    # 5. Classification Logic (Adjust thresholds based on your grading scale)
    # Perfect clear quartz has low cloudiness and close to 0 edge density.
    if edge_density < 0.5 and cloudiness_score < 15:
        grade = "Grade AAA (Water Clear)"
    elif edge_density < 1.5 and cloudiness_score < 30:
        grade = "Grade AA (Minor Inclusions)"
    else:
        grade = "Grade B/C (Cloudy / Heavily Fractured)"
        
    # 6. Visualization output
    output_img = img.copy()
    cv2.drawContours(output_img, [stone_contour], -1, (0, 255, 0), 2)
    
    return {
        "grade": grade,
        "cloudiness_score": round(float(cloudiness_score), 2),
        "inclusion_edge_density_pct": round(float(edge_density), 4),
        "visual_result": output_img
    }

# --- Execution Example ---
if __name__ == "__main__":
    try:
        # Replace with your actual image path from the vision sensor
        result = analyze_quartz_clarity("quartz_tumble.jpg")
        
        print(f"--- Analysis Results ---")
        print(f"Assigned Grade: {result['grade']}")
        print(f"Cloudiness Score (Lower is better): {result['cloudiness_score']}")
        print(f"Inclusion Density: {result['inclusion_edge_density_pct']}%")
        
        # Display the monitored output
        cv2.imshow("Detected Stone Boundaries", result["visual_result"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Execution failed: {e}")
