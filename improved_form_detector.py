import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from transformers import AutoProcessor, AutoModel
import torch
from sklearn.cluster import DBSCAN
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FormField:
    """Data class to represent a detected form field"""
    field_name: str
    field_type: str  # text, checkbox, radio, dropdown, signature
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    value: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FormStructure:
    """Data class to represent the overall form structure"""
    fields: List[FormField]
    document_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedFormDetector:
    """Advanced form field detector using multiple computer vision and ML techniques"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize models and processors
        self._initialize_models()
        
        # Field detection patterns
        self.field_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|THB)',
            'postal_code': r'\b\d{5}(?:-\d{4})?\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
    def _initialize_models(self):
        """Initialize the required models and processors"""
        try:
            # Use a lightweight document understanding model
            self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
            self.model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
            self.model.to(self.device)
            logger.info("LayoutLMv3 model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LayoutLMv3 model: {e}")
            self.processor = None
            self.model = None
    
    def detect_form_fields(self, image_path: str) -> FormStructure:
        """Main method to detect form fields from an image"""
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            
            # Detect form structure using multiple approaches
            cv_fields = self._detect_fields_with_cv(image)
            ml_fields = self._detect_fields_with_ml(image) if self.model else []
            pattern_fields = self._detect_fields_with_patterns(image)
            
            # Combine and deduplicate fields
            all_fields = cv_fields + ml_fields + pattern_fields
            final_fields = self._deduplicate_fields(all_fields)
            
            # Determine document type
            doc_type = self._classify_document_type(final_fields, image)
            
            # Calculate overall confidence
            avg_confidence = np.mean([f.confidence for f in final_fields]) if final_fields else 0.0
            
            return FormStructure(
                fields=final_fields,
                document_type=doc_type,
                confidence=avg_confidence,
                metadata={"processing_method": "hybrid_cv_ml", "total_fields": len(final_fields)}
            )
            
        except Exception as e:
            logger.error(f"Error in form field detection: {e}")
            return FormStructure(fields=[], document_type="unknown", confidence=0.0)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        if isinstance(image_path, str):
            if image_path.lower().endswith('.pdf'):
                return self._extract_image_from_pdf(image_path)
            else:
                image = cv2.imread(image_path)
        else:
            image = np.array(image_path)
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        return image
    
    def _extract_image_from_pdf(self, pdf_path: str) -> np.ndarray:
        """Extract first page from PDF as image"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling
            img_data = pix.tobytes("ppm")
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            doc.close()
            return img
        except Exception as e:
            logger.error(f"Error extracting image from PDF: {e}")
            raise
    
    def _detect_fields_with_cv(self, image: np.ndarray) -> List[FormField]:
        """Detect form fields using computer vision techniques"""
        fields = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text boxes
        text_fields = self._detect_text_boxes(gray)
        fields.extend(text_fields)
        
        # Detect checkboxes
        checkbox_fields = self._detect_checkboxes(gray)
        fields.extend(checkbox_fields)
        
        # Detect form lines and boxes
        form_elements = self._detect_form_elements(gray)
        fields.extend(form_elements)
        
        return fields
    
    def _detect_text_boxes(self, gray_image: np.ndarray) -> List[FormField]:
        """Detect text input fields using edge detection"""
        fields = []
        
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter contours by area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Text boxes typically have a certain aspect ratio
            if 2 < aspect_ratio < 10 and h > 15:
                # Extract text from this region
                roi = gray_image[y:y+h, x:x+w]
                try:
                    text = pytesseract.image_to_string(roi, config='--psm 7').strip()
                    if len(text) > 0:
                        fields.append(FormField(
                            field_name=f"text_field_{len(fields)}",
                            field_type="text",
                            coordinates=(x, y, x+w, y+h),
                            value=text,
                            confidence=0.7,
                            metadata={"detection_method": "cv_contour"}
                        ))
                except Exception:
                    # If OCR fails, still register as potential text field
                    fields.append(FormField(
                        field_name=f"text_field_{len(fields)}",
                        field_type="text",
                        coordinates=(x, y, x+w, y+h),
                        confidence=0.5,
                        metadata={"detection_method": "cv_contour"}
                    ))
        
        return fields
    
    def _detect_checkboxes(self, gray_image: np.ndarray) -> List[FormField]:
        """Detect checkboxes using template matching and contour analysis"""
        fields = []
        
        # Create checkbox templates
        checkbox_templates = self._create_checkbox_templates()
        
        for template_name, template in checkbox_templates.items():
            # Template matching
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.6)
            
            for pt in zip(*locations[::-1]):
                h, w = template.shape
                fields.append(FormField(
                    field_name=f"checkbox_{len(fields)}",
                    field_type="checkbox",
                    coordinates=(pt[0], pt[1], pt[0]+w, pt[1]+h),
                    confidence=float(result[pt[1], pt[0]]),
                    metadata={"detection_method": "template_matching", "template": template_name}
                ))
        
        # Also detect square-like contours that might be checkboxes
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Reasonable checkbox size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Checkboxes are roughly square
                if 0.8 < aspect_ratio < 1.2:
                    fields.append(FormField(
                        field_name=f"checkbox_{len(fields)}",
                        field_type="checkbox",
                        coordinates=(x, y, x+w, y+h),
                        confidence=0.6,
                        metadata={"detection_method": "contour_analysis"}
                    ))
        
        return fields
    
    def _create_checkbox_templates(self) -> Dict[str, np.ndarray]:
        """Create checkbox templates for template matching"""
        templates = {}
        
        # Empty checkbox
        empty_box = np.zeros((20, 20), dtype=np.uint8)
        cv2.rectangle(empty_box, (2, 2), (18, 18), 255, 2)
        templates["empty"] = empty_box
        
        # Checked checkbox
        checked_box = empty_box.copy()
        cv2.line(checked_box, (5, 10), (9, 14), 255, 2)
        cv2.line(checked_box, (9, 14), (15, 6), 255, 2)
        templates["checked"] = checked_box
        
        return templates
    
    def _detect_form_elements(self, gray_image: np.ndarray) -> List[FormField]:
        """Detect lines and rectangular form elements"""
        fields = []
        
        # Detect horizontal lines (potential underlines for text fields)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h < 5:  # Horizontal line characteristics
                fields.append(FormField(
                    field_name=f"underline_field_{len(fields)}",
                    field_type="text",
                    coordinates=(x, y-10, x+w, y+10),  # Expand vertically for text area
                    confidence=0.6,
                    metadata={"detection_method": "line_detection"}
                ))
        
        return fields
    
    def _detect_fields_with_ml(self, image: np.ndarray) -> List[FormField]:
        """Detect fields using machine learning model (LayoutLMv3)"""
        if not self.model or not self.processor:
            return []
        
        fields = []
        
        try:
            # Convert image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with LayoutLMv3
            encoding = self.processor(pil_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in encoding.items()})
            
            # Extract features and classify regions
            # Note: This is a simplified version - in practice, you'd need a fine-tuned model
            # for specific form field detection
            
            # For now, we'll use the attention weights to identify important regions
            last_hidden_states = outputs.last_hidden_state
            
            # This is a placeholder - you would implement proper field detection logic here
            # based on the model outputs
            
        except Exception as e:
            logger.error(f"Error in ML-based detection: {e}")
        
        return fields
    
    def _detect_fields_with_patterns(self, image: np.ndarray) -> List[FormField]:
        """Detect fields using regex patterns on extracted text"""
        fields = []
        
        try:
            # Extract all text from image
            text = pytesseract.image_to_string(image)
            
            # Get word bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Apply pattern matching
            for pattern_name, pattern in self.field_patterns.items():
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    # Find approximate location in image
                    # This is simplified - in practice, you'd need more sophisticated text localization
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Find corresponding bounding box
                    word_idx = self._find_word_index(data, start_pos, end_pos)
                    if word_idx is not None:
                        x = data['left'][word_idx]
                        y = data['top'][word_idx]
                        w = data['width'][word_idx]
                        h = data['height'][word_idx]
                        
                        fields.append(FormField(
                            field_name=f"{pattern_name}_{len(fields)}",
                            field_type=pattern_name,
                            coordinates=(x, y, x+w, y+h),
                            value=match.group(),
                            confidence=0.8,
                            metadata={"detection_method": "pattern_matching", "pattern": pattern_name}
                        ))
        
        except Exception as e:
            logger.error(f"Error in pattern-based detection: {e}")
        
        return fields
    
    def _find_word_index(self, ocr_data: Dict, start_pos: int, end_pos: int) -> Optional[int]:
        """Find word index in OCR data corresponding to text position"""
        # Simplified implementation
        current_pos = 0
        for i, word in enumerate(ocr_data['text']):
            if word.strip():
                word_start = current_pos
                word_end = current_pos + len(word)
                
                if start_pos <= word_start and end_pos >= word_end:
                    return i
                
                current_pos = word_end + 1
        
        return None
    
    def _deduplicate_fields(self, fields: List[FormField]) -> List[FormField]:
        """Remove duplicate fields based on coordinate proximity"""
        if not fields:
            return fields
        
        # Convert coordinates to centroids for clustering
        centroids = []
        for field in fields:
            x1, y1, x2, y2 = field.coordinates
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            centroids.append([centroid_x, centroid_y])
        
        # Use DBSCAN clustering to group nearby fields
        clustering = DBSCAN(eps=30, min_samples=1).fit(centroids)
        
        # Keep the field with highest confidence from each cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(fields[i])
        
        deduplicated_fields = []
        for cluster_fields in clusters.values():
            # Keep field with highest confidence
            best_field = max(cluster_fields, key=lambda f: f.confidence)
            deduplicated_fields.append(best_field)
        
        return deduplicated_fields
    
    def _classify_document_type(self, fields: List[FormField], image: np.ndarray) -> str:
        """Classify the type of document based on detected fields"""
        field_types = [f.field_type for f in fields]
        field_values = [f.value for f in fields if f.value]
        
        # Simple rule-based classification
        if any('email' in f.field_type for f in fields):
            return "contact_form"
        elif any('currency' in f.field_type for f in fields):
            return "financial_document"
        elif len([f for f in fields if f.field_type == 'checkbox']) > 3:
            return "survey_form"
        elif any('date' in f.field_type for f in fields):
            return "application_form"
        else:
            return "generic_form"
    
    def process_pdf_document(self, pdf_path: str) -> List[FormStructure]:
        """Process multi-page PDF document"""
        results = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract image from page
                    img = page.to_image(resolution=200)
                    img_array = np.array(img.original)
                    
                    # Detect fields
                    form_structure = self.detect_form_fields(img_array)
                    form_structure.metadata["page_number"] = page_num + 1
                    
                    results.append(form_structure)
                    
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
        
        return results
    
    def export_results(self, form_structure: FormStructure, output_path: str, format: str = "json"):
        """Export detection results to file"""
        data = {
            "document_type": form_structure.document_type,
            "confidence": form_structure.confidence,
            "fields": [
                {
                    "name": field.field_name,
                    "type": field.field_type,
                    "coordinates": field.coordinates,
                    "value": field.value,
                    "confidence": field.confidence,
                    "metadata": field.metadata
                }
                for field in form_structure.fields
            ],
            "metadata": form_structure.metadata
        }
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            df = pd.DataFrame([
                {
                    "field_name": field.field_name,
                    "field_type": field.field_type,
                    "x1": field.coordinates[0],
                    "y1": field.coordinates[1],
                    "x2": field.coordinates[2],
                    "y2": field.coordinates[3],
                    "value": field.value,
                    "confidence": field.confidence
                }
                for field in form_structure.fields
            ])
            df.to_csv(output_path, index=False)

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = AdvancedFormDetector(use_gpu=True)
    
    # Example usage
    image_path = "sample_form.jpg"  # Replace with actual image path
    
    try:
        # Detect form fields
        result = detector.detect_form_fields(image_path)
        
        print(f"Document Type: {result.document_type}")
        print(f"Overall Confidence: {result.confidence:.2f}")
        print(f"Number of fields detected: {len(result.fields)}")
        
        for i, field in enumerate(result.fields):
            print(f"\nField {i+1}:")
            print(f"  Name: {field.field_name}")
            print(f"  Type: {field.field_type}")
            print(f"  Coordinates: {field.coordinates}")
            print(f"  Value: {field.value}")
            print(f"  Confidence: {field.confidence:.2f}")
        
        # Export results
        detector.export_results(result, "form_analysis_results.json", "json")
        detector.export_results(result, "form_analysis_results.csv", "csv")
        
    except Exception as e:
        print(f"Error processing form: {e}")