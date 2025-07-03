#!/usr/bin/env python3
"""
Simplified Demo of the Advanced Form Detection System
Shows basic concept without requiring heavy dependencies
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

class SimpleFormDetector:
    """Simplified form detector for demonstration purposes"""
    
    def __init__(self):
        # Field detection patterns
        self.field_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|THB)',
            'thai_phone': r'\b0[0-9]{1}-[0-9]{4}-[0-9]{4}\b',
            'thai_id': r'\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b'
        }
    
    def create_sample_form(self, width: int = 800, height: int = 600) -> Image.Image:
        """Create a sample form image for demonstration"""
        # Create white background
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw form title
        draw.text((50, 30), "SAMPLE APPLICATION FORM", fill='black')
        draw.text((50, 50), "Advanced Form Detection Demo", fill='gray')
        
        # Draw form fields with labels and boxes
        fields = [
            ("Name:", 50, 100),
            ("Email:", 50, 150),
            ("Phone:", 50, 200),
            ("Date of Birth:", 400, 150),
            ("Address:", 50, 250),
            ("Salary:", 400, 200),
        ]
        
        for label, x, y in fields:
            draw.text((x, y), label, fill='black')
            draw.rectangle([x + 120, y - 5, x + 350, y + 20], outline='black', width=1)
        
        # Add some sample data
        sample_data = [
            ("John Doe", 170, 100),
            ("john.doe@example.com", 170, 150),
            ("(555) 123-4567", 170, 200),
            ("01/15/1990", 520, 150),
            ("123 Main St, City, State", 170, 250),
            ("$75,000", 520, 200),
        ]
        
        for text, x, y in sample_data:
            draw.text((x, y), text, fill='blue')
        
        # Draw checkboxes
        checkboxes = [
            ("☐ I agree to terms", 50, 320),
            ("☑ Send newsletter", 50, 350),
            ("☐ Contact by phone", 50, 380)
        ]
        
        for checkbox, x, y in checkboxes:
            draw.text((x, y), checkbox, fill='black')
        
        # Draw signature line
        draw.text((50, 450), "Signature:", fill='black')
        draw.line([150, 470, 400, 470], fill='black', width=1)
        
        return image
    
    def detect_form_fields(self, image: Image.Image) -> FormStructure:
        """Simulate form field detection (simplified version)"""
        fields = []
        
        # Convert image to text simulation (normally would use OCR)
        sample_text = """
        John Doe
        john.doe@example.com
        (555) 123-4567
        01/15/1990
        123 Main St, City, State
        $75,000
        I agree to terms
        Send newsletter
        Contact by phone
        """
        
        # Detect patterns in text
        for pattern_name, pattern in self.field_patterns.items():
            matches = re.finditer(pattern, sample_text, re.IGNORECASE)
            
            for i, match in enumerate(matches):
                # Simulate coordinate detection
                x = 170 + (i * 20)
                y = 150 + (i * 50)
                w = len(match.group()) * 8
                h = 20
                
                fields.append(FormField(
                    field_name=f"{pattern_name}_{i}",
                    field_type=pattern_name,
                    coordinates=(x, y, x+w, y+h),
                    value=match.group(),
                    confidence=0.85 + (i * 0.02),
                    metadata={"detection_method": "pattern_matching_demo"}
                ))
        
        # Add some simulated structural fields
        structural_fields = [
            FormField("name_field", "text", (170, 100, 350, 120), "John Doe", 0.9, 
                     {"detection_method": "structural_analysis"}),
            FormField("address_field", "text", (170, 250, 350, 270), "123 Main St, City, State", 0.8,
                     {"detection_method": "structural_analysis"}),
            FormField("checkbox_1", "checkbox", (50, 320, 70, 340), "unchecked", 0.95,
                     {"detection_method": "template_matching"}),
            FormField("checkbox_2", "checkbox", (50, 350, 70, 370), "checked", 0.95,
                     {"detection_method": "template_matching"}),
        ]
        
        fields.extend(structural_fields)
        
        # Classify document type
        doc_type = self._classify_document_type(fields)
        
        # Calculate overall confidence
        avg_confidence = np.mean([f.confidence for f in fields]) if fields else 0.0
        
        return FormStructure(
            fields=fields,
            document_type=doc_type,
            confidence=avg_confidence,
            metadata={"processing_method": "demo_simulation", "total_fields": len(fields)}
        )
    
    def _classify_document_type(self, fields: List[FormField]) -> str:
        """Classify document type based on detected fields"""
        field_types = [f.field_type for f in fields]
        
        if 'email' in field_types and 'phone' in field_types:
            return "application_form"
        elif 'currency' in field_types:
            return "financial_document"
        elif len([f for f in fields if f.field_type == 'checkbox']) > 2:
            return "survey_form"
        else:
            return "generic_form"
    
    def visualize_detections(self, image: Image.Image, form_structure: FormStructure) -> Image.Image:
        """Visualize detected fields on the image"""
        # Create a copy for visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Colors for different field types
        colors = {
            'text': 'green',
            'checkbox': 'red',
            'email': 'blue',
            'phone': 'orange',
            'date': 'purple',
            'currency': 'brown',
            'thai_phone': 'pink',
            'thai_id': 'gray'
        }
        
        # Draw bounding boxes
        for i, field in enumerate(form_structure.fields):
            x1, y1, x2, y2 = field.coordinates
            color = colors.get(field.field_type, 'black')
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{field.field_type} ({field.confidence:.2f})"
            draw.text((x1, y1-15), label, fill=color)
        
        return vis_image
    
    def export_results(self, form_structure: FormStructure, output_path: str):
        """Export results to JSON file"""
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Main demo function"""
    print("🚀 Advanced Form Detection System - Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = SimpleFormDetector()
    print("✅ Detector initialized")
    
    # Create sample form
    print("📄 Creating sample form...")
    form_image = detector.create_sample_form()
    form_image.save("demo_form.png")
    print("✅ Sample form created: demo_form.png")
    
    # Detect form fields
    print("🔍 Detecting form fields...")
    result = detector.detect_form_fields(form_image)
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Document Type: {result.document_type}")
    print(f"   Overall Confidence: {result.confidence:.3f}")
    print(f"   Fields Detected: {len(result.fields)}")
    print(f"   Processing Method: {result.metadata['processing_method']}")
    
    print(f"\n📋 Detected Fields:")
    print("-" * 80)
    for i, field in enumerate(result.fields, 1):
        print(f"{i:2d}. {field.field_name} ({field.field_type})")
        print(f"    Value: {field.value}")
        print(f"    Confidence: {field.confidence:.3f}")
        print(f"    Location: {field.coordinates}")
        print(f"    Method: {field.metadata.get('detection_method', 'unknown')}")
        print()
    
    # Create visualization
    print("🎨 Creating visualization...")
    vis_image = detector.visualize_detections(form_image, result)
    vis_image.save("demo_form_detected.png")
    print("✅ Visualization saved: demo_form_detected.png")
    
    # Export results
    print("💾 Exporting results...")
    detector.export_results(result, "demo_results.json")
    print("✅ Results exported: demo_results.json")
    
    print(f"\n🎯 Demo Summary:")
    print(f"   ✅ Sample form created")
    print(f"   ✅ {len(result.fields)} fields detected")
    print(f"   ✅ Document classified as: {result.document_type}")
    print(f"   ✅ Results exported in JSON format")
    print(f"   ✅ Visualization image created")
    
    print(f"\n📖 Files Created:")
    print(f"   📄 demo_form.png - Original sample form")
    print(f"   🖼️  demo_form_detected.png - Form with detected fields highlighted")
    print(f"   📊 demo_results.json - Detailed detection results")
    
    print(f"\n💡 This demo shows the concept of the improved form detection system.")
    print(f"   The full system would use computer vision and ML models for real detection.")

if __name__ == "__main__":
    main()