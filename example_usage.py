#!/usr/bin/env python3
"""
Example usage script for the Advanced Form Detector
Demonstrates practical real-world usage scenarios
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json

from improved_form_detector import AdvancedFormDetector, FormField, FormStructure

def visualize_detections(image_path: str, form_structure: FormStructure, output_path: str = None):
    """Visualize detected form fields on the image"""
    # Load original image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path)
    
    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Define colors for different field types
    colors = {
        'text': 'green',
        'checkbox': 'red', 
        'email': 'blue',
        'phone': 'orange',
        'date': 'purple',
        'currency': 'brown',
        'postal_code': 'pink',
        'ssn': 'gray',
        'credit_card': 'yellow'
    }
    
    # Draw bounding boxes for each detected field
    for i, field in enumerate(form_structure.fields):
        x1, y1, x2, y2 = field.coordinates
        color = colors.get(field.field_type, 'black')
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label = f"{field.field_type} ({field.confidence:.2f})"
        draw.text((x1, y1-20), label, fill=color)
        
        # Draw field number
        draw.text((x1, y1-5), str(i+1), fill=color)
    
    # Save or display result
    if output_path:
        pil_image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        pil_image.show()
    
    return pil_image

def process_single_form(image_path: str, detector: AdvancedFormDetector):
    """Process a single form image"""
    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"{'='*50}")
    
    try:
        # Detect form fields
        result = detector.detect_form_fields(image_path)
        
        # Print results
        print(f"📄 Document Type: {result.document_type}")
        print(f"🎯 Overall Confidence: {result.confidence:.2f}")
        print(f"🔍 Number of fields detected: {len(result.fields)}")
        print(f"⚙️  Processing Method: {result.metadata.get('processing_method', 'unknown')}")
        
        if result.fields:
            print(f"\n📋 Detected Fields:")
            print("-" * 80)
            
            for i, field in enumerate(result.fields, 1):
                print(f"{i:2d}. {field.field_name}")
                print(f"    Type: {field.field_type}")
                print(f"    Value: {field.value or 'N/A'}")
                print(f"    Confidence: {field.confidence:.2f}")
                print(f"    Location: {field.coordinates}")
                print(f"    Method: {field.metadata.get('detection_method', 'unknown')}")
                print()
        
        # Create output directory
        output_dir = Path("form_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate base filename
        base_name = Path(image_path).stem
        
        # Export results
        json_path = output_dir / f"{base_name}_results.json"
        csv_path = output_dir / f"{base_name}_results.csv"
        viz_path = output_dir / f"{base_name}_visualization.jpg"
        
        detector.export_results(result, str(json_path), "json")
        detector.export_results(result, str(csv_path), "csv")
        
        # Create visualization
        visualize_detections(image_path, result, str(viz_path))
        
        print(f"✅ Results exported to:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")
        print(f"   🖼️  Visualization: {viz_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def process_batch_forms(directory_path: str, detector: AdvancedFormDetector):
    """Process multiple form images in a directory"""
    print(f"\n🔄 Batch Processing Forms in: {directory_path}")
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
    
    # Find all form files
    directory = Path(directory_path)
    form_files = []
    
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in supported_formats:
            form_files.append(file_path)
    
    if not form_files:
        print(f"❌ No supported form files found in {directory_path}")
        return
    
    print(f"📁 Found {len(form_files)} form files to process")
    
    # Process each file
    results = []
    for file_path in form_files:
        try:
            if file_path.suffix.lower() == '.pdf':
                # Handle PDF files
                pdf_results = detector.process_pdf_document(str(file_path))
                results.extend(pdf_results)
            else:
                # Handle image files
                result = process_single_form(str(file_path), detector)
                if result:
                    results.append(result)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    # Generate batch summary
    if results:
        generate_batch_summary(results, directory_path)

def generate_batch_summary(results: list, directory_path: str):
    """Generate a summary report for batch processing"""
    output_dir = Path("form_analysis_output")
    summary_path = output_dir / "batch_summary.json"
    
    # Calculate statistics
    total_forms = len(results)
    total_fields = sum(len(result.fields) for result in results)
    avg_confidence = np.mean([result.confidence for result in results]) if results else 0.0
    
    # Count document types
    doc_types = {}
    field_types = {}
    
    for result in results:
        doc_type = result.document_type
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for field in result.fields:
            field_type = field.field_type
            field_types[field_type] = field_types.get(field_type, 0) + 1
    
    # Create summary
    summary = {
        "batch_processing_summary": {
            "directory": directory_path,
            "total_forms_processed": total_forms,
            "total_fields_detected": total_fields,
            "average_confidence": round(avg_confidence, 3),
            "document_types": doc_types,
            "field_types": field_types,
            "processing_statistics": {
                "successful_forms": total_forms,
                "average_fields_per_form": round(total_fields / total_forms, 2) if total_forms > 0 else 0
            }
        }
    }
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"   📁 Directory: {directory_path}")
    print(f"   📄 Forms Processed: {total_forms}")
    print(f"   🔍 Total Fields: {total_fields}")
    print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
    print(f"   📋 Summary saved to: {summary_path}")

def demo_with_sample_data():
    """Demonstrate the system with sample/mock data if no real forms are available"""
    print("\n🎯 Running Demo with Sample Data")
    
    # Create a sample form image programmatically
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw form elements
    # Title
    draw.text((50, 30), "SAMPLE APPLICATION FORM", fill='black')
    
    # Text fields with labels
    fields = [
        ("Name:", 50, 100),
        ("Email:", 50, 150),
        ("Phone:", 50, 200),
        ("Date:", 400, 150),
        ("Address:", 50, 250)
    ]
    
    for label, x, y in fields:
        draw.text((x, y), label, fill='black')
        draw.rectangle([x + 100, y - 5, x + 300, y + 20], outline='black', width=1)
    
    # Checkboxes
    checkboxes = [
        ("☐ Option A", 50, 320),
        ("☐ Option B", 50, 350),
        ("☐ Option C", 50, 380)
    ]
    
    for checkbox, x, y in checkboxes:
        draw.text((x, y), checkbox, fill='black')
    
    # Save sample image
    sample_path = "sample_form_demo.jpg"
    image.save(sample_path)
    
    print(f"✅ Sample form created: {sample_path}")
    
    # Process the sample form
    detector = AdvancedFormDetector(use_gpu=False)  # Use CPU for demo
    result = process_single_form(sample_path, detector)
    
    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)
    
    return result

def main():
    """Main function demonstrating different usage scenarios"""
    print("🚀 Advanced Form Detector - Practical Usage Examples")
    print("=" * 60)
    
    # Initialize detector
    print("⚙️  Initializing Advanced Form Detector...")
    detector = AdvancedFormDetector(use_gpu=True)
    print("✅ Detector initialized successfully!")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isfile(input_path):
            # Process single file
            process_single_form(input_path, detector)
        elif os.path.isdir(input_path):
            # Process directory
            process_batch_forms(input_path, detector)
        else:
            print(f"❌ Invalid path: {input_path}")
    else:
        # Run demo if no arguments provided
        print("\n💡 No input provided, running demo...")
        demo_with_sample_data()
        
        print("\n📖 Usage Instructions:")
        print("   Single file: python example_usage.py path/to/form.jpg")
        print("   Directory:   python example_usage.py path/to/forms/")
        print("   PDF file:    python example_usage.py path/to/document.pdf")

if __name__ == "__main__":
    main()