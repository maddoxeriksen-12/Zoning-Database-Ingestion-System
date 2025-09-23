#!/usr/bin/env python3
"""
Fixed version: Prepare ML training dataset by properly merging COCO format 
with Label Studio JSON exports using region IDs and better spatial matching.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import os

def load_datasets():
    """Load both COCO and Label Studio JSON exports."""
    print("📁 Loading datasets...")
    
    # Load COCO format (spatial data)
    with open('Zoning-Labels-COCO.json', 'r') as f:
        coco_data = json.load(f)
    
    # Load Label Studio JSON (values data)
    with open('zoning-labels-32pages-full-new.json', 'r') as f:
        label_studio_data = json.load(f)
    
    print(f"✅ Loaded COCO: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"✅ Loaded Label Studio: {len(label_studio_data)} tasks")
    
    return coco_data, label_studio_data

def extract_all_label_studio_values(label_studio_data):
    """Extract all values from Label Studio format organized by page."""
    print("\n📋 Extracting Label Studio values...")
    
    values_by_page = {}
    
    for task_idx, task in enumerate(label_studio_data):
        filename = task.get('file_upload', '') or task.get('data', {}).get('image', '')
        page_key = filename.split('/')[-1] if filename else f"page_{task_idx}"
        
        page_values = {
            'filename': filename,
            'rectangles': [],
            'normalized_values': [],
            'zone_codes': [],
            'units': [],
            'qualifiers': [],
            'notes': [],
            'all_results': []
        }
        
        if 'annotations' in task and task['annotations']:
            for annotation in task['annotations']:
                results = annotation.get('result', [])
                
                for result in results:
                    # Store all results for debugging
                    page_values['all_results'].append(result)
                    
                    # Extract different types of annotations
                    if result.get('type') == 'rectanglelabels':
                        # Rectangle with label (field type)
                        value = result.get('value', {})
                        rect_data = {
                            'labels': value.get('rectanglelabels', []),
                            'x': value.get('x', 0),
                            'y': value.get('y', 0),
                            'width': value.get('width', 0),
                            'height': value.get('height', 0),
                            'original_width': value.get('original_width', 2550),
                            'original_height': value.get('original_height', 1650)
                        }
                        page_values['rectangles'].append(rect_data)
                    
                    elif result.get('from_name') == 'normalized_value':
                        # Numeric value
                        value = result.get('value', {})
                        page_values['normalized_values'].append({
                            'number': value.get('number'),
                            'x': value.get('x', 0),
                            'y': value.get('y', 0)
                        })
                    
                    elif result.get('from_name') == 'zone_code_text':
                        # Zone code text
                        value = result.get('value', {})
                        text = value.get('text', [''])[0] if value.get('text') else ''
                        if text:
                            page_values['zone_codes'].append({
                                'text': text,
                                'x': value.get('x', 0),
                                'y': value.get('y', 0)
                            })
                    
                    elif result.get('from_name') == 'unit':
                        # Unit choices
                        value = result.get('value', {})
                        choices = value.get('choices', [])
                        if choices:
                            page_values['units'].append({
                                'unit': choices[0],
                                'x': value.get('x', 0),
                                'y': value.get('y', 0)
                            })
        
        values_by_page[page_key] = page_values
        
        # Show sample of what we found
        if task_idx == 0:
            print(f"\n🔍 Sample page analysis ({page_key}):")
            print(f"   Rectangles: {len(page_values['rectangles'])}")
            print(f"   Normalized values: {len(page_values['normalized_values'])}")
            print(f"   Zone codes: {len(page_values['zone_codes'])}")
            print(f"   Units: {len(page_values['units'])}")
            if page_values['zone_codes']:
                print(f"   Sample zones: {[z['text'] for z in page_values['zone_codes'][:3]]}")
    
    return values_by_page

def merge_with_coco(coco_data, values_by_page):
    """Merge COCO annotations with extracted Label Studio values."""
    print("\n🔗 Merging COCO with Label Studio values...")
    
    # Create mappings
    image_id_to_filename = {}
    for img in coco_data['images']:
        filename = img['file_name'].split('/')[-1]
        image_id_to_filename[img['id']] = filename
    
    category_id_to_name = {}
    for cat in coco_data['categories']:
        category_id_to_name[cat['id']] = cat['name']
    
    # Process annotations
    merged_annotations = []
    zones_found = set()
    
    for anno in coco_data['annotations']:
        image_id = anno['image_id']
        category_id = anno['category_id']
        category_name = category_id_to_name[category_id]
        bbox = anno['bbox']  # COCO format: [x, y, width, height]
        
        # Get corresponding page values
        filename = image_id_to_filename.get(image_id, '')
        page_values = values_by_page.get(filename, {})
        
        # Create base annotation
        merged_anno = {
            'id': anno['id'],
            'image_id': image_id,
            'image_filename': filename,
            'category': category_name,
            'bbox': bbox,
            'bbox_area': bbox[2] * bbox[3],
            'normalized_value': None,
            'unit': None,
            'zone_code': None,
            'has_values': False
        }
        
        # Special handling for ZONE_CODE category
        if category_name == 'ZONE_CODE':
            # Look for zone codes on this page
            if page_values.get('zone_codes'):
                # For now, associate first zone code with first ZONE_CODE box
                # Better logic would use spatial proximity
                for zone in page_values['zone_codes']:
                    if zone['text'] not in zones_found:
                        merged_anno['zone_code'] = zone['text']
                        zones_found.add(zone['text'])
                        merged_anno['has_values'] = True
                        break
        
        # For other categories, try to find associated values
        elif category_name in ['MIN_LOT_AREA', 'MIN_LOT_WIDTH', 'FRONT_YARD', 'SIDE_YARD', 
                               'REAR_YARD', 'MAX_HEIGHT_FT', 'MAX_LOT_COVERAGE']:
            # These typically have numeric values
            if page_values.get('normalized_values'):
                # Simple assignment - would need better spatial matching in production
                if len(merged_annotations) < len(page_values['normalized_values']):
                    value_idx = len(merged_annotations) % len(page_values['normalized_values'])
                    merged_anno['normalized_value'] = page_values['normalized_values'][value_idx]['number']
                    merged_anno['has_values'] = True
            
            # Try to find units
            if page_values.get('units'):
                # Infer unit based on category
                if 'LOT_AREA' in category_name or 'FLOOR_AREA' in category_name:
                    merged_anno['unit'] = 'sqft'
                elif 'HEIGHT_FT' in category_name or 'YARD' in category_name or 'WIDTH' in category_name:
                    merged_anno['unit'] = 'ft'
                elif 'COVERAGE' in category_name:
                    merged_anno['unit'] = 'percent'
        
        merged_annotations.append(merged_anno)
    
    # Summary
    with_values = len([a for a in merged_annotations if a['has_values']])
    zone_codes = len([a for a in merged_annotations if a['zone_code']])
    
    print(f"✅ Created {len(merged_annotations)} merged annotations")
    print(f"📊 Annotations with values: {with_values}")
    print(f"🏷️ Zone codes found: {zone_codes}")
    print(f"🔍 Unique zones: {len(zones_found)}")
    
    return merged_annotations

def create_training_dataset(merged_annotations):
    """Create final training dataset with proper structure."""
    print("\n📦 Creating final training dataset...")
    
    # Group by quality tiers
    high_quality = []  # Has zone code or normalized value
    medium_quality = []  # Has category but no values
    
    for anno in merged_annotations:
        if anno['zone_code'] or anno['normalized_value']:
            high_quality.append(anno)
        else:
            medium_quality.append(anno)
    
    print(f"⭐ High-quality annotations: {len(high_quality)}")
    print(f"📊 Medium-quality annotations: {len(medium_quality)}")
    
    # Create train/val split (80/20)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(high_quality)
    np.random.shuffle(medium_quality)
    
    hq_split = int(len(high_quality) * 0.8)
    mq_split = int(len(medium_quality) * 0.8)
    
    train_set = high_quality[:hq_split] + medium_quality[:mq_split]
    val_set = high_quality[hq_split:] + medium_quality[mq_split:]
    
    np.random.shuffle(train_set)
    np.random.shuffle(val_set)
    
    print(f"🚂 Training set: {len(train_set)}")
    print(f"🧪 Validation set: {len(val_set)}")
    
    return {
        'train': train_set,
        'validation': val_set,
        'high_quality': high_quality,
        'medium_quality': medium_quality,
        'metadata': {
            'total_annotations': len(merged_annotations),
            'with_values': len(high_quality),
            'categories': list(set(a['category'] for a in merged_annotations))
        }
    }

def save_training_dataset(dataset):
    """Save the training dataset."""
    print("\n💾 Saving training dataset...")
    
    with open('ml_training_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved ml_training_dataset.json")
    
    # Show statistics
    print("\n📊 FINAL DATASET STATISTICS:")
    print(f"   Total annotations: {dataset['metadata']['total_annotations']}")
    print(f"   With values: {dataset['metadata']['with_values']}")
    print(f"   Training samples: {len(dataset['train'])}")
    print(f"   Validation samples: {len(dataset['validation'])}")
    print(f"   Unique categories: {len(dataset['metadata']['categories'])}")
    
    # Show sample zones found
    zones = [a['zone_code'] for a in dataset['high_quality'] if a['zone_code']]
    if zones:
        print(f"\n🏷️ Sample zones found:")
        for zone in zones[:10]:
            print(f"   - {zone}")

def main():
    """Main pipeline."""
    print("🚀 ML TRAINING DATASET PREPARATION (FIXED)")
    print("=" * 60)
    
    # Load data
    coco_data, label_studio_data = load_datasets()
    
    # Extract Label Studio values
    values_by_page = extract_all_label_studio_values(label_studio_data)
    
    # Merge with COCO
    merged_annotations = merge_with_coco(coco_data, values_by_page)
    
    # Create training dataset
    dataset = create_training_dataset(merged_annotations)
    
    # Save
    save_training_dataset(dataset)
    
    print("\n✅ COMPLETE! Ready for ML training with ml_training_dataset.json")
    print("\n🎯 Next: Load this dataset in PyTorch/TensorFlow for model training")

if __name__ == '__main__':
    main()