"""
Run validation and create final ground truth dataset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from validation_workflow import AnnotationValidator

# Initialize validator
validator = AnnotationValidator(
    annotation_file='data/evaluation/annotation_interface_v2.json',
    min_agreement=0.6
)

# Run validation and create ground truth
validated_queries, reports = validator.validate_and_create_ground_truth(
    annotator_files=[
        'data/evaluation/annotations_demo_annotator_1.json',
        'data/evaluation/annotations_demo_annotator_2.json'
    ],
    output_version="2.0"
)

print("\n✅ GROUND TRUTH V2.0 GENERATION COMPLETE")
print(f"\n📊 Final Statistics:")
print(f"  - Validated queries: {len(validated_queries)}")
print(f"  - Overall Cohen's Kappa: {reports['agreement_report']['overall_kappa']:.3f}")
print(f"  - Low agreement queries: {len(reports['low_agreement_queries'])}")
print(f"  - Bias recommendations: {len(reports['bias_report']['recommendations'])}")
