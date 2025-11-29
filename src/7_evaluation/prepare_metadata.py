"""
Prepare metadata from JSON to CSV format for ground truth generation.
"""

import json
import pandas as pd
from pathlib import Path

def prepare_metadata_csv():
    """Convert enriched_metadata.json to CSV format"""

    # Load JSON metadata
    json_path = Path("data/metadata/enriched_metadata.json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Extract fields for CSV
    records = []
    for i, doc in enumerate(metadata):
        record = {
            'doc_id': i,
            'title': doc.get('title', ''),
            'source': doc.get('source', ''),
            'url': doc.get('url', ''),
            'word_count': doc.get('word_count', 0),
            'char_count': doc.get('char_count', 0),
            'raw_path': doc.get('raw_path', ''),
            'cleaned_path': doc.get('cleaned_path', ''),
        }

        # Extract entities
        entities = doc.get('entities', {})
        record['locations'] = entities.get('locations', [])
        record['persons'] = entities.get('persons', [])
        record['organizations'] = entities.get('organizations', [])

        # Extract classifications (if available from preprocessing)
        classifications = doc.get('classifications', {})
        record['heritage_types'] = classifications.get('heritage_types', [])
        record['domains'] = classifications.get('domains', [])
        record['time_period'] = classifications.get('time_period', 'unknown')
        record['region'] = classifications.get('region', 'unknown')
        record['tangibility'] = classifications.get('tangibility', 'unknown')

        # Cluster (if available)
        record['cluster'] = doc.get('cluster', -1)

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / "heritage_metadata.csv"
    df.to_csv(output_path, index=False)

    print(f"✓ Metadata CSV created: {output_path}")
    print(f"  Total documents: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return output_path

if __name__ == "__main__":
    prepare_metadata_csv()
