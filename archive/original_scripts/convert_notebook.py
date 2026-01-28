#!/usr/bin/env python
"""
Script to convert the Colab notebook to a local Jupyter notebook
Removes pip install commands and updates paths for local execution
"""

import json
import sys

def convert_notebook_for_local():
    # Read the original notebook
    with open('course_notebook.ipynb', 'r') as f:
        notebook = json.load(f)

    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Get the source code
            source = cell['source']

            # Convert source to string if it's a list
            if isinstance(source, list):
                source_str = ''.join(source)
            else:
                source_str = source

            # Remove pip install commands
            lines = source_str.split('\n')
            filtered_lines = []
            for line in lines:
                if not line.strip().startswith('!pip install'):
                    filtered_lines.append(line)

            # Update data file paths to use local files
            new_source = '\n'.join(filtered_lines)

            # Replace GitHub URLs with local paths
            new_source = new_source.replace(
                'url_A = "https://raw.githubusercontent.com/LinkedInLearning/build-with-ai-executing-and-evaluating-hugging-face-models-4077220/main/T_ONTIME_REPORTING11_A.parquet"',
                'url_A = "T_ONTIME_REPORTING11_A.parquet"'
            )
            new_source = new_source.replace(
                'url_B = "https://raw.githubusercontent.com/LinkedInLearning/build-with-ai-executing-and-evaluating-hugging-face-models-4077220/main/T_ONTIME_REPORTING11_B.parquet"',
                'url_B = "T_ONTIME_REPORTING11_B.parquet"'
            )
            new_source = new_source.replace(
                'url = "https://raw.githubusercontent.com/LinkedInLearning/build-with-ai-executing-and-evaluating-hugging-face-models-4077220/main/ARP-NPIAS-2025-2029-AppendixA.xlsx"',
                'url = "ARP-NPIAS-2025-2029-AppendixA.xlsx"'
            )

            # Update model save paths
            new_source = new_source.replace(
                '/content/AutogluonModels/',
                './AutogluonModels/'
            )
            new_source = new_source.replace(
                'predictor.save("path_to_save_model")',
                'predictor.save("./AutogluonModels/flight_cancellation_model")'
            )
            new_source = new_source.replace(
                'saved_model = TabularPredictor.load("/content/AutogluonModels/ag-20251117_043540")',
                'saved_model = TabularPredictor.load("./AutogluonModels/flight_cancellation_model")'
            )

            # Update cell source
            cell['source'] = new_source

    # Save the converted notebook
    with open('course_notebook_local.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print("✅ Notebook converted successfully!")
    print("📁 Saved as: course_notebook_local.ipynb")
    print("\n📝 Changes made:")
    print("  - Removed all pip install commands")
    print("  - Updated data file paths to use local files")
    print("  - Changed model save/load paths to ./AutogluonModels/")
    print("\n🚀 To run the notebook:")
    print("  uv run jupyter notebook course_notebook_local.ipynb")

if __name__ == '__main__':
    convert_notebook_for_local()