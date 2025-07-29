# Liber777revised JP2 Dataset Processor - Usage Guide

## Overview

The Liber777revised JP2 Dataset Processor is a comprehensive tool designed to analyze and process the Liber777revised_jp2 collection of 170 JP2 image files. This tool provides detailed metadata extraction, content analysis, quality assessment, and multiple export formats.

## Features

### 🔍 **Comprehensive Analysis**
- **File Metadata**: Size, dimensions, compression ratios, timestamps
- **Content Classification**: Text, image, mixed content detection
- **Quality Assessment**: Corruption detection, readability scoring
- **Page Analysis**: Sequential numbering, document type classification

### 📊 **Multiple Export Formats**
- **CSV**: Tabular data for spreadsheet analysis
- **Excel**: Multi-sheet workbook with metadata
- **JSON**: Structured data for programmatic access
- **YAML**: Human-readable metadata configuration
- **Text Report**: Summary report with recommendations

### 🎯 **Advanced Features**
- **Image Analysis**: Using PIL and OpenCV for content detection
- **Quality Metrics**: Completeness, consistency, and quality scoring
- **Error Handling**: Graceful handling of corrupted or problematic files
- **Progress Tracking**: Real-time progress bars and logging

## Installation

### Prerequisites
- Python 3.8 or higher
- Access to the Liber777revised_jp2 directory

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `Pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Advanced image analysis
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `tqdm>=4.65.0` - Progress bars
- `PyYAML>=6.0.0` - YAML export

## Usage

### Quick Start

1. **Run the test script** (recommended for first-time users):
```bash
python test_liber777_processor.py
```

2. **Process the full dataset**:
```bash
python liber777_dataset_processor.py
```

3. **Custom processing**:
```bash
python liber777_dataset_processor.py --dataset-path "C:\Users\thoma\Desktop\Liber777revised_jp2" --output-dir "my_analysis" --format all
```

### Command Line Options

```bash
python liber777_dataset_processor.py [OPTIONS]

Options:
  --dataset-path PATH    Path to Liber777revised_jp2 directory
                         [default: C:\Users\thoma\Desktop\Liber777revised_jp2]
  
  --output-dir PATH      Output directory for results
                         [default: liber777_processed]
  
  --format FORMAT        Output format(s): csv, excel, json, yaml, all
                         [default: all]
```

### Programmatic Usage

```python
from liber777_dataset_processor import Liber777DatasetProcessor

# Create processor
processor = Liber777DatasetProcessor(
    dataset_path="C:\\Users\\thoma\\Desktop\\Liber777revised_jp2",
    output_dir="my_analysis"
)

# Process dataset
df = processor.process_full_dataset()

# Save results
saved_files = processor.save_analysis("all")

# Access metadata
metadata = processor.dataset_metadata
print(f"Quality Score: {metadata.quality_score}")
print(f"Total Files: {metadata.total_files}")
```

## Output Files

### Generated Files Structure
```
liber777_processed/
├── liber777_dataset_analysis.csv      # Main analysis data
├── liber777_dataset_analysis.xlsx     # Excel workbook with sheets
├── liber777_dataset_analysis.json     # Structured JSON data
├── liber777_metadata.yaml             # YAML metadata
├── liber777_summary_report.txt        # Human-readable report
└── liber777_processing.log           # Processing log
```

### CSV Output Columns
- `filename`: Original filename
- `filepath`: Full file path
- `file_size_bytes/mb`: File size information
- `page_number`: Extracted page number
- `image_width/height`: Image dimensions
- `content_type`: Text/image/mixed classification
- `quality_score`: Overall quality assessment
- `document_type`: Page classification
- `has_text/images/diagrams`: Content flags
- `processing_errors`: Any errors encountered

### Excel Workbook Sheets
1. **Dataset_Analysis**: Main file analysis data
2. **Metadata**: Dataset-level statistics and metrics

## Analysis Results

### Content Classification
- **Text Pages**: Primarily textual content
- **Image Pages**: Illustrations, diagrams, artwork
- **Mixed Pages**: Combination of text and images
- **Unknown**: Unable to classify content

### Quality Metrics
- **Quality Score**: 0.0-1.0 (higher is better)
- **Completeness**: Percentage of successfully processed files
- **Consistency**: Page numbering sequence integrity

### Document Types
- **cover_preliminary**: First few pages (covers, title pages)
- **table_of_contents**: Index or contents pages
- **content_page**: Main document content

## Troubleshooting

### Common Issues

1. **"No JP2 files found"**
   - Verify the dataset path is correct
   - Check file extensions are `.jp2`

2. **"PIL/Pillow not available"**
   - Install Pillow: `pip install Pillow`
   - Image analysis will be limited without it

3. **"OpenCV not available"**
   - Install OpenCV: `pip install opencv-python`
   - Advanced image analysis will be limited

4. **Memory issues with large datasets**
   - Process in smaller batches
   - Ensure sufficient RAM (recommended: 8GB+)

### Performance Tips

- **Processing Time**: ~2 seconds per file (170 files ≈ 5-6 minutes)
- **Memory Usage**: ~100MB for full dataset processing
- **Storage**: ~50MB for output files

### Error Recovery

The processor handles errors gracefully:
- Corrupted files are marked with `corruption_detected=True`
- Processing errors are logged in `processing_errors` field
- Failed files still generate basic metadata

## Advanced Usage

### Custom Analysis

```python
# Filter by content type
text_pages = df[df['content_type'] == 'text']
image_pages = df[df['content_type'] == 'image']

# Quality analysis
high_quality = df[df['quality_score'] > 0.8]
low_quality = df[df['quality_score'] < 0.5]

# Page sequence analysis
missing_pages = find_missing_pages(df['page_number'])
```

### Integration with Other Tools

The processor output can be used with:
- **Pandas**: For data analysis and manipulation
- **Matplotlib/Seaborn**: For visualization
- **Database systems**: For storage and querying
- **Machine learning**: For content classification training

## Dataset Information

### Liber777revised_jp2 Collection
- **Total Files**: 170 JP2 images
- **File Pattern**: `Liber777revised_XXXX.jp2`
- **Page Range**: 0000-0169 (sequential numbering)
- **Format**: JPEG 2000 (JP2)
- **Typical Size**: 300-500 KB per file
- **Total Size**: ~50-100 MB

### Expected Content
- **Text Pages**: Book content, tables, lists
- **Image Pages**: Illustrations, diagrams, artwork
- **Mixed Pages**: Pages with both text and images
- **Special Pages**: Covers, title pages, indexes

## Support

For issues or questions:
1. Check the processing log (`liber777_processing.log`)
2. Review the summary report for recommendations
3. Verify all dependencies are installed
4. Ensure sufficient disk space and memory

## Version History

- **v1.0**: Initial release with full dataset processing
- Features: Content classification, quality assessment, multiple export formats
- Support: CSV, Excel, JSON, YAML, and text report outputs 