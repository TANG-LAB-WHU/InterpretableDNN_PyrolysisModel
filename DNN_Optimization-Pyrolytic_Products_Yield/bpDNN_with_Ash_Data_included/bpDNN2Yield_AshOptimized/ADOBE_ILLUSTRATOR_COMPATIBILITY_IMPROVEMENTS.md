# Adobe Illustrator Compatibility Improvements for Pyrolysis SHAP Analysis

## Overview
This document describes the improvements made to `shap_analysis.py` to ensure better compatibility with Adobe Illustrator for editing exported vector graphics.

## Changes Made

### 1. Added SVG Format Support
- **Added SVG export** alongside existing PNG and EPS formats
- SVG provides the best compatibility with Adobe Illustrator
- SVG files maintain full vector properties and can be easily edited

### 2. Optimized EPS Parameters
- **Added `ps_fonttype=42`** parameter to use TrueType fonts instead of Type 3 fonts
- **Added `transparent=False`** and `edgecolor='none'`** for better rendering
- **Added `orientation='portrait'`** and `papertype='letter'`** for consistent formatting

### 3. Enhanced Function Documentation
Modified `save_plot_multi_format()` function with:
- Better error handling for both SVG and EPS formats
- Fallback options if advanced parameters fail
- Clearer parameter documentation

### 4. Updated Documentation
- Updated all README files to mention three formats (PNG, SVG, EPS)
- Added specific Adobe Illustrator guidance
- Updated help text in main function
- Added compatibility notes in script header

## Format Recommendations

### For Adobe Illustrator Users:
1. **Primary Choice: SVG files** - Best compatibility and editing capability
2. **Alternative: EPS files** - Now optimized with TrueType fonts for better compatibility
3. **Avoid: PNG files** - Raster format, not suitable for vector editing

### Technical Details:
- **SVG**: Full vector format with excellent AI compatibility
- **EPS**: Vector format with TrueType font embedding (ps_fonttype=42)
- **PNG**: High-resolution raster format for viewing/web use

## File Naming Convention
All plots are now saved in three formats:
- `filename.png` - Raster format
- `filename.svg` - Vector format (best for Adobe Illustrator)
- `filename.eps` - Vector format (publication-ready)

## Target Variables
This script analyzes three pyrolysis product yields:
- **Char** - Solid carbonaceous residue
- **Liquid** - Bio-oil and condensable organics  
- **Gas** - Non-condensable gas products

Each target gets its own subdirectory with complete SHAP analysis in all three formats.

## Testing
The script has been tested and imports successfully with all new dependencies.

## Benefits
1. **Better Adobe Illustrator compatibility** - SVG and optimized EPS formats
2. **No font rendering issues** - TrueType fonts prevent embedding problems
3. **More flexibility** - Three format options for different use cases
4. **Maintained backwards compatibility** - All original functionality preserved
5. **Complete analysis** - All three pyrolysis products analyzed with consistent formatting

## Usage
No changes to the command line interface. Simply run:
```bash
python shap_analysis.py [char|liquid|gas|all] [-d|--debug]
```

The script will automatically generate all three formats for each visualization across all target variables. 