"""
SolidWorks File Viewer
Attempts to visualize SolidWorks files (.sldprt, .sldasm)
Note: SLDPRT files are proprietary - converting to STEP is recommended
"""

import sys
import os


def convert_sldprt_info():
    """
    Provide information about converting SLDPRT files
    """
    print("=" * 60)
    print("SLDPRT File Format Notice")
    print("=" * 60)
    print("\n.sldprt files are proprietary SolidWorks binary formats.")
    print("\nRECOMMENDED APPROACH:")
    print("1. Open your part in SolidWorks")
    print("2. File → Save As → STEP (*.step, *.stp)")
    print("3. Use step_viewer.py to visualize")
    print("\nAlternatively, you can batch convert using SolidWorks API")
    print("or use free converters like FreeCAD.\n")
    print("=" * 60)


def batch_convert_with_freecad(sldprt_file):
    """
    Attempt to convert SLDPRT to STEP using FreeCAD (if installed)
    
    Args:
        sldprt_file (str): Path to the SLDPRT file
        
    Returns:
        str: Path to the converted STEP file, or None if failed
    """
    try:
        import FreeCAD
        import Import
        
        # Load the SLDPRT file
        print(f"Attempting to load: {sldprt_file}")
        doc = FreeCAD.newDocument()
        Import.insert(sldprt_file, doc.Name)
        
        # Export as STEP
        base_name = os.path.splitext(sldprt_file)[0]
        step_file = f"{base_name}_converted.step"
        
        print(f"Converting to: {step_file}")
        Import.export(doc.Objects, step_file)
        
        FreeCAD.closeDocument(doc.Name)
        print(f"Successfully converted to STEP format!")
        return step_file
        
    except ImportError:
        print("\nFreeCAD not found. Install it with:")
        print("  conda install -c conda-forge freecad")
        print("\nOr download from: https://www.freecad.org/")
        return None
    except Exception as e:
        print(f"\nError during conversion: {e}")
        return None


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python solidworks_viewer.py <path_to_sldprt_file>")
        print("\nExample:")
        print("  python solidworks_viewer.py model.sldprt")
        convert_sldprt_info()
        return
    
    sldprt_file = sys.argv[1]
    
    if not os.path.exists(sldprt_file):
        print(f"Error: File not found: {sldprt_file}")
        sys.exit(1)
    
    # Check file extension
    ext = os.path.splitext(sldprt_file)[1].lower()
    if ext not in ['.sldprt', '.sldasm']:
        print(f"Warning: File extension '{ext}' is not a SolidWorks format")
    
    print("\nAttempting conversion to STEP format...\n")
    
    # Try to convert using FreeCAD
    step_file = batch_convert_with_freecad(sldprt_file)
    
    if step_file and os.path.exists(step_file):
        print("\nWould you like to visualize the converted STEP file?")
        print(f"Run: python PythonSim/step_viewer.py {step_file}")
    else:
        print("\n" + "=" * 60)
        convert_sldprt_info()


if __name__ == "__main__":
    main()
