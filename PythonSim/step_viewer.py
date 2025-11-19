"""
STEP File Viewer for SolidWorks exports
Visualize 3D CAD models in an interactive 3D environment
"""

import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
import sys


def load_step_file(filename):
    """
    Load a STEP file and return the shape
    
    Args:
        filename (str): Path to the STEP file
        
    Returns:
        TopoDS_Shape: The loaded 3D shape
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    
    if status != IFSelect_RetDone:
        raise Exception(f"Error reading STEP file: {filename}")
    
    # Transfer the contents of the STEP file to the shape
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    print(f"Successfully loaded: {filename}")
    return shape


def visualize_step(filename, background_color=(0.1, 0.1, 0.1), show_origin=True, origin_size=5.0):
    """
    Visualize a STEP file in an interactive 3D viewer
    
    Args:
        filename (str): Path to the STEP file
        background_color (tuple): RGB values (0-1) for background color
        show_origin (bool): Whether to show origin marker
        origin_size (float): Size of the origin marker sphere
    """
    # Load the STEP file
    shape = load_step_file(filename)
    
    # Initialize the 3D display
    display, start_display, add_menu, add_function_to_menu = init_display()
    
    # Set background color using Quantity_Color
    bg_color = Quantity_Color(background_color[0], background_color[1], background_color[2], Quantity_TOC_RGB)
    display.View.SetBackgroundColor(bg_color)
    
    # Display the shape
    display.DisplayShape(shape, update=True)
    
    # Add origin marker
    if show_origin:
        origin_point = gp_Pnt(0, 0, 0)
        origin_marker = BRepPrimAPI_MakeSphere(origin_point, origin_size).Shape()
        origin_color = Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB)  # Red
        display.DisplayShape(origin_marker, color=origin_color, update=True)
        print(f"\nOrigin marker (red sphere) displayed at (0, 0, 0) with radius {origin_size}")
    
    # Fit the view to show the entire model
    display.FitAll()
    
    print("\nControls:")
    print("  - Left mouse button: Rotate")
    print("  - Middle mouse button: Pan")
    print("  - Right mouse button: Zoom")
    print("  - F: Fit all")
    print("  - ESC: Exit")
    
    # Start the display loop
    start_display()


def get_mesh_data(shape, linear_deflection=0.1):
    """
    Extract mesh data (vertices and triangles) from a shape
    Useful for custom rendering or analysis
    
    Args:
        shape: TopoDS_Shape object
        linear_deflection (float): Mesh quality (lower = finer mesh)
        
    Returns:
        tuple: (vertices, triangles) as numpy arrays
    """
    # Mesh the shape
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesh.Perform()
    
    vertices = []
    triangles = []
    vertex_index = 0
    
    # Explore all faces
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = face.Location()
        facing = BRep_Tool.Triangulation(face, location)
        
        if facing:
            # Get transformation
            trsf = location.Transformation()
            
            # Extract vertices
            for i in range(1, facing.NbNodes() + 1):
                pnt = facing.Node(i)
                pnt.Transform(trsf)
                vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            
            # Extract triangles
            for i in range(1, facing.NbTriangles() + 1):
                triangle = facing.Triangle(i)
                n1, n2, n3 = triangle.Get()
                triangles.append([
                    vertex_index + n1 - 1,
                    vertex_index + n2 - 1,
                    vertex_index + n3 - 1
                ])
            
            vertex_index += facing.NbNodes()
        
        explorer.Next()
    
    return np.array(vertices), np.array(triangles)


def main():
    """Main entry point for the STEP viewer"""
    if len(sys.argv) < 2:
        print("Usage: python step_viewer.py <path_to_step_file>")
        print("\nExample:")
        print("  python step_viewer.py model.step")
        print("  python step_viewer.py C:\\Models\\assembly.stp")
        return
    
    step_file = sys.argv[1]
    
    try:
        visualize_step(step_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
