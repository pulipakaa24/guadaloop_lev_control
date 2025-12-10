"""
URDF Structure Visualizer for Lev Pod using PyBullet
Loads and displays the pod.urdf file in PyBullet's GUI
"""

import pybullet as p
import pybullet_data
import time
import os

# Initialize PyBullet in GUI mode
physicsClient = p.connect(p.GUI)

# Set up the simulation environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

# Configure camera view - looking at inverted maglev system
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=45,
    cameraPitch=1,  # Look up at the hanging pod
    cameraTargetPosition=[0, 0, 0]
)

# Create the maglev track with collision physics (ABOVE, like a monorail)
# The track BOTTOM surface is at Z=0, pod hangs below
track_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[1.0, 0.2, 0.010]  # 2m long × 0.4m wide × 2cm thick
)
track_visual = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[1.0, 0.2, 0.010],
    rgbaColor=[0.3, 0.3, 0.3, 0.8]  # Gray, semi-transparent
)
trackId = p.createMultiBody(
    baseMass=0,  # Static object
    baseCollisionShapeIndex=track_collision,
    baseVisualShapeIndex=track_visual,
    basePosition=[0, 0, 0.010]  # Track bottom at Z=0, center at Z=10mm
)
# Set track surface properties (steel)
p.changeDynamics(trackId, -1, 
                 lateralFriction=0.3,  # Steel-on-steel
                 restitution=0.1)      # Minimal bounce

# Load the lev pod URDF
urdf_path = "pod.xml"
if not os.path.exists(urdf_path):
    print(f"Error: Could not find {urdf_path}")
    print(f"Current directory: {os.getcwd()}")
    p.disconnect()
    exit(1)

# INVERTED SYSTEM: Pod hangs BELOW track
# Track bottom is at Z=0
# Gap height = distance from track bottom DOWN to magnetic yoke (top of pod body)
# URDF collision spheres at +25mm from center = bolts that contact track from below
# URDF box top at +25mm from center = magnetic yoke
# 
# For 10mm gap: 
#   - Yoke (top of pod at +25mm from center) should be at Z = 0 - 10mm = -10mm
#   - Therefore: pod center = -10mm - 25mm = -35mm
#   - Bolts (at +25mm from center) end up at: -35mm + 25mm = -10mm (touching track at Z=0)
start_pos = [0, 0, -0.10085]  # Pod center 100.85mm BELOW track → yoke at 10mm gap
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
podId = p.loadURDF(urdf_path, start_pos, start_orientation)

print("\n" + "=" * 60)
print("PyBullet URDF Visualizer")
print("=" * 60)
print(f"Loaded: {urdf_path}")
print(f"Position: {start_pos}")
print("\nControls:")
print("  • Mouse: Rotate view (left drag), Pan (right drag), Zoom (scroll)")
print("  • Ctrl+Mouse: Apply forces to the pod")
print("  • Press ESC or close window to exit")
print("=" * 60)

# Get and display URDF information
num_joints = p.getNumJoints(podId)
print(f"\nNumber of joints: {num_joints}")

# Get base info
base_mass, base_lateral_friction = p.getDynamicsInfo(podId, -1)[:2]
print(f"\nBase link:")
print(f"  Mass: {base_mass} kg")
print(f"  Lateral friction: {base_lateral_friction}")

# Get collision shape info
collision_shapes = p.getCollisionShapeData(podId, -1)
print(f"\nCollision shapes: {len(collision_shapes)}")
for i, shape in enumerate(collision_shapes):
    shape_type = shape[2]
    dimensions = shape[3]
    local_pos = shape[5]
    shape_names = {
        p.GEOM_BOX: "Box",
        p.GEOM_SPHERE: "Sphere",
        p.GEOM_CAPSULE: "Capsule",
        p.GEOM_CYLINDER: "Cylinder",
        p.GEOM_MESH: "Mesh"
    }
    print(f"  Shape {i}: {shape_names.get(shape_type, 'Unknown')}")
    print(f"    Dimensions: {dimensions}")
    print(f"    Position: {local_pos}")

# Enable visualization of collision shapes
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

# Add coordinate frame visualization at the pod's origin
axis_length = 0.1
p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], lineWidth=3, parentObjectUniqueId=podId, parentLinkIndex=-1)
p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], lineWidth=3, parentObjectUniqueId=podId, parentLinkIndex=-1)
p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], lineWidth=3, parentObjectUniqueId=podId, parentLinkIndex=-1)

print("\n" + "=" * 60)
print("Visualization is running. Interact with the viewer...")
print("Close the PyBullet window to exit.")
print("=" * 60 + "\n")

# Store collision shape local positions and types for tracking
collision_local_positions = []
collision_types = []
for shape in collision_shapes:
    collision_local_positions.append(shape[5])  # Local position (x, y, z)
    collision_types.append(shape[2])  # Shape type (BOX, CYLINDER, etc.)

# Identify the 4 yoke top collision boxes and 4 sensor cylinders
# Both are at Z ≈ 0.08585m, but yokes are BOXES and sensors are CYLINDERS
yoke_indices = []
yoke_labels = []
sensor_indices = []
sensor_labels = []

for i, (local_pos, shape_type) in enumerate(zip(collision_local_positions, collision_types)):
    # Check if at sensor/yoke height (Z ≈ 0.08585m)
    if abs(local_pos[2] - 0.08585) < 0.001:  # Within 1mm tolerance
        if shape_type == p.GEOM_BOX:
            # Yoke collision boxes (BOX shapes)
            yoke_indices.append(i)
            x_pos = "Front" if local_pos[0] > 0 else "Back"
            y_pos = "Right" if local_pos[1] > 0 else "Left"
            yoke_labels.append(f"{x_pos} {y_pos}")
        elif shape_type == p.GEOM_CYLINDER or shape_type == p.GEOM_MESH:
            # Sensor cylinders (may be loaded as MESH by PyBullet)
            # Distinguish from yokes by X or Y position patterns
            # Yokes have both X and Y non-zero, sensors have one coordinate near zero
            if abs(local_pos[0]) < 0.06 or abs(local_pos[1]) < 0.02:
                sensor_indices.append(i)
                # Label sensors by position
                if abs(local_pos[0]) < 0.001:  # X ≈ 0 (center sensors)
                    label = "Center Right" if local_pos[1] > 0 else "Center Left"
                else:
                    label = "Front" if local_pos[0] > 0 else "Back"
                sensor_labels.append(label)

print(f"\nIdentified {len(yoke_indices)} yoke collision boxes for gap height tracking")
print(f"Identified {len(sensor_indices)} sensor cylinders for gap height tracking")

# Run the simulation with position tracking
import numpy as np
step_count = 0
print_interval = 240  # Print every second (at 240 Hz)

try:
    while p.isConnected():
        p.stepSimulation()
        
        # Extract positions periodically
        if step_count % print_interval == 0:
            # Get pod base position and orientation
            pos, orn = p.getBasePositionAndOrientation(podId)
            
            # Convert quaternion to rotation matrix
            rot_matrix = p.getMatrixFromQuaternion(orn)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            
            # Calculate world positions of yoke tops and sensors
            print(f"\n--- Time: {step_count/240:.2f}s ---")
            print(f"Pod center: [{pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}] mm")
            
            print("\nYoke Gap Heights:")
            yoke_gap_heights = []
            for i, yoke_idx in enumerate(yoke_indices):
                local_pos = collision_local_positions[yoke_idx]
                
                # Transform local position to world coordinates
                local_vec = np.array(local_pos)
                world_offset = rot_matrix @ local_vec
                world_pos = np.array(pos) + world_offset
                
                # Add 0.005m (5mm) to get top surface of yoke box (half-height of 10mm box)
                yoke_top_z = world_pos[2] + 0.005
                
                # Gap height: distance from track bottom (Z=0) down to yoke top
                gap_height = 0.0 - yoke_top_z  # Negative means below track
                gap_height_mm = gap_height * 1000
                yoke_gap_heights.append(gap_height_mm)
                
                print(f"  {yoke_labels[i]} yoke: pos=[{world_pos[0]*1000:.1f}, {world_pos[1]*1000:.1f}, {yoke_top_z*1000:.1f}] mm | Gap: {gap_height_mm:.2f} mm")
            
            # Calculate average yoke gap height (for Ansys model input)
            avg_yoke_gap = np.mean(yoke_gap_heights)
            print(f"  Average yoke gap: {avg_yoke_gap:.2f} mm")
            
            print("\nSensor Gap Heights:")
            sensor_gap_heights = []
            for i, sensor_idx in enumerate(sensor_indices):
                local_pos = collision_local_positions[sensor_idx]
                
                # Transform local position to world coordinates
                local_vec = np.array(local_pos)
                world_offset = rot_matrix @ local_vec
                world_pos = np.array(pos) + world_offset
                
                # Add 0.005m (5mm) to get top surface of cylinder (half-length of 10mm cylinder)
                sensor_top_z = world_pos[2] + 0.005
                
                # Gap height: distance from track bottom (Z=0) down to sensor top
                gap_height = 0.0 - sensor_top_z
                gap_height_mm = gap_height * 1000
                sensor_gap_heights.append(gap_height_mm)
                
                print(f"  {sensor_labels[i]} sensor: pos=[{world_pos[0]*1000:.1f}, {world_pos[1]*1000:.1f}, {sensor_top_z*1000:.1f}] mm | Gap: {gap_height_mm:.2f} mm")
            
            # Calculate average sensor gap height
            avg_sensor_gap = np.mean(sensor_gap_heights)
            print(f"  Average sensor gap: {avg_sensor_gap:.2f} mm")
        
        step_count += 1
        time.sleep(1./240.)  # Run at 240 Hz
        
except KeyboardInterrupt:
    print("\nExiting...")

p.disconnect()
print("PyBullet session closed.")
