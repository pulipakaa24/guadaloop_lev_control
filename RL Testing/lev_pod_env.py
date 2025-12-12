import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from mag_lev_coil import MagLevCoil
from maglev_predictor import MaglevPredictor

TARGET_GAP = 16.491741 / 1000 # target gap height for 5.8 kg pod in meters

class LevPodEnv(gym.Env):
    def __init__(self, use_gui=False, initial_gap_mm=10.0, max_steps=2000):
        super(LevPodEnv, self).__init__()
        
        # Store initial gap height parameter
        self.initial_gap_mm = initial_gap_mm
        self.max_episode_steps = max_steps
        self.current_step = 0
        
        # The following was coded by AI - see [1]
        # --- 1. Define Action & Observation Spaces ---
        # Action: 4 PWM duty cycles between -1 and 1 (4 independent coils)
        # [front_left, front_right, back_left, back_right] corresponding to +Y and -Y ends of each U-yoke
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Observation: 4 normalized noisy sensor gap heights + 4 normalized velocities
        # Gaps normalized by 0.030m, velocities by 0.1 m/s
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
        
        # --- 2. Setup Physics & Actuators ---
        self.dt = 1./240.  # PyBullet default timestep
        self.coil_front_L = MagLevCoil(1.1, 0.0025, 12, 10.2)
        self.coil_front_R = MagLevCoil(1.1, 0.0025, 12, 10.2)
        self.coil_back_L = MagLevCoil(1.1, 0.0025, 12, 10.2)
        self.coil_back_R = MagLevCoil(1.1, 0.0025, 12, 10.2)
        
        # Sensor noise parameters
        self.sensor_noise_std = 0.0001  # 0.1mm standard deviation
        
        # Normalization constants for observations
        self.gap_scale = 0.015  # Normalize gaps by +-15mm max expected deviation from middle
        self.velocity_scale = 0.1  # Normalize velocities by 0.1 m/s max expected velocity
        
        # Maglev force/torque predictor
        self.predictor = MaglevPredictor()
        
        # Connect to PyBullet (DIRECT is faster for training, GUI for debugging)
        self.client = p.connect(p.GUI if use_gui else p.DIRECT) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Store references
        self.trackId = None
        self.podId = None
        self.collision_local_positions = []
        self.yoke_indices = []  # For force application
        self.yoke_labels = []
        self.sensor_indices = []  # For gap height measurement
        self.sensor_labels = []
        
        # For velocity calculation
        self.prev_sensor_gaps = None
        
    def reset(self, seed=None, options=None):
        # Reset PyBullet simulation
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        
        # Create the maglev track (inverted system - track above, pod hangs below)
        # Track bottom surface at Z=0
        track_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[1.0, 0.2, 0.010],
            physicsClientId=self.client
        )
        track_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[1.0, 0.2, 0.010],
            rgbaColor=[0.3, 0.3, 0.3, 0.8],
            physicsClientId=self.client
        )
        self.trackId = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=track_collision,
            baseVisualShapeIndex=track_visual,
            basePosition=[0, 0, 0.010],  # Track center at Z=10mm, bottom at Z=0
            physicsClientId=self.client
        )
        p.changeDynamics(self.trackId, -1, 
                         lateralFriction=0.3,
                         restitution=0.1,
                         physicsClientId=self.client)
        
        urdf_path = self._create_modified_urdf()
        
        # Determine start condition
        if np.random.rand() > 0.5:
            # Spawn exactly at target
            spawn_gap_mm = TARGET_GAP * 1000.0
            # # Add tiny noise
            # spawn_gap_mm += np.random.uniform(-0.5, 0.5)
        else:
            spawn_gap_mm = self.initial_gap_mm
            
        start_z = -(0.09085 + spawn_gap_mm / 1000.0)
        start_pos = [0, 0, start_z]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.podId = p.loadURDF(urdf_path, start_pos, start_orientation, physicsClientId=self.client)
        
        # The following was coded by AI - see [2]
        # Parse collision shapes to identify yokes and sensors
        collision_shapes = p.getCollisionShapeData(self.podId, -1, physicsClientId=self.client)
        self.collision_local_positions = []
        self.yoke_indices = []
        self.yoke_labels = []
        self.sensor_indices = []
        self.sensor_labels = []
        
        # Expected heights for detection
        expected_yoke_sensor_z = 0.08585  # Yokes and sensors always at this height
        expected_bolt_z = 0.08585 + self.initial_gap_mm / 1000.0  # Bolts at gap-dependent height
        
        for i, shape in enumerate(collision_shapes):
            shape_type = shape[2]
            local_pos = shape[5]
            self.collision_local_positions.append(local_pos)
            
            # Check if at sensor/yoke height (Z ≈ 0.08585m) - NOT bolts
            if abs(local_pos[2] - expected_yoke_sensor_z) < 0.001:
                if shape_type == p.GEOM_BOX:
                    # Yokes are BOX type at the four corners (size 0.0254)
                    self.yoke_indices.append(i)
                    x_pos = "Front" if local_pos[0] > 0 else "Back"
                    y_pos = "Left" if local_pos[1] > 0 else "Right"
                    self.yoke_labels.append(f"{x_pos}_{y_pos}")
                elif shape_type == p.GEOM_CYLINDER or shape_type == p.GEOM_MESH:
                    # Sensors: distinguish by position pattern
                    if abs(local_pos[0]) < 0.06 or abs(local_pos[1]) < 0.02:
                        self.sensor_indices.append(i)
                        if abs(local_pos[0]) < 0.001:  # Center sensors (X ≈ 0)
                            label = "Center_Right" if local_pos[1] > 0 else "Center_Left"
                        else:  # Front/back sensors (Y ≈ 0)
                            label = "Front" if local_pos[0] > 0 else "Back"
                        self.sensor_labels.append(label)
        
        self.coil_front_L.current = 0
        self.coil_front_R.current = 0
        self.coil_back_L.current = 0
        self.coil_back_R.current = 0
        
        self.prev_sensor_gaps = None
        obs = self._get_obs(initial_reset=True)
        self.current_step = 0
        
        return obs, {}

    # The following was generated by AI - see [14]
    def step(self, action):
        # Check if PyBullet connection is still active (GUI might be closed)
        try:
            p.getConnectionInfo(physicsClientId=self.client)
        except p.error:
            # Connection lost - GUI was closed
            return self._get_obs(), -100.0, True, True, {'error': 'GUI closed'}
        
        # Update Coil Currents from PWM Actions
        pwm_front_L = action[0]  # yoke +x,+y
        pwm_front_R = action[1]  # yoke +x,-y
        pwm_back_L = action[2]   # yoke -x,+y
        pwm_back_R = action[3]   # yoke -x,-y
        
        curr_front_L = self.coil_front_L.update(pwm_front_L, self.dt)
        curr_front_R = self.coil_front_R.update(pwm_front_R, self.dt)
        curr_back_L = self.coil_back_L.update(pwm_back_L, self.dt)
        curr_back_R = self.coil_back_R.update(pwm_back_R, self.dt)
        
        # --- 2. Get Current Pod State ---
        pos, orn = p.getBasePositionAndOrientation(self.podId, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.podId, physicsClientId=self.client)
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # --- 3. Calculate Gap Heights at Yoke Positions (for force prediction) ---
        # Calculate world positions of the 4 yokes (ends of U-yokes)
        yoke_gap_heights_dict = {}  # Store by label for easy access
        
        for i, yoke_idx in enumerate(self.yoke_indices):
            local_pos = self.collision_local_positions[yoke_idx]
            local_vec = np.array(local_pos)
            world_offset = rot_matrix @ local_vec
            world_pos = np.array(pos) + world_offset
            
            # Top surface of yoke box (add half-height = 5mm)
            yoke_top_z = world_pos[2] + 0.005
            
            # Gap height: track bottom (Z=0) to yoke top (negative Z)
            gap_height = -yoke_top_z  # Convert to positive gap in meters
            yoke_gap_heights_dict[self.yoke_labels[i]] = gap_height
        
        # Average gap heights for each U-shaped yoke (average left and right ends)
        # Front yoke: average of Front_Left and Front_Right
        # Back yoke: average of Back_Left and Back_Right
        avg_gap_front = (yoke_gap_heights_dict.get('Front_Left', 0.010) + 
                        yoke_gap_heights_dict.get('Front_Right', 0.010)) / 2
        avg_gap_back = (yoke_gap_heights_dict.get('Back_Left', 0.010) + 
                       yoke_gap_heights_dict.get('Back_Right', 0.010)) / 2
        
        front_left_gap = yoke_gap_heights_dict.get('Front_Left', 0.010)
        front_right_gap = yoke_gap_heights_dict.get('Front_Right', 0.010)
        back_left_gap = yoke_gap_heights_dict.get('Back_Left', 0.010)
        back_right_gap = yoke_gap_heights_dict.get('Back_Right', 0.010)
        
        # hypotenuses
        y_distance = 0.1016  # 2 * 0.0508m (left to right distance)
        x_distance = 0.2518  # 2 * 0.1259m (front to back distance)
        
        # Roll angle
        # When right side has larger gap, roll is negative
        roll_angle_front = np.arcsin(-(front_right_gap - front_left_gap) / y_distance)
        roll_angle_back = np.arcsin(-(back_right_gap - back_left_gap) / y_distance)
        roll_angle = (roll_angle_front + roll_angle_back) / 2
        
        # When back has larger gap, pitch is positive
        pitch_angle_left = np.arcsin((back_left_gap - front_left_gap) / x_distance)
        pitch_angle_right = np.arcsin((back_right_gap - front_right_gap) / x_distance)
        pitch_angle = (pitch_angle_left + pitch_angle_right) / 2
        
        # Predict Forces and Torques using Maglev Predictor
        # Gap heights in mm
        gap_front_mm = avg_gap_front * 1000
        gap_back_mm = avg_gap_back * 1000
        
        # Roll angle in degrees
        roll_deg = np.degrees(roll_angle)
        
        # Predict force and torque for each U-shaped yoke
        # Front yoke
        force_front, torque_front = self.predictor.predict(
            curr_front_L, curr_front_R, roll_deg, gap_front_mm
        )
        
        # Back yoke
        force_back, torque_back = self.predictor.predict(
            curr_back_L, curr_back_R, roll_deg, gap_back_mm
        )
        
        # --- 5. Apply Forces and Torques to Pod ---
        # Forces are applied at Y=0 (center of U-yoke) at each X position
        # This is where the actual magnetic force acts on the U-shaped yoke
        
        # Apply force at front yoke center (X=+0.1259, Y=0)
        front_yoke_center = [0.1259, 0, 0.08585]  # From pod.xml yoke positions
        p.applyExternalForce(
            self.podId, -1,
            forceObj=[0, 0, force_front],
            posObj=front_yoke_center,
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )
        
        # Apply force at back yoke center (X=-0.1259, Y=0)
        back_yoke_center = [-0.1259, 0, 0.08585]
        p.applyExternalForce(
            self.podId, -1,
            forceObj=[0, 0, force_back],
            posObj=back_yoke_center,
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )

        
        
        # Apply roll torques
        # Each yoke produces its own torque about X axis
        torque_front_Nm = torque_front / 1000  # Convert from mN·m to N·m
        torque_back_Nm = torque_back / 1000
        
        # Apply torques at respective yoke positions
        p.applyExternalTorque(
            self.podId, -1,
            torqueObj=[torque_front_Nm, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )
        p.applyExternalTorque(
            self.podId, -1,
            torqueObj=[torque_back_Nm, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )
        
        # --- 6. Step Simulation ---
        p.stepSimulation(physicsClientId=self.client)
        self.current_step += 1
        
        # Check for physical contact with track (bolts touching)
        contact_points = p.getContactPoints(bodyA=self.podId, bodyB=self.trackId, physicsClientId=self.client)
        has_contact = len(contact_points) > 0
        
        # --- 7. Get New Observation ---
        obs = self._get_obs()
        
        # --- 8. Calculate Reward ---
        # Goal: Hover at target gap (16.5mm), minimize roll/pitch, minimize power
        target_gap = TARGET_GAP  # 16.5mm in meters
        avg_gap = (avg_gap_front + avg_gap_back) / 2
        
        gap_error = abs(avg_gap - target_gap)
        
        # Power dissipation (all 4 coils)
        power = (curr_front_L**2 * self.coil_front_L.R + 
                curr_front_R**2 * self.coil_front_R.R +
                curr_back_L**2 * self.coil_back_L.R +
                curr_back_R**2 * self.coil_back_R.R)
        
        # --- Improved Reward Function ---
        # Use reward shaping with reasonable scales to enable learning
        
        # 1. Gap Error Reward (most important)
        # Use exponential decay for smooth gradient near target
        gap_error_mm = gap_error * 1000  # Convert to mm
        gap_reward = 10.0 * np.exp(-0.5 * (gap_error_mm / 3.0)**2)  # Peak at 0mm error, 3mm std dev
        
        # 2. Orientation Penalties (smaller scale)
        roll_penalty = abs(np.degrees(roll_angle)) * 0.02
        pitch_penalty = abs(np.degrees(pitch_angle)) * 0.02
        
        # 3. Velocity Penalty (discourage rapid oscillations)
        z_velocity = lin_vel[2]
        velocity_penalty = abs(z_velocity) * 0.1
        
        # 4. Contact Penalty
        contact_points = p.getContactPoints(bodyA=self.podId, bodyB=self.trackId)
        contact_penalty = len(contact_points) * 0.2
        
        # 5. Power Penalty (encourage efficiency, but small weight)
        power_penalty = power * 0.001
        
        # Combine rewards (scaled to ~[-5, +1] range per step)
        reward = gap_reward - roll_penalty - pitch_penalty - velocity_penalty - contact_penalty - power_penalty
        
        # Check Termination (tighter bounds for safety)
        terminated = False
        truncated = False
        
        # Terminate if gap is too small (crash) or too large (lost)
        if avg_gap < 0.003 or avg_gap > 0.035:
            terminated = True
            reward = -10.0  # Failure penalty (scaled down)
            
        # Terminate if orientation is too extreme
        if abs(roll_angle) > np.radians(15) or abs(pitch_angle) > np.radians(15):
            terminated = True
            reward = -10.0
            
        # Success bonus for stable hovering near target
        if gap_error_mm < 1.0 and abs(np.degrees(roll_angle)) < 2.0 and abs(np.degrees(pitch_angle)) < 2.0:
            reward += 2.0  # Bonus for excellent control
        
        info = {
            'curr_front_L': curr_front_L,
            'curr_front_R': curr_front_R,
            'curr_back_L': curr_back_L,
            'curr_back_R': curr_back_R,
            'gap_front_yoke': avg_gap_front,
            'gap_back_yoke': avg_gap_back,
            'roll': roll_angle,
            'force_front': force_front,
            'force_back': force_back,
            'torque_front': torque_front,
            'torque_back': torque_back
        }
        
        return obs, reward, terminated, truncated, info

    # The following was generated by AI - see [15]
    def _get_obs(self, initial_reset=False):
        """
        Returns observation: [gaps(4), velocities(4)]
        Uses noisy sensor readings + computed velocities for microcontroller-friendly deployment
        """
        pos, orn = p.getBasePositionAndOrientation(self.podId, physicsClientId=self.client)
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # Calculate sensor gap heights with noise
        sensor_gap_heights = {}
        
        for i, sensor_idx in enumerate(self.sensor_indices):
            local_pos = self.collision_local_positions[sensor_idx]
            local_vec = np.array(local_pos)
            world_offset = rot_matrix @ local_vec
            world_pos = np.array(pos) + world_offset
            
            # Top surface of sensor (add half-height = 5mm)
            sensor_top_z = world_pos[2] + 0.005
            
            # Gap height: track bottom (Z=0) to sensor top
            gap_height = -sensor_top_z
            
            # Add measurement noise
            noisy_gap = gap_height + np.random.normal(0, self.sensor_noise_std)
            # sensor_gap_heights[self.sensor_labels[i]] = noisy_gap
            sensor_gap_heights[self.sensor_labels[i]] = gap_height
        
        # Pack sensor measurements in consistent order
        # [center_right, center_left, front, back]
        gaps = np.array([
            sensor_gap_heights.get('Center_Right', 0.010),
            sensor_gap_heights.get('Center_Left', 0.010),
            sensor_gap_heights.get('Front', 0.010),
            sensor_gap_heights.get('Back', 0.010)
        ], dtype=np.float32)
        
        # Compute velocities (d_gap/dt)
        if initial_reset or (self.prev_sensor_gaps is None):
            # First observation - no velocity information yet
            velocities = np.zeros(4, dtype=np.float32)
        else:
            # Compute velocity as finite difference
            velocities = (gaps - self.prev_sensor_gaps) / self.dt
        
        # Store for next step
        self.prev_sensor_gaps = gaps.copy()
        
        # Normalize observations
        gaps_normalized = (gaps - TARGET_GAP) / self.gap_scale
        velocities_normalized = velocities / self.velocity_scale
        
        # Concatenate: [normalized_gaps, normalized_velocities]
        obs = np.concatenate([gaps_normalized, velocities_normalized])
        
        return obs
    
    # The following was generated by AI - see [16]
    def _create_modified_urdf(self):
        """
        Create a modified URDF with bolt positions adjusted based on initial gap height.
        Bolts are at Z = 0.08585 + gap_mm/1000 (relative to pod origin).
        Yokes and sensors remain at Z = 0.08585 (relative to pod origin).
        """
        import tempfile
        
        # Calculate bolt Z position
        bolt_z = 0.08585 + self.initial_gap_mm / 1000.0
        
        # Read the original URDF template
        urdf_template_path = os.path.join(os.path.dirname(__file__), "pod.xml")
        with open(urdf_template_path, 'r') as f:
            urdf_content = f.read()
        
        # Replace the bolt Z positions (originally at 0.09585)
        # There are 4 bolts at different X,Y positions but same Z
        urdf_modified = urdf_content.replace(
            'xyz="0.285 0.03 0.09585"',
            f'xyz="0.285 0.03 {bolt_z:.6f}"'
        ).replace(
            'xyz="0.285 -0.03 0.09585"',
            f'xyz="0.285 -0.03 {bolt_z:.6f}"'
        ).replace(
            'xyz="-0.285 0.03 0.09585"',
            f'xyz="-0.285 0.03 {bolt_z:.6f}"'
        ).replace(
            'xyz="-0.285 -0.03 0.09585"',
            f'xyz="-0.285 -0.03 {bolt_z:.6f}"'
        )
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_modified)
            temp_urdf_path = f.name
        
        return temp_urdf_path

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except p.error:
            pass  # Already disconnected
    
    def render(self):
        """Rendering is handled by PyBullet GUI mode"""
        pass