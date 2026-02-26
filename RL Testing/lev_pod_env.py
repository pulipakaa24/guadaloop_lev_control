import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from datetime import datetime
from mag_lev_coil import MagLevCoil
from maglev_predictor import MaglevPredictor

TARGET_GAP = 11.86 / 1000 # target gap height for 9.4 kg pod in meters

class LevPodEnv(gym.Env):
    def __init__(self, use_gui=False, initial_gap_mm=10.0, max_steps=2000, disturbance_force_std=0.0,
                 record_video=False, record_telemetry=False, record_dir="recordings"):
        super(LevPodEnv, self).__init__()

        # Store initial gap height parameter
        self.initial_gap_mm = initial_gap_mm
        self.max_episode_steps = max_steps
        self.current_step = 0

        # Stochastic disturbance parameter (standard deviation of random force in Newtons)
        self.disturbance_force_std = disturbance_force_std

        # Recording parameters
        self.record_video = record_video
        self.record_telemetry = record_telemetry
        self.record_dir = record_dir
        self._frame_skip = 4  # Capture every 4th step → 60fps video from 240Hz sim
        self._video_width = 640
        self._video_height = 480
        self._frames = []
        self._telemetry = {}
        self._recording_active = False
        
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
        # if np.random.rand() > 0.5:
        #     # Spawn exactly at target
        #     spawn_gap_mm = TARGET_GAP * 1000.0
        #     # # Add tiny noise
        #     # spawn_gap_mm += np.random.uniform(-0.5, 0.5)
        # else:
        
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
                            # +Y = Left, -Y = Right (consistent with yoke labeling)
                            label = "Center_Left" if local_pos[1] > 0 else "Center_Right"
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

        # --- Recording setup ---
        # Finalize any previous episode's recording before starting new one
        if self._recording_active:
            self._finalize_recording()

        if self.record_video or self.record_telemetry:
            self._recording_active = True
            os.makedirs(self.record_dir, exist_ok=True)

        if self.record_video:
            self._frames = []
            # Pod body center ≈ start_z, yoke tops at ≈ start_z + 0.091
            # Track bottom at Z=0. Focus camera on the pod body, looking from the side
            # so both the pod and the track bottom (with gap between) are visible.
            pod_center_z = start_z + 0.045  # Approximate visual center of pod
            self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, pod_center_z],
                distance=0.55,
                yaw=50,       # Front-right angle
                pitch=-5,     # Nearly horizontal to see gap from the side
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.client
            )
            self._proj_matrix = p.computeProjectionMatrixFOV(
                fov=35,       # Narrow FOV for less distortion at close range
                aspect=self._video_width / self._video_height,
                nearVal=0.01,
                farVal=2.0,
                physicsClientId=self.client
            )

        if self.record_telemetry:
            self._telemetry = {
                'time': [], 'gap_FL': [], 'gap_FR': [], 'gap_BL': [], 'gap_BR': [],
                'gap_front_avg': [], 'gap_back_avg': [], 'gap_avg': [],
                'roll_deg': [], 'pitch_deg': [],
                'curr_FL': [], 'curr_FR': [], 'curr_BL': [], 'curr_BR': [],
                'force_front': [], 'force_back': [],
                'torque_front': [], 'torque_back': [],
            }

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

        # --- 5b. Apply Stochastic Disturbance Force and Torques (if enabled) ---
        if self.disturbance_force_std > 0:
            disturbance_force = np.random.normal(0, self.disturbance_force_std)
            p.applyExternalForce(
                self.podId, -1,
                forceObj=[0, 0, disturbance_force],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client
            )
            # Roll and pitch disturbance torques, scaled from heave force (torque ~ force * moment_arm)
            # Moment arm ~ 0.15 m so e.g. 1 N force -> 0.15 N·m torque std
            disturbance_torque_std = self.disturbance_force_std * 0.15
            roll_torque = np.random.normal(0, disturbance_torque_std)
            pitch_torque = np.random.normal(0, disturbance_torque_std)
            p.applyExternalTorque(
                self.podId, -1,
                torqueObj=[roll_torque, pitch_torque, 0],
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
        # Goal: Hover at target gap (11.8mm), minimize roll/pitch, minimize power
        target_gap = TARGET_GAP  # 11.8mm in meters
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
        
        # Ground truth info dict (from PyBullet physical state, not sensor observations)
        info = {
            'curr_front_L': curr_front_L,
            'curr_front_R': curr_front_R,
            'curr_back_L': curr_back_L,
            'curr_back_R': curr_back_R,
            'gap_front_yoke': avg_gap_front,
            'gap_back_yoke': avg_gap_back,
            'gap_avg': avg_gap,
            'roll': roll_angle,
            'pitch': pitch_angle,
            'force_front': force_front,
            'force_back': force_back,
            'torque_front': torque_front,
            'torque_back': torque_back
        }

        # --- Recording ---
        if self.record_video and self.current_step % self._frame_skip == 0:
            _, _, rgb, _, _ = p.getCameraImage(
                width=self._video_width,
                height=self._video_height,
                viewMatrix=self._view_matrix,
                projectionMatrix=self._proj_matrix,
                physicsClientId=self.client
            )
            self._frames.append(np.array(rgb, dtype=np.uint8).reshape(
                self._video_height, self._video_width, 4)[:, :, :3])  # RGBA → RGB

        if self.record_telemetry:
            t = self._telemetry
            t['time'].append(self.current_step * self.dt)
            t['gap_FL'].append(front_left_gap * 1000)
            t['gap_FR'].append(front_right_gap * 1000)
            t['gap_BL'].append(back_left_gap * 1000)
            t['gap_BR'].append(back_right_gap * 1000)
            t['gap_front_avg'].append(avg_gap_front * 1000)
            t['gap_back_avg'].append(avg_gap_back * 1000)
            t['gap_avg'].append(avg_gap * 1000)
            t['roll_deg'].append(np.degrees(roll_angle))
            t['pitch_deg'].append(np.degrees(pitch_angle))
            t['curr_FL'].append(curr_front_L)
            t['curr_FR'].append(curr_front_R)
            t['curr_BL'].append(curr_back_L)
            t['curr_BR'].append(curr_back_R)
            t['force_front'].append(force_front)
            t['force_back'].append(force_back)
            t['torque_front'].append(torque_front)
            t['torque_back'].append(torque_back)

        if terminated or truncated:
            self._finalize_recording()

        return obs, reward, terminated, truncated, info

    def apply_impulse(self, force_z: float, position: list = None):
        """
        Apply a one-time impulse force to the pod.

        Args:
            force_z: Vertical force in Newtons (positive = upward)
            position: Local position [x, y, z] to apply force (default: center of mass)
        """
        if position is None:
            position = [0, 0, 0]
        p.applyExternalForce(
            self.podId, -1,
            forceObj=[0, 0, force_z],
            posObj=position,
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )

    def apply_torque_impulse(self, torque_nm: list):
        """
        Apply a one-time impulse torque to the pod (body frame).

        Args:
            torque_nm: [Tx, Ty, Tz] in N·m (LINK_FRAME: X=roll, Y=pitch, Z=yaw)
        """
        p.applyExternalTorque(
            self.podId, -1,
            torqueObj=torque_nm,
            flags=p.LINK_FRAME,
            physicsClientId=self.client
        )

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

    def _finalize_recording(self):
        """Save recorded video and/or telemetry plot to disk."""
        if not self._recording_active:
            return
        self._recording_active = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Save video ---
        if self.record_video and len(self._frames) > 0:
            import imageio.v3 as iio
            video_path = os.path.join(self.record_dir, f"sim_{timestamp}.mp4")
            fps = int(round(1.0 / (self.dt * self._frame_skip)))  # 60fps
            iio.imwrite(video_path, self._frames, fps=fps, codec="h264")
            print(f"Video saved: {video_path} ({len(self._frames)} frames, {fps}fps)")
            self._frames = []

        # --- Save telemetry plot ---
        if self.record_telemetry and len(self._telemetry.get('time', [])) > 0:
            self._save_telemetry_plot(timestamp)

    def _save_telemetry_plot(self, timestamp):
        """Generate and save a 4-panel telemetry figure."""
        import matplotlib.pyplot as plt

        t = {k: np.array(v) for k, v in self._telemetry.items()}
        time = t['time']
        target_mm = TARGET_GAP * 1000
        weight = 9.4 * 9.81  # Pod weight in N

        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Simulation Telemetry  |  gap₀={self.initial_gap_mm}mm  target={target_mm:.1f}mm',
                     fontsize=14, fontweight='bold')

        # --- Panel 1: Gap heights ---
        ax = axes[0]
        ax.plot(time, t['gap_FL'], lw=1, alpha=0.6, label='FL')
        ax.plot(time, t['gap_FR'], lw=1, alpha=0.6, label='FR')
        ax.plot(time, t['gap_BL'], lw=1, alpha=0.6, label='BL')
        ax.plot(time, t['gap_BR'], lw=1, alpha=0.6, label='BR')
        ax.plot(time, t['gap_avg'], lw=2, color='black', label='Average')
        ax.axhline(y=target_mm, color='orange', ls='--', lw=2, label=f'Target ({target_mm:.1f}mm)')
        ax.set_ylabel('Gap Height (mm)')
        ax.legend(loc='best', ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)
        final_err = abs(t['gap_avg'][-1] - target_mm)
        ax.text(0.98, 0.02, f'Final error: {final_err:.3f}mm',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # --- Panel 2: Roll & Pitch ---
        ax = axes[1]
        ax.plot(time, t['roll_deg'], lw=1.5, label='Roll')
        ax.plot(time, t['pitch_deg'], lw=1.5, label='Pitch')
        ax.axhline(y=0, color='gray', ls='--', lw=1)
        ax.set_ylabel('Angle (degrees)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Panel 3: Coil currents ---
        ax = axes[2]
        ax.plot(time, t['curr_FL'], lw=1, alpha=0.8, label='FL')
        ax.plot(time, t['curr_FR'], lw=1, alpha=0.8, label='FR')
        ax.plot(time, t['curr_BL'], lw=1, alpha=0.8, label='BL')
        ax.plot(time, t['curr_BR'], lw=1, alpha=0.8, label='BR')
        total = np.abs(t['curr_FL']) + np.abs(t['curr_FR']) + np.abs(t['curr_BL']) + np.abs(t['curr_BR'])
        ax.plot(time, total, lw=2, color='black', ls='--', label='Total |I|')
        ax.set_ylabel('Current (A)')
        ax.legend(loc='best', ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Panel 4: Forces ---
        ax = axes[3]
        ax.plot(time, t['force_front'], lw=1.5, label='Front yoke')
        ax.plot(time, t['force_back'], lw=1.5, label='Back yoke')
        f_total = t['force_front'] + t['force_back']
        ax.plot(time, f_total, lw=2, color='black', label='Total')
        ax.axhline(y=weight, color='red', ls='--', lw=1.5, label=f'Weight ({weight:.1f}N)')
        ax.set_ylabel('Force (N)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.record_dir, f"sim_{timestamp}_telemetry.png")
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Telemetry plot saved: {plot_path}")
        self._telemetry = {}

    def close(self):
        self._finalize_recording()
        try:
            p.disconnect(physicsClientId=self.client)
        except p.error:
            pass  # Already disconnected

    def render(self):
        """Rendering is handled by PyBullet GUI mode"""
        pass