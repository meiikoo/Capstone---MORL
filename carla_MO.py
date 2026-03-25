import carla
import numpy as np
import math
import weakref
import time
from collections import deque

class MOCarlaWrapper:
    """
    Multi-Objective CARLA Wrapper for autonomous driving.
    Returns a reward vector: [efficiency, safety, stability]
    """

    # Vehicle parameters for SSF calculation
    TRACK_WIDTH = 1.8  # meters (T)
    CG_HEIGHT = 0.5    # meters (h)
    GRAVITY = 9.81     # m/s^2 (g)

    # SSF Threshold = T / 2h
    SSF_LIMIT_G = (TRACK_WIDTH / (2 * CG_HEIGHT)) * GRAVITY

    #Lidar Safety Thresholds
    CRITICAL_DISTANCE = 5.0   # meters
    WARNING_DISTANCE = 15.0   # meters
    SAFE_DISTANCE = 30.0      # meters

    # Speed thresholds for efficiency
    MIN_SPEED = 0.0    # m/s
    MAX_SPEED = 35.0   # m/s (126 km/h, typical highway speed)

    def __init__(self, host='localhost', port=2000, town='Town04',
                 episode_length=400, delta_seconds=0.05, action_repeat=4,
                 reload_world=True, sync_mode=True):
        #connect to CARLA server
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)

        # Load or reuse world.
        if reload_world:
            self.world = self.client.load_world(town)
        else:
            self.world = self.client.get_world()
            if town and self.world.get_map().name.split("/")[-1] != town:
                print(f"Warning: current map is {self.world.get_map().name}, expected {town}")

        # Keep previous world settings so we can restore them on close.
        self._prev_settings = self.world.get_settings()
        self._owns_world_settings = sync_mode

        # Set synchronous mode only when this wrapper should own simulation stepping.
        self.delta_seconds = delta_seconds
        if sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.delta_seconds
            self.world.apply_settings(settings)

        #Episode parameters 
        self.episode_length = episode_length # max number of steps before episode ends
        self.current_step = 0 # time passes per step
        self.action_repeat = max(1, int(action_repeat))

        #Actor references
        self.vehicle = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.owns_vehicle = True

        #Sensor Data Storage
        self.lidar_sensor = None
        self.collision_history = deque(maxlen=10)  # Store recent collisions

        #State tracking for derivatives (to help with turning of vehicle)
        self.last_heading = None
        self.last_location = None
        self.crashed = False

        # Spawn blueprint library (catalog of all the things you can spawn in carla like cars or sensors or props)
        self.blueprint_library = self.world.get_blueprint_library()

    def reset(self):
        """Reset the environment and spawn a new vehicle."""
        # Destroy existing actors
        self._destroy_actors()
        
        # Reset state
        self.current_step = 0
        self.last_heading = None
        self.last_location = None
        self.crashed = False
        self.collision_history.clear()
        self.lidar_data = None
        
        # Spawn vehicle
        self._spawn_vehicle()
        
        # Spawn sensors
        self._spawn_lidar()
        self._spawn_collision_sensor()

        # Position camera to see vehicle
        self._follow_vehicle_with_spectator()
        
         # Warmup ticks
        for _ in range(10):
            self.world.tick()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def _find_candidate_vehicle(self, role_names=('hero', 'ego'), allow_any_vehicle=False):
        """Return a suitable vehicle actor or None."""
        role_matches = []
        all_vehicles = list(self.world.get_actors().filter('vehicle.*'))

        for actor in all_vehicles:
            attrs = actor.attributes if hasattr(actor, "attributes") else {}
            role = attrs.get('role_name', '')
            if role in role_names:
                role_matches.append(actor)

        if role_matches:
            return role_matches[0]
        if allow_any_vehicle and all_vehicles:
            return all_vehicles[0]
        return None

    def attach_to_existing_vehicle(
        self,
        role_names=('hero', 'ego'),
        wait_seconds=60.0,
        poll_interval=0.5,
        allow_any_vehicle=True,
        spawn_if_missing=False,
        enable_autopilot_if_spawned=True
    ):
        """
        Attach sensors to an already spawned vehicle controlled elsewhere.
        Useful with manual_control.py so this wrapper only evaluates rewards.
        """
        # Destroy only our own old actors before attaching to external ego.
        self._destroy_actors()
        self.current_step = 0
        self.last_heading = None
        self.last_location = None
        self.crashed = False
        self.collision_history.clear()
        self.lidar_data = None

        print(
            f"Waiting up to {wait_seconds:.0f}s for external vehicle "
            f"(roles: {role_names})..."
        )
        t_start = time.time()
        next_status_log = t_start
        self.vehicle = None
        while self.vehicle is None and (time.time() - t_start) <= wait_seconds:
            self.vehicle = self._find_candidate_vehicle(
                role_names=role_names,
                allow_any_vehicle=allow_any_vehicle
            )
            if self.vehicle is not None:
                break

            now = time.time()
            if now >= next_status_log:
                print("  ...still waiting for a spawned vehicle client.")
                next_status_log = now + 2.0
            time.sleep(max(0.05, poll_interval))

        if self.vehicle is None:
            if spawn_if_missing:
                print("No external vehicle found. Spawning one from this script...")
                self._spawn_vehicle(role_name='ego', enable_autopilot=enable_autopilot_if_spawned)
                self.owns_vehicle = True
                print(
                    f"Spawned local vehicle id={self.vehicle.id}, "
                    f"role_name={self.vehicle.attributes.get('role_name', 'unknown')}, "
                    f"autopilot={enable_autopilot_if_spawned}"
                )
            else:
                raise RuntimeError(
                    "No external vehicle found. Start manual_control.py, spawn/select a vehicle, "
                    "then run this script (or run this script first and start manual_control.py within the wait window)."
                )
        else:
            self.owns_vehicle = False
            print(
                f"Attached to external vehicle id={self.vehicle.id}, "
                f"role_name={self.vehicle.attributes.get('role_name', 'unknown')}"
            )

        self._spawn_lidar()
        self._spawn_collision_sensor()
        self._follow_vehicle_with_spectator()

        # Warm up sensor callbacks.
        if self._owns_world_settings:
            for _ in range(6):
                self.world.tick()
        else:
            for _ in range(6):
                self.world.wait_for_tick(2.0)

        return self._get_observation(), self._get_info()
    
    def _spawn_vehicle(self, role_name='ego', enable_autopilot=False):
        """Spawn a vehicle using map spawn points with clear fallbacks."""
        # Get vehicle blueprint.
        vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        if vehicle_bp.has_attribute('role_name'):
            vehicle_bp.set_attribute('role_name', role_name)
        if vehicle_bp.has_attribute('color'):
            rec = vehicle_bp.get_attribute('color').recommended_values
            if rec:
                vehicle_bp.set_attribute('color', rec[0])

        # Get map spawn points.
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Map has no spawn points.")

        print(f"Found {len(spawn_points)} map spawn points.")

        # Prefer deterministic points first, then random fallbacks.
        ordered_indices = list(range(min(10, len(spawn_points))))
        random_indices = list(np.random.permutation(len(spawn_points)))
        candidate_indices = ordered_indices + [i for i in random_indices if i not in ordered_indices]

        for attempt, idx in enumerate(candidate_indices[:30], start=1):
            spawn_point = spawn_points[idx]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                # Make sure the ego vehicle can be controlled by this script.
                self.vehicle.set_autopilot(enable_autopilot)
                self.vehicle.set_simulate_physics(True)
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                ))
                loc = self.vehicle.get_transform().location
                print(
                    f"Vehicle spawned successfully at attempt {attempt} "
                    f"(spawn_index={idx}, x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})"
                )
                return

        raise RuntimeError("Failed to spawn vehicle after trying 30 spawn points.")

    def _spawn_lidar(self):
        """Spawn a LiDAR sensor and attach it to the vehicle."""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        
        #this is similar to highway-env with 64 arrays
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')

        # spawn ontop of vehicle
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )

        #set up the callback
        weak_self = weakref.ref(self)
        self.lidar_sensor.listen(
            lambda data: MOCarlaWrapper._on_lidar_data(weak_self, data)
        )

    def _spawn_collision_sensor(self):
        """Spawn a collision sensor and attach it to the vehicle."""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )

        # Set up the callback
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: MOCarlaWrapper._on_collision(weak_self, event)
        )

    @staticmethod
    def _on_lidar_data(weak_self, data):
        """Callback for LiDAR data."""
        self = weak_self()
        if not self:
            return
        
        # Convert raw LiDAR data to numpy array
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    
        #calculate distances (x,y,z,intensity)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)

        # Bin into 64 directional rays (similar to highway-env)
        # Calculate angles for each point
        angles = np.arctan2(points[:, 1], points[:, 0])
        
        # Create 64 bins for directions
        bins = np.linspace(-np.pi, np.pi, 65)
        ray_distances = np.zeros(64)

        #for each slice we are finding the closest point 
        for i in range(64):
            # Find points in this angular bin
            mask = (angles >= bins[i]) & (angles < bins[i + 1])
            if np.any(mask):
                # Take minimum distance in this direction
                ray_distances[i] = np.min(distances[mask])
            else:
                # No detection in this direction
                ray_distances[i] = 50.0  # Max range
        
        # Normalize to [0, 1] like highway-env
        self.lidar_data = ray_distances / 50.0

    @staticmethod
    def _on_collision(weak_self, event):
        """Callback for collision events."""
        self = weak_self()
        if not self:
            return
        
        # Mark that a collision has occurred
        self.crashed = True
        self.collision_history.append(event)
    
    def _get_observation(self):
        """Get the current observation."""
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        ego_x = transform.location.x
        ego_y = transform.location.y
        ego_vx = velocity.x
        ego_vy = velocity.y
        ego_heading = math.radians(transform.rotation.yaw)

        # Kinematics: [presence, x, y, vx, vy, cos_h, sin_h]
        # For simplicity, we'll just have ego vehicle (other vehicles can be added)
        kinematics_obs = np.zeros((15, 7), dtype=np.float32)
        
        # Ego vehicle (first row)
        kinematics_obs[0] = [
            1.0,  # presence
            0.0,  # x (relative to self)
            0.0,  # y (relative to self)
            ego_vx,
            ego_vy,
            np.cos(ego_heading),
            np.sin(ego_heading)
        ]
        
        # LIDAR observation
        if self.lidar_data is None:
            lidar_obs = np.ones(64, dtype=np.float32)  # All at max range
        else:
            lidar_obs = self.lidar_data.copy()
        
        return (kinematics_obs, lidar_obs)

    def _follow_vehicle_with_spectator(self):
        """Position spectator camera to follow the vehicle."""
        if self.vehicle is not None:
            spectator = self.world.get_spectator()
            transform = self.vehicle.get_transform()
            
            # Position camera behind and above the vehicle
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(x=-10, z=5),  # 10m behind, 5m up
                carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)  # Look at vehicle
            ))
    
    def _get_info(self):
        """Get additional info for debugging."""
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        _, lidar_obs = self._get_observation()
        distances = lidar_obs * 50.0
        min_distance = np.min(distances)
        
        info = {
            'crashed': self.crashed,
            'collision_count': len(self.collision_history),
            'speed': speed,
            'lateral_accel': 0.0,
            'rollover_risk': 0.0,
            'min_lidar_distance': min_distance
        }
        return info

    def _calculate_rollover_risk(self):
        """Estimate rollover risk from lateral acceleration and SSF."""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_heading = math.radians(transform.rotation.yaw)

        if self.last_heading is None:
            yaw_rate = 0.0
        else:
            # Wrap heading difference to [-pi, pi] for stable yaw-rate estimation.
            delta_heading = current_heading - self.last_heading
            delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = delta_heading / max(self.delta_seconds, 1e-6)

        lateral_accel = abs(speed * yaw_rate)
        risk = np.clip(lateral_accel / self.SSF_LIMIT_G, 0.0, 1.0)

        self.last_heading = current_heading
        self.last_location = transform.location

        return risk, lateral_accel
    
    def _calculate_lidar_safety(self, lidar_obs):
        """Calculate safety reward based on LiDAR data."""
        # Convert normalized distances back to actual distances
        distances = lidar_obs * 50.0
        
        # Calculate collision risk based on closest object
        min_distance = np.min(distances)
        
        #non-linear risk increase as the obstacles get cloesr
        if min_distance < self.CRITICAL_DISTANCE:
            collision_risk = 1.0  # High risk
        elif min_distance < self.WARNING_DISTANCE:
            collision_risk = 0.5  # Medium risk
        elif min_distance < self.SAFE_DISTANCE:
            collision_risk = 0.1  # Low risk
        else:
            collision_risk = 0.0  # No risk
        
        return collision_risk, min_distance
    
    def step(self, action):
        """Take a step in the environment."""
        # Parse action
        if len(action) == 2:
            throttle, steer = action
            brake = 0.0 if throttle > 0 else abs(throttle)
            throttle = max(0.0, throttle)
        else:
            throttle, steer, brake = action
        
        # Clip values
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)
        
        # Apply control
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        
        # Tick simulation for a few frames so throttle/brake visibly affects speed.
        for _ in range(self.action_repeat):
            self.world.tick()
        self.current_step += 1
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        kinematics_obs, lidar_obs = obs

        # 1. Efficiency (Speed)
        current_speed = info['speed']
        efficiency_reward = np.clip(
            (current_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED),
            0.0, 1.0
        )
        
        # 2. Safety (Collision Risk)
        collision_risk, min_distance = self._calculate_lidar_safety(lidar_obs)
        
        if self.crashed:
            safety_reward = -1.0
        else:
            safety_reward = 1.0 - collision_risk
        
        # 3. Stability (SSF / Rollover Risk)
        risk, lat_accel = self._calculate_rollover_risk()
        stability_reward = -1.0 * risk
        stability_reward = np.clip(stability_reward, -1.0, 0.0)

        info['lateral_accel'] = float(lat_accel)
        info['rollover_risk'] = float(risk)
        info['min_lidar_distance'] = float(min_distance)
        
        # Combine rewards
        vec_reward = np.array([efficiency_reward, safety_reward, stability_reward], 
                             dtype=np.float32)
        
        # Check termination
        terminated = self.crashed
        truncated = self.current_step >= self.episode_length
        
        return obs, vec_reward, terminated, truncated, info

    def observe_step(self):
        """Advance one frame (without controlling) and compute MO rewards."""
        # If this wrapper spawned the vehicle itself (monitor fallback),
        # apply a light cruise command so the vehicle does not remain idle.
        if self.owns_vehicle and self.vehicle is not None:
            pre_vel = self.vehicle.get_velocity()
            pre_speed = math.sqrt(pre_vel.x**2 + pre_vel.y**2 + pre_vel.z**2)
            throttle_cmd = 0.55 if pre_speed < 6.0 else 0.20
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle_cmd),
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

        if self._owns_world_settings:
            self.world.tick()
        else:
            self.world.wait_for_tick(2.0)
        self.current_step += 1

        obs = self._get_observation()
        info = self._get_info()
        _, lidar_obs = obs

        # Efficiency
        current_speed = info['speed']
        efficiency_reward = np.clip(
            (current_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED),
            0.0, 1.0
        )

        # Safety
        collision_risk, min_distance = self._calculate_lidar_safety(lidar_obs)
        safety_reward = -1.0 if self.crashed else (1.0 - collision_risk)

        # Stability
        risk, lat_accel = self._calculate_rollover_risk()
        stability_reward = np.clip(-1.0 * risk, -1.0, 0.0)

        info['lateral_accel'] = float(lat_accel)
        info['rollover_risk'] = float(risk)
        info['min_lidar_distance'] = float(min_distance)

        vec_reward = np.array([efficiency_reward, safety_reward, stability_reward], dtype=np.float32)
        terminated = self.crashed
        truncated = self.current_step >= self.episode_length
        return obs, vec_reward, terminated, truncated, info
    
    def _destroy_actors(self):
        """Destroy all actors to clean up the environment."""
        actors = []
        
        if self.lidar_sensor is not None:
            actors.append(self.lidar_sensor)
            self.lidar_sensor = None
            
        if self.collision_sensor is not None:
            actors.append(self.collision_sensor)
            self.collision_sensor = None
            
        if self.vehicle is not None and self.owns_vehicle:
            actors.append(self.vehicle)
        self.vehicle = None
        self.owns_vehicle = True
        
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        
        # Tick/wait to ensure cleanup callbacks flush.
        if self._owns_world_settings:
            self.world.tick()
        else:
            self.world.wait_for_tick(2.0)

    def close(self):
        """Clean up the environment."""
        self._destroy_actors()
        
        # Restore previous world settings only if this wrapper changed them.
        if self._owns_world_settings:
            self.world.apply_settings(self._prev_settings)

    def __del__(self):
        try:
            self.close()
        except:
            pass

if __name__ == "__main__":
    # Test the CARLA wrapper
    print("Initializing CARLA Multi-Objective Environment...")
    print("\nChoose control mode:")
    print("  1) Random action smoke test")
    print("  2) Monitor external driving (manual_control.py) with auto-spawn fallback")
    mode = input("Enter 1 or 2 [default 2]: ").strip() or "2"
    if mode not in ("1", "2"):
        print(f"Unknown mode '{mode}', switching to mode 2.")
        mode = "2"

    if mode == "2":
        # In monitor mode, do NOT reload map and do NOT force sync.
        env = MOCarlaWrapper(
            town='Town04',
            episode_length=20,
            reload_world=False,
            sync_mode=False
        )
        print(f"SSF Threshold (g): {env.SSF_LIMIT_G:.2f} m/s^2")
        print("\nAttaching to external ego vehicle (or spawning one if missing)...")
        obs, info = env.attach_to_existing_vehicle(
            role_names=('hero', 'ego'),
            wait_seconds=5.0,
            poll_interval=0.5,
            allow_any_vehicle=False,
            spawn_if_missing=True,
            enable_autopilot_if_spawned=False
        )
        kinematics_obs, lidar_obs = obs
        print("Observation shapes:")
        print(f"  Kinematics: {kinematics_obs.shape}")
        print(f"  LIDAR: {lidar_obs.shape}")
        print("\nMonitoring live driving. Press Ctrl+C to stop.")

        i = 0
        try:
            while True:
                i += 1
                obs, vec_reward, terminated, truncated, info = env.observe_step()
                print(
                    f"Step {i:04d} | Reward={vec_reward} | "
                    f"Speed={info['speed']:.2f} m/s | "
                    f"LatAcc={info['lateral_accel']:.2f} m/s^2 | "
                    f"Risk={info['rollover_risk']:.2f} | "
                    f"MinLiDAR={info['min_lidar_distance']:.2f} m"
                )
                if terminated or truncated:
                    break
        except KeyboardInterrupt:
            pass
    else:
        env = MOCarlaWrapper(town='Town04', episode_length=400)
        print(f"SSF Threshold (g): {env.SSF_LIMIT_G:.2f} m/s^2")
        print("\nResetting environment...")

        obs, info = env.reset()
        kinematics_obs, lidar_obs = obs

        print(f"Observation shapes:")
        print(f"  Kinematics: {kinematics_obs.shape}")
        print(f"  LIDAR: {lidar_obs.shape}")

        print("\nRunning random-action test steps...")
        for i in range(20):
            # Sample throttle in [0, 1] so the car actually moves in smoke tests.
            action = np.array([
                np.random.uniform(0.2, 0.8),
                np.random.uniform(-0.35, 0.35)
            ], dtype=np.float32)

            obs, vec_reward, terminated, truncated, info = env.step(action)

            print(f"\nStep {i+1}:")
            print(f"  Action: throttle={action[0]:.2f}, steer={action[1]:.2f}")
            print(f"  Reward: {vec_reward}")
            print(f"  Speed: {info['speed']:.2f} m/s")
            print(f"  Lat Accel: {info['lateral_accel']:.2f} m/s^2")
            print(f"  Rollover Risk: {info['rollover_risk']:.2f}")
            print(f"  Min LIDAR Dist: {info['min_lidar_distance']:.2f} m")

            if terminated:
                print("\n! Episode terminated (crashed)")
                break
            if truncated:
                print("\n! Episode truncated (max steps)")
                break
    
    print("\nClosing environment...")
    env.close()
    print("Done!")