import carla
import numpy as np
import math
import weakref
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
                 episode_length=400, delta_seconds=0.05):
        #connect to CARLA server
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)

        #Load world
        self.world = self.client.load_world(town)

        #Set synchronous mode ( means that unlike default carla the server doesnt run freely but instead server freezes and waits for your world.tick() command before advancing time)
        self.delta_seconds = delta_seconds
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)

        #Episode parameters 
        self.episode_length = episode_length # max number of steps before episode ends
        self.current_step = 0 # time passes per step

        #Actor references
        self.vehicle = None
        self.lidar_sensor = None
        self.collision_sensor = None

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
        
        # Tick once to initialize sensors
        self.world.tick()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _spawn_vehicle(self):
        """Spawn a vehicle at a random location."""
        #getting vehicle blueprint
        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]  # Choose a random vehicle
        #get random spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        #spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

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
        self.lidar_data = points  # Store the latest LiDAR data

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

        Kinematics: [presence, x, y, vx, vy, cos_h, sin_h]
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
    
    def _get_info(self):
        """Get additional info for debugging."""
        info = {
            'crashed': self.crashed,
            'collision_count': len(self.collision_history)
        }
        return info
    
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
        
        # Tick simulation
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
        
        # Combine rewards
        vec_reward = np.array([efficiency_reward, safety_reward, stability_reward], 
                             dtype=np.float32)
        
        # Check termination
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
            
        if self.vehicle is not None:
            actors.append(self.vehicle)
            self.vehicle = None
        
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        
        # Tick to ensure cleanup
        self.world.tick()

    def close(self):
        """Clean up the environment."""
        self._destroy_actors()
        
        # Restore asynchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def __del__(self):
        try:
            self.close()
        except:
            pass

if __name__ == "__main__":
    # Test the CARLA wrapper
    print("Initializing CARLA Multi-Objective Environment...")
    env = MOCarlaWrapper(town='Town04', episode_length=400)
    
    print(f"SSF Threshold (g): {env.SSF_LIMIT_G:.2f} m/s^2")
    print("\nResetting environment...")
    
    obs, info = env.reset()
    kinematics_obs, lidar_obs = obs
    
    print(f"Observation shapes:")
    print(f"  Kinematics: {kinematics_obs.shape}")
    print(f"  LIDAR: {lidar_obs.shape}")
    
    print("\nRunning test steps...")
    for i in range(10):
        # Random continuous action [throttle, steer]
        action = np.random.uniform([-0.5, -0.3], [1.0, 0.3])
        
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