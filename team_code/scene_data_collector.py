"""
Scene data collector for CARLA agents.

This module provides a reusable class for collecting information about surrounding
objects (vehicles, walkers, traffic lights, etc.) in a CARLA simulation.
It can be used by any agent that has access to the CARLA world and ego vehicle.

NOTE: For more comprehensive bounding box data collection including detailed lane
information, see DataAgent.get_bounding_boxes() in data_agent.py which has more
complete lane traversal logic.
"""

import carla
import numpy as np
from scipy.spatial import KDTree
from webcolors import CSS2_HEX_TO_NAMES, hex_to_rgb

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def convert_rgb_to_names(rgb_tuple):
    """Convert RGB tuple to color name using CSS2 color names."""
    css3_db = CSS2_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def get_relative_transform(ego_matrix, actor_matrix):
    """
    Get the position of an actor relative to the ego vehicle.
    
    Args:
        ego_matrix: 4x4 transformation matrix of the ego vehicle
        actor_matrix: 4x4 transformation matrix of the actor
        
    Returns:
        numpy array [x, y, z] position relative to ego
    """
    actor_pos = actor_matrix[:3, 3]
    ego_pos = ego_matrix[:3, 3]
    
    # Get relative position in world coordinates
    relative_pos_world = actor_pos - ego_pos
    
    # Rotate to ego coordinate system
    ego_rotation_inv = ego_matrix[:3, :3].T
    relative_pos = ego_rotation_inv @ relative_pos_world
    
    return relative_pos


def get_forward_speed(velocity, transform):
    """
    Calculate the forward speed of a vehicle.
    
    Args:
        velocity: carla.Vector3D velocity
        transform: carla.Transform of the vehicle
        
    Returns:
        float: Forward speed in m/s
    """
    velocity_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch_rad = np.deg2rad(transform.rotation.pitch)
    yaw_rad = np.deg2rad(transform.rotation.yaw)
    
    orientation_vector = np.array([
        np.cos(pitch_rad) * np.cos(yaw_rad),
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.sin(pitch_rad)
    ])
    
    return np.dot(velocity_np, orientation_vector)


class SceneDataCollector:
    """
    Collects scene data (surrounding objects) from CARLA simulation.
    
    This class can be used by any CARLA agent to collect information about
    vehicles, walkers, traffic lights, stop signs, and other objects in the scene.
    """
    
    def __init__(self, ego_vehicle, world=None, world_map=None, bb_save_radius=50.0):
        """
        Initialize the scene data collector.
        
        Args:
            ego_vehicle: The ego vehicle actor (carla.Vehicle)
            world: CARLA world object (optional, will use CarlaDataProvider if not provided)
            world_map: CARLA map object (optional, will use CarlaDataProvider if not provided)
            bb_save_radius: Radius in meters to collect objects around the ego vehicle
        """
        self.ego_vehicle = ego_vehicle
        self._world = world if world is not None else CarlaDataProvider.get_world()
        self.world_map = world_map if world_map is not None else CarlaDataProvider.get_map()
        self.bb_save_radius = bb_save_radius
    
    def get_scene_data(self, include_ego=True, include_vehicles=True, include_walkers=True,
                       include_traffic_lights=True, include_stop_signs=True, 
                       include_weather=True, include_ego_info=True, lidar=None):
        """
        Collect comprehensive scene data.
        
        Args:
            include_ego: Include ego vehicle information
            include_vehicles: Include other vehicles
            include_walkers: Include pedestrians
            include_traffic_lights: Include traffic lights
            include_stop_signs: Include stop signs
            include_weather: Include weather information
            include_ego_info: Include detailed ego lane/road information
            lidar: Optional LiDAR point cloud for computing point counts in bounding boxes
            
        Returns:
            list: List of dictionaries containing object information
        """
        results = []
        
        # Get ego vehicle information
        ego_transform = self.ego_vehicle.get_transform()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self.ego_vehicle.bounding_box.extent
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_speed = get_forward_speed(ego_velocity, ego_transform)
        ego_location = ego_transform.location
        
        # Get ego waypoint for lane information
        ego_wp = self.world_map.get_waypoint(
            ego_location, 
            project_to_road=True, 
            lane_type=carla.libcarla.LaneType.Any
        )
        
        if include_ego:
            ego_data = self._get_ego_data(ego_transform, ego_extent, ego_speed)
            results.append(ego_data)
        
        # Get all actors
        actors = self._world.get_actors()
        
        if include_vehicles:
            vehicles = self._get_vehicles_data(
                actors.filter('*vehicle*'), 
                ego_transform, ego_matrix, ego_yaw, ego_wp, lidar
            )
            results.extend(vehicles)
        
        if include_walkers:
            walkers = self._get_walkers_data(
                actors.filter('*walker*'),
                ego_transform, ego_matrix, ego_yaw, lidar
            )
            results.extend(walkers)
        
        if include_traffic_lights:
            traffic_lights = self._get_traffic_lights_data(
                actors.filter('*light*'),
                ego_transform, ego_matrix, ego_yaw
            )
            results.extend(traffic_lights)
        
        if include_stop_signs:
            stop_signs = self._get_stop_signs_data(
                actors.filter('*stop*'),
                ego_transform, ego_matrix, ego_yaw
            )
            results.extend(stop_signs)
        
        if include_weather:
            weather_data = self._get_weather_data()
            results.append(weather_data)
        
        if include_ego_info:
            ego_info = self._get_ego_lane_info(ego_wp)
            results.append(ego_info)
        
        return results
    
    def _get_ego_data(self, ego_transform, ego_extent, ego_speed):
        """Get ego vehicle basic data."""
        ego_control = self.ego_vehicle.get_control()
        return {
            'class': 'ego_car',
            'extent': [ego_extent.x, ego_extent.y, ego_extent.z],
            'position': [0.0, 0.0, 0.0],  # Ego is always at origin in ego coordinates
            'yaw': 0.0,
            'num_points': -1,
            'distance': 0.0,
            'speed': ego_speed,
            'brake': ego_control.brake,
            'id': int(self.ego_vehicle.id),
            'matrix': ego_transform.get_matrix()
        }
    
    def _get_vehicles_data(self, vehicle_list, ego_transform, ego_matrix, ego_yaw, ego_wp, lidar):
        """Get data for all vehicles in range."""
        results = []
        ego_location = ego_transform.location
        
        for vehicle in vehicle_list:
            if vehicle.id == self.ego_vehicle.id:
                continue
                
            if vehicle.get_location().distance(ego_location) >= self.bb_save_radius:
                continue
            
            vehicle_transform = vehicle.get_transform()
            vehicle_rotation = vehicle_transform.rotation
            vehicle_matrix = np.array(vehicle_transform.get_matrix())
            vehicle_velocity = vehicle.get_velocity()
            vehicle_control = vehicle.get_control()
            vehicle_extent = vehicle.bounding_box.extent
            
            # Get waypoint for lane information
            vehicle_wp = self.world_map.get_waypoint(
                vehicle.get_location(), 
                project_to_road=True, 
                lane_type=carla.libcarla.LaneType.Any
            )
            
            # Compute relative position and yaw
            yaw = np.deg2rad(vehicle_rotation.yaw)
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
            
            # Compute speed
            vehicle_speed = get_forward_speed(vehicle_velocity, vehicle_transform)
            
            # Compute distance
            distance = np.linalg.norm(relative_pos)
            
            # Get vehicle extent as list
            extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
            
            # Count LiDAR points in bounding box
            num_points = -1
            if lidar is not None:
                num_points = self._get_points_in_bbox(relative_pos, relative_yaw, extent_list, lidar)
            
            # Get color
            try:
                rgb = tuple(map(int, vehicle.attributes['color'].split(',')))
                color_name = convert_rgb_to_names(rgb)
            except:
                rgb = None
                color_name = None
            
            # Get light state
            try:
                light_state = vehicle.get_light_state()
                light_state_bin = bin(int(light_state))
                light_state_bin_pos = [i for i, x in enumerate(reversed(light_state_bin)) if x == '1']
                light_state_dec_pos = [2**i for i in light_state_bin_pos]
            except:
                light_state_dec_pos = []
            
            # Get traffic light state for vehicle
            tl = self._world.get_traffic_lights_from_waypoint(vehicle_wp, 30.0)
            tl_state = str(tl[0].state) if len(tl) > 0 else 'None'
            
            # Check lane relationship to ego
            same_road_as_ego = vehicle_wp.road_id == ego_wp.road_id
            same_direction_as_ego = False
            lane_relative_to_ego = None
            
            if same_road_as_ego:
                ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id) if ego_wp.lane_id != 0 else 1
                vehicle_lane_direction = vehicle_wp.lane_id / abs(vehicle_wp.lane_id) if vehicle_wp.lane_id != 0 else 1
                same_direction_as_ego = ego_lane_direction == vehicle_lane_direction
                lane_relative_to_ego = vehicle_wp.lane_id - ego_wp.lane_id
            
            result = {
                'class': 'car',
                'color_rgb': rgb,
                'color_name': color_name,
                'road_id': vehicle_wp.road_id,
                'lane_id': vehicle_wp.lane_id,
                'lane_type_str': str(vehicle_wp.lane_type),
                'is_in_junction': vehicle_wp.is_junction,
                'junction_id': vehicle_wp.junction_id,
                'same_road_as_ego': same_road_as_ego,
                'same_direction_as_ego': same_direction_as_ego,
                'lane_relative_to_ego': lane_relative_to_ego,
                'light_state': light_state_dec_pos,
                'traffic_light_state': tl_state,
                'is_at_traffic_light': vehicle.is_at_traffic_light(),
                'base_type': vehicle.attributes.get('base_type', 'unknown'),
                'number_of_wheels': vehicle.attributes.get('number_of_wheels', 'unknown'),
                'extent': extent_list,
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'num_points': int(num_points),
                'distance': distance,
                'speed': vehicle_speed,
                'brake': vehicle_control.brake,
                'steer': vehicle_control.steer,
                'throttle': vehicle_control.throttle,
                'id': int(vehicle.id),
                'role_name': vehicle.attributes.get('role_name', 'unknown'),
                'type_id': vehicle.type_id,
                'matrix': vehicle_transform.get_matrix()
            }
            results.append(result)
        
        return results
    
    def _get_walkers_data(self, walker_list, ego_transform, ego_matrix, ego_yaw, lidar):
        """Get data for all walkers/pedestrians in range."""
        results = []
        ego_location = ego_transform.location
        
        for walker in walker_list:
            if walker.get_location().distance(ego_location) >= self.bb_save_radius:
                continue
            
            walker_transform = walker.get_transform()
            walker_rotation = walker_transform.rotation
            walker_matrix = np.array(walker_transform.get_matrix())
            walker_velocity = walker.get_velocity()
            walker_extent = walker.bounding_box.extent
            
            # Compute relative position and yaw
            yaw = np.deg2rad(walker_rotation.yaw)
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, walker_matrix)
            
            # Compute speed
            walker_speed = np.sqrt(walker_velocity.x**2 + walker_velocity.y**2 + walker_velocity.z**2)
            
            # Compute distance
            distance = np.linalg.norm(relative_pos)
            
            # Get extent as list
            extent_list = [walker_extent.x, walker_extent.y, walker_extent.z]
            
            # Count LiDAR points
            num_points = -1
            if lidar is not None:
                num_points = self._get_points_in_bbox(relative_pos, relative_yaw, extent_list, lidar)
            
            result = {
                'class': 'walker',
                'extent': extent_list,
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'num_points': int(num_points),
                'distance': distance,
                'speed': walker_speed,
                'id': int(walker.id),
                'type_id': walker.type_id,
                'matrix': walker_transform.get_matrix()
            }
            results.append(result)
        
        return results
    
    def _get_traffic_lights_data(self, traffic_light_list, ego_transform, ego_matrix, ego_yaw):
        """Get data for all traffic lights in range."""
        results = []
        ego_location = ego_transform.location
        
        for traffic_light in traffic_light_list:
            tl_location = traffic_light.get_location()
            if tl_location.distance(ego_location) >= self.bb_save_radius:
                continue
            
            tl_transform = traffic_light.get_transform()
            tl_rotation = tl_transform.rotation
            tl_matrix = np.array(tl_transform.get_matrix())
            
            # Compute relative position and yaw
            yaw = np.deg2rad(tl_rotation.yaw)
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, tl_matrix)
            
            # Compute distance
            distance = np.linalg.norm(relative_pos)
            
            result = {
                'class': 'traffic_light',
                'state': str(traffic_light.state),
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'distance': distance,
                'id': int(traffic_light.id),
                'type_id': traffic_light.type_id,
                'matrix': tl_transform.get_matrix()
            }
            results.append(result)
        
        return results
    
    def _get_stop_signs_data(self, stop_sign_list, ego_transform, ego_matrix, ego_yaw):
        """Get data for all stop signs in range."""
        results = []
        ego_location = ego_transform.location
        
        for stop_sign in stop_sign_list:
            ss_location = stop_sign.get_location()
            if ss_location.distance(ego_location) >= self.bb_save_radius:
                continue
            
            ss_transform = stop_sign.get_transform()
            ss_rotation = ss_transform.rotation
            ss_matrix = np.array(ss_transform.get_matrix())
            
            # Compute relative position and yaw
            yaw = np.deg2rad(ss_rotation.yaw)
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, ss_matrix)
            
            # Compute distance
            distance = np.linalg.norm(relative_pos)
            
            result = {
                'class': 'stop_sign',
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'distance': distance,
                'id': int(stop_sign.id),
                'type_id': stop_sign.type_id,
                'matrix': ss_transform.get_matrix()
            }
            results.append(result)
        
        return results
    
    def _get_weather_data(self):
        """Get current weather information."""
        weather = self._world.get_weather()
        return {
            'class': 'weather',
            'cloudiness': weather.cloudiness,
            'dust_storm': weather.dust_storm,
            'fog_density': weather.fog_density,
            'fog_distance': weather.fog_distance,
            'fog_falloff': weather.fog_falloff,
            'mie_scattering_scale': weather.mie_scattering_scale,
            'precipitation': weather.precipitation,
            'precipitation_deposits': weather.precipitation_deposits,
            'rayleigh_scattering_scale': weather.rayleigh_scattering_scale,
            'scattering_intensity': weather.scattering_intensity,
            'sun_altitude_angle': weather.sun_altitude_angle,
            'sun_azimuth_angle': weather.sun_azimuth_angle,
            'wetness': weather.wetness,
            'wind_intensity': weather.wind_intensity,
        }
    
    def _wps_next_until_lane_end(self, wp):
        """
        Get all waypoints from current position until the end of the current lane.
        Adapted from data_agent.py.
        """
        try:
            road_id_cur = wp.road_id
            lane_id_cur = wp.lane_id
            road_id_next = road_id_cur
            lane_id_next = lane_id_cur
            curr_wp = [wp]
            next_wps = []
            # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
            while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
                next_wp = curr_wp[0].next(1)
                if len(next_wp) == 0:
                    break
                curr_wp = next_wp
                next_wps.append(next_wp[0])
                road_id_next = next_wp[0].road_id
                lane_id_next = next_wp[0].lane_id
        except:
            next_wps = []
        return next_wps

    def _get_ego_lane_info(self, ego_wp):
        """
        Get detailed lane/road information for the ego vehicle.
        """
        # Get traffic light state
        tl = self._world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
        tl_state = str(tl[0].state) if len(tl) > 0 else 'None'
        
        # Compute distance to next junction
        next_wps = self._wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
        elif len(next_lane_wps_ego) > 0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
        else:
            distance_to_junction_ego = None
        
        # Compute next road IDs
        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for i, wp in enumerate(next_lane_wps_ego):
            next_road_ids_ego.append(wp.road_id)
            next_next_wps = self._wps_next_until_lane_end(wp)
            try:
                next_next_lane_wps_ego = next_next_wps[-1].next(1)
                if len(next_next_lane_wps_ego) == 0:
                    next_next_lane_wps_ego = [next_next_wps[-1]]
            except:
                next_next_lane_wps_ego = []
            for j, wp2 in enumerate(next_next_lane_wps_ego):
                if wp2.road_id not in next_next_road_ids_ego:
                    next_next_road_ids_ego.append(wp2.road_id)
        
        # Next junction info
        try:
            next_is_junction = next_lane_wps_ego[0].is_junction
            next_junction_id = next_lane_wps_ego[0].junction_id
        except:
            next_is_junction = None
            next_junction_id = None
        
        # Lane counting and lane info using is_opposite flag to avoid infinite loops
        ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id) if ego_wp.lane_id != 0 else 1
        lanes_to_the_left = []
        lanes_to_the_right = []
        num_lanes_same_direction = 1  # ego lane
        lane_ids_same_direction = [ego_wp.lane_id]
        lane_id_left_most_lane_same_direction = ego_wp.lane_id
        num_lanes_opposite_direction = 0
        shoulder_left = False
        shoulder_right = False
        parking_left = False
        parking_right = False
        sidewalk_left = False
        sidewalk_right = False
        bikelane_left = False
        bikelane_right = False
        
        # Traverse left (i=0) and right (i=1)
        for i, lanes in enumerate([lanes_to_the_left, lanes_to_the_right]):
            lane_wp = ego_wp
            is_road = True
            # is_opposite is needed because get_left_lane() returns the left lane 
            # from the viewpoint of that lane. When in an oncoming lane, "left" 
            # would point back to where we came from, causing an infinite loop.
            is_opposite = False
            max_iterations = 20  # Safety limit
            iterations = 0
            
            while is_road and iterations < max_iterations:
                iterations += 1
                
                # Get next lane based on direction and whether we're in opposite lanes
                if i == 0:  # Going left
                    lane_wp = lane_wp.get_left_lane() if not is_opposite else lane_wp.get_right_lane()
                else:  # Going right
                    lane_wp = lane_wp.get_right_lane() if not is_opposite else lane_wp.get_left_lane()
                
                if lane_wp is None:
                    is_road = False
                else:
                    direction = lane_wp.lane_id / abs(lane_wp.lane_id) if lane_wp.lane_id != 0 else 1
                    lane_type = lane_wp.lane_type
                    
                    if lane_type == carla.LaneType.Driving and direction == ego_lane_direction:
                        num_lanes_same_direction += 1
                        lane_ids_same_direction.append(lane_wp.lane_id)
                        if i == 0:
                            lane_id_left_most_lane_same_direction = lane_wp.lane_id
                    elif lane_type == carla.LaneType.Driving and direction != ego_lane_direction:
                        num_lanes_opposite_direction += 1
                    elif lane_type == carla.LaneType.Shoulder and i == 0 and lane_wp.lane_width > 1.0:
                        shoulder_left = True
                    elif lane_type == carla.LaneType.Shoulder and i == 1 and lane_wp.lane_width > 1.0:
                        shoulder_right = True
                    elif lane_type == carla.LaneType.Parking and i == 0:
                        parking_left = True
                    elif lane_type == carla.LaneType.Parking and i == 1:
                        parking_right = True
                    elif lane_type == carla.LaneType.Sidewalk and i == 0:
                        sidewalk_left = True
                    elif lane_type == carla.LaneType.Sidewalk and i == 1:
                        sidewalk_right = True
                    elif lane_type == carla.LaneType.Biking and i == 0:
                        bikelane_left = True
                    elif lane_type == carla.LaneType.Biking and i == 1:
                        bikelane_right = True
                    
                    # When we cross to opposite direction, flip traversal direction
                    if direction != ego_lane_direction:
                        is_opposite = True
                    
                    lanes.append(lane_wp)
        
        # Compute ego lane number (counted from left to right)
        ego_lane_number = abs(ego_wp.lane_id - lane_id_left_most_lane_same_direction)
        
        # Build lane info dicts
        ego_lane = {
            'type:': str(ego_wp.lane_type),
            'width': ego_wp.lane_width,
        }
        left_lanes = [
            {
                'type:': str(lane_wp.lane_type),
                'width': lane_wp.lane_width,
            } for lane_wp in lanes_to_the_left
        ]
        right_lanes = [
            {
                'type:': str(lane_wp.lane_type),
                'width': lane_wp.lane_width,
            } for lane_wp in lanes_to_the_right
        ]
        
        return {
            'class': 'ego_info',
            'traffic_light_state': tl_state,
            'distance_to_junction': distance_to_junction_ego,
            'ego_lane_number': ego_lane_number,
            'road_id': ego_wp.road_id,
            'lane_id': ego_wp.lane_id,
            'is_in_junction': ego_wp.is_junction,
            'is_intersection': ego_wp.is_intersection,
            'junction_id': ego_wp.junction_id,
            'next_road_junction': next_is_junction,
            'next_junction_id': next_junction_id,
            'next_road_ids': next_road_ids_ego,
            'next_next_road_ids_ego': next_next_road_ids_ego,
            'ego_lane': ego_lane,
            'left_lanes': left_lanes,
            'right_lanes': right_lanes,
            'num_lanes_same_direction': num_lanes_same_direction,
            'num_lanes_opposite_direction': num_lanes_opposite_direction,
            'lane_change': ego_wp.lane_change,
            'lane_change_str': str(ego_wp.lane_change),
            'lane_type': ego_wp.lane_type,
            'lane_type_str': str(ego_wp.lane_type),
            'left_lane_marking_color': ego_wp.left_lane_marking.color,
            'left_lane_marking_color_str': str(ego_wp.left_lane_marking.color),
            'left_lane_marking_type': ego_wp.left_lane_marking.type,
            'left_lane_marking_type_str': str(ego_wp.left_lane_marking.type),
            'right_lane_marking_color': ego_wp.right_lane_marking.color,
            'right_lane_marking_color_str': str(ego_wp.right_lane_marking.color),
            'right_lane_marking_type': ego_wp.right_lane_marking.type,
            'right_lane_marking_type_str': str(ego_wp.right_lane_marking.type),
            'shoulder_left': shoulder_left,
            'shoulder_right': shoulder_right,
            'parking_left': parking_left,
            'parking_right': parking_right,
            'sidewalk_left': sidewalk_left,
            'sidewalk_right': sidewalk_right,
            'bike_lane_left': bikelane_left,
            'bike_lane_right': bikelane_right,
        }
    
    def _get_points_in_bbox(self, position, yaw, extent, lidar):
        """
        Count LiDAR points within a bounding box.
        
        Args:
            position: [x, y, z] position relative to ego
            yaw: Relative yaw angle
            extent: [x, y, z] half-extents of bounding box
            lidar: Nx3 numpy array of LiDAR points
            
        Returns:
            int: Number of points within the bounding box
        """
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Transform LiDAR points to object coordinate system
        object_lidar = (rotation_matrix.T @ (lidar - position).T).T
        
        # Check points in bbox
        x, y, z = extent[0], extent[1], extent[2]
        num_points = (
            (object_lidar[:, 0] < x) & (object_lidar[:, 0] > -x) &
            (object_lidar[:, 1] < y) & (object_lidar[:, 1] > -y) &
            (object_lidar[:, 2] < z) & (object_lidar[:, 2] > -z)
        ).sum()
        
        return num_points
