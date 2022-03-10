import pygame
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
sys.path.append("/home/blind/Downloads/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg")
import carla

SECONDS_PER_EPISODE = 15


class PlayGround:
    def __init__(self):
        self.num = 0
        self.im_height = 150
        self.im_width = 150
        self.lidar_range = 100
        self.collision = False
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        # Load layered map for Town 01 with minimum layout plus buildings and parked vehicles
        self.world = self.client.load_world('/Game/Carla/Maps/Town05_Opt', carla.MapLayer.ParkedVehicles)
        # self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.waypoint_list = self.map.generate_waypoints(2.0)
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.model_3.set_attribute('color', '255,0,0')
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=0,y=0,z=220),\
            carla.Rotation(pitch=-90, yaw=0, roll=0)))
        # self.spectator.set_transform(carla.Transform(carla.Location(x=-10,y=30,z=220),\
        #     carla.Rotation(pitch=-90, yaw=0, roll=0)))

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = 0.05
        self.set_synchronous_mode(True)

        self.depth_image = None
        self.lidar_image = None
        self.radar_dist = None
        self.radar_vel = None
        self.dvs_image = None
        self.vehicle = None

        # Initialize pygame
        pygame.init()
        # Create a screen
        self.screen = pygame.display.set_mode((self.im_width,self.im_height))
        pygame.display.set_caption("Testidos")
        icon = pygame.image.load('electric-car.png')
        pygame.display.set_icon(icon)
        self.surface = None
        self.quit = False
        self.clock = pygame.time.Clock()
    
    def set_synchronous_mode(self, synchronous = True):
        """Set whether to use the synchronous mode"""
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
    
    def to_quit(self):
        """Check if pygame window terminated"""
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                return True
        return False
    
    def render(self, image):
        # Fill background color
        self.screen.fill((0,0,0))
        self.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.screen.blit(self.surface, (0,0))

        # Update display/game window
        pygame.display.update()
    
    def destroy(self):
        for actor in self.actors:
            actor.destroy()
        self.vehicle = None

    def reset(self):

        if self.vehicle is not None:
            self.destroy()

        self.collision = []
        self.actors = []
        self.col_hist = []
        self.dvs_data = []
        self.radar_data = []
        self.depth_image = None
        self.lidar_image = None
        self.radar_dist = None
        self.radar_vel = None
        self.dvs_image = None
        self.rgb_image = None

        # self.set_synchronous_mode(False)

        self.vehicle = self.world.spawn_actor(self.model_3, random.choice(self.map.get_spawn_points()))
        self.actors.append(self.vehicle)

        # Instantiate Depth Camera
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.im_width))
        cam_bp.set_attribute('image_size_y', str(self.im_height))
        cam_bp.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=0, z=2.4))
        camera_rgb = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
        self.actors.append(camera_rgb)
        camera_rgb.listen(self.save_rgb_image) 

        # Instantiate Depth Camera
        cam_bp = self.blueprint_library.find('sensor.camera.depth')
        cam_bp.set_attribute('image_size_x', str(self.im_width))
        cam_bp.set_attribute('image_size_y', str(self.im_height))
        cam_bp.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=0, z=2.4))
        camera_depth = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
        self.actors.append(camera_depth)
        camera_depth.listen(self.save_depth_image) 

        # Instantiate DVS Camera
        cam_bp = self.blueprint_library.find('sensor.camera.dvs')
        cam_bp.set_attribute('image_size_x', str(self.im_width))
        cam_bp.set_attribute('image_size_y', str(self.im_height))
        cam_bp.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=0, z=2.4))
        camera_dvs = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
        self.actors.append(camera_dvs)
        camera_dvs.listen(self.save_dvs_image) 

        # Instantiate Radar
        radar_bp = self.blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '60')
        radar_bp.set_attribute('vertical_fov', '30')
        transform = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=5))
        radar = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle)
        self.actors.append(radar)
        radar.listen(self.save_radar_image) 

        # Instantiate LiDar
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', str(self.lidar_range))
        lidar_bp.set_attribute('points_per_second', '250000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        transform = carla.Transform(carla.Location(x=0, z=2.4))
        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.actors.append(lidar)
        lidar.listen(self.save_lidar_image)  
        
        # Instantiate Colision Sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actors.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # self.set_synchronous_mode(True)

        # self.clock.tick()     # pygame tick
        # self.world.wait_for_tick()

        # Wait for camera sensor to come online
        while (self.rgb_image is None) and (self.lidar_image is None):
            print('Waiting for sensors')
            try:
                self.world.tick(0.1)
                # print('Tick success')
            except:
                print("WARNING: tick not received")
        
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


        self.state = np.zeros([12,self.im_height,self.im_width])
        _, _ = self.skipFrames()
        # self.state = np.transpose(self.dvs_image,(2,0,1))

        return self.state
    
    def get_waypoints_in_range(self, dist, plot=False):
        vehicle_location = self.vehicle.get_location()
        waypoints_near = []
        for waypoint in self.waypoint_list:
            waypoint_location = waypoint.transform.location
            if abs(vehicle_location.distance(waypoint_location)) < dist:
                waypoints_near.append(waypoint)
        
        if plot:
            #Save graph of plotted points as bezier.png

            x = [p.transform.location.x for p in self.waypoint_list]
            y = [p.transform.location.y for p in self.waypoint_list]
            plt.scatter(x, y, marker = 'o', c='Blue')
            x = [p.transform.location.x for p in waypoints_near]
            y = [p.transform.location.y for p in waypoints_near]
            plt.scatter(x, y, marker = '*', c='Red')
            plt.savefig("bezier.png")

        return waypoints_near

    def skipFrames(self):
        reward_total = 0
        for i in range(0,3):
            tick_success = False
            count = 0
            while not tick_success:
                try:
                    self.world.tick(0.1)
                    tick_success = True
                    # print('step success')
                except:
                    count += 1
                    print("WARNING: tick not received: ")
                if count > 2 :
                    print('Consecutive tick failure!')
                    break
            self.state[i,:,:] = self.lidar_image
            self.state[3+(i*3):6+(i*3),:,:] = np.transpose(self.rgb_image,(2,0,1))
            reward,done = self._get_reward()
            reward_total += reward
        self.num += 1
        return reward_total, done


    def step(self, action):
        self.clock.tick()     # pygame tick

        if action == 0: # Brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0.5))
        elif action == 1: # Forward
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer= 0))
        elif action == 2: # 45 Left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=-0.5))
        elif action == 3: # 45 Right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=0.5))
        elif action == 4: # Left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=-1))
        elif action == 5: # Right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=1))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # reward,done = self._get_reward()
        reward,done = self.skipFrames()
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        # self.state = np.transpose(self.dvs_image,(2,0,1))
        return self.state, reward, done, None
    
    def _get_reward(self):
        """Calculate the step reward"""
        
        done = False
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        # near_points = self.get_waypoints_in_range(10,plot=False)

        r_way = 0
        # if len(near_points)==0:
        #     done = True
        #     r_way = -1

        r_radar = 0
        if (self.radar_dist is not None) and (self.radar_dist < 1):
            self.radar_dist = None
            r_radar = -1

        r_s = (-0.0016*kmh**2)+(0.16*kmh)-1

        r_con = 0
        if done == False:
            r_con = 1

        r_col = 0
        if self.collision != []:
            print("Collision event occurred")
            done = True
            r_col = -1

        reward = 200*r_col + 1*r_radar + 1*r_s + 1*r_con + 1*r_way

        return reward,done
    
    def save_rgb_image(self, data):
        # print("Received depth image")
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:,:,:3]        # 3 channel image
        array = array[:,:,::-1]
        # self.depth_image = array[:,:,0]
        self.rgb_image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        # self.render(array)

    
    def save_depth_image(self, data):
        # print("Received depth image")
        data.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:,:,:3]        # 3 channel image
        array = array[:,:,::-1]
        self.depth_image = array[:,:,0]
        # self.render(array)

    def save_lidar_image(self, data):
        # print("Received lidar data")
        disp_size = [self.im_width,self.im_height]
        # disp_size = [self.im_height,self.im_width]

        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / (2*self.lidar_range)
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        self.lidar_image = np.swapaxes(lidar_img[:,:,0],0,1)
        # self.render(self.lidar_image)

    def save_radar_image(self, radar_data):
        # print("Received radar data")
        """Get radar data and find the closest detection"""
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))
        min_dist = np.min(points[:,3])
        min_vel = points[np.where(points[:,3] == min_dist)[0][0],0]
        # print("FR; Dist: "+str(min_dist)+"m, relative vel: "+str(min_vel)+"m/s")

        self.radar_dist = min_dist
        self.radar_vel = min_vel
    
    def save_dvs_image(self, image):
        # print("Received dvs image")
        """Convert dvs data to image and plot"""
        dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))
        dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        # Blue is positive, red is negative
        dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
        # self.render(dvs_img)
        # cv2.imshow("", np.array(dvs_img, dtype = np.uint8))
        # cv2.waitKey(1)

        # self.render(dvs_img)

        self.dvs_image = dvs_img

    def collision_data(self, event):
        self.collision.append(event)



if __name__=="__main__":
    env = PlayGround()
    _ = env.reset()

    try:
        # Game loop
        running = True
        while running:
            _,_,done,_ = env.step(1)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running = False
            if done:
                env.reset()
    finally:
        env.set_synchronous_mode(False)
        env.destroy()