#!/usr/bin/env python3

import os
import sys
import glob

import carla

import threading

import queue
import struct
import numpy as np

from modules import ply
from modules.utils import rotation_carla, translation_carla


def sensor_callback(sensor_data, sensor_queue):
    sensor_queue.put(sensor_data)


class Sensor:
    initial_ts = 0.0
    initial_loc = carla.Location()
    initial_rot = carla.Rotation()

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        self.queue = queue.Queue()
        self.bp = self.set_attributes(world.get_blueprint_library())
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        actor_list.append(self.sensor)
        self.sensor.listen(lambda data: sensor_callback(data, self.queue))
        self.sensor_id = self.__class__.sensor_id_glob
        self.__class__.sensor_id_glob += 1
        self.folder_output = folder_output
        self.ts_tmp = 0


class Camera(Sensor):
    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Sensor.__init__(self, vehicle, world, actor_list,
                        folder_output, transform)
        self.sensor_frame_id = 0
        self.frame_output = self.folder_output + \
            "/images_%s" % str.lower(self.__class__.__name__)
        os.makedirs(self.frame_output) if not os.path.exists(self.frame_output) else [
            os.remove(f) for f in glob.glob(self.frame_output+"/*") if os.path.isfile(f)]

        with open(self.folder_output+"/full_ts_camera.txt", 'w') as file:
            file.write("# frame_id timestamp\n")

        print('created %s' % self.sensor)

    def save(self, color_converter=carla.ColorConverter.Raw):
        while not self.queue.empty():
            data = self.queue.get()

            ts = data.timestamp - Sensor.initial_ts
            if (ts - self.ts_tmp > 0.11) or (ts - self.ts_tmp) < 0:  # check for 10Hz camera acquisition
                print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %
                      (self.ts_tmp, ts))
                sys.exit()
            self.ts_tmp = ts

            file_path = self.frame_output + \
                "/%04d_%d.png" % (self.sensor_frame_id, self.sensor_id)
            x = threading.Thread(target=data.save_to_disk,
                                 args=(file_path, color_converter))
            x.start()
            print("Export : "+file_path)

            if self.sensor_id == 0:
                with open(self.folder_output+"/full_ts_camera.txt", 'a') as file:
                    # bug in CARLA 0.9.10: timestamp of camera is one tick late. 1 tick = 1/fps_simu seconds
                    file.write(str(self.sensor_frame_id)+" " +
                               str(data.timestamp - Sensor.initial_ts)+"\n")
            self.sensor_frame_id += 1
        return self.sensor_frame_id


class RGB(Camera):
    sensor_id_glob = 0

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list,
                        folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '1392')
        camera_bp.set_attribute('image_size_y', '1024')
        # 72 degrees # Always fov on width even if width is different than height
        camera_bp.set_attribute('fov', '72')
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera
        camera_bp.set_attribute('gamma', '2.2')
        camera_bp.set_attribute('motion_blur_intensity', '0')
        camera_bp.set_attribute('motion_blur_max_distortion', '0')
        camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
        camera_bp.set_attribute('shutter_speed', '1000')  # 1 ms shutter_speed
        camera_bp.set_attribute('lens_k', '0')
        camera_bp.set_attribute('lens_kcube', '0')
        camera_bp.set_attribute('lens_x_size', '0')
        camera_bp.set_attribute('lens_y_size', '0')
        return camera_bp

    def save(self):
        return Camera.save(self)


class SS(Camera):
    sensor_id_glob = 10

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list,
                        folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find(
            'sensor.camera.semantic_segmentation')

        camera_ss_bp.set_attribute('image_size_x', '1392')
        camera_ss_bp.set_attribute('image_size_y', '1024')
        # 72 degrees # Always fov on width even if width is different than height
        camera_ss_bp.set_attribute('fov', '72')
        camera_ss_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera
        return camera_ss_bp

    def save(self, color_converter=carla.ColorConverter.CityScapesPalette):
        Camera.save(self, color_converter)


class Depth(Camera):
    sensor_id_glob = 20

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        Camera.__init__(self, vehicle, world, actor_list,
                        folder_output, transform)

    def set_attributes(self, blueprint_library):
        camera_ss_bp = blueprint_library.find('sensor.camera.depth')

        camera_ss_bp.set_attribute('image_size_x', '1392')
        camera_ss_bp.set_attribute('image_size_y', '1024')
        # 72 degrees # Always fov on width even if width is different than height
        camera_ss_bp.set_attribute('fov', '72')
        camera_ss_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera
        return camera_ss_bp

    # def save(self, color_converter=carla.ColorConverter.Depth):
    #    Camera.save(self, color_converter)

    def save(self):
        Camera.save(self)


class HDL64E(Sensor):
    sensor_id_glob = 100

    def __init__(self, vehicle, world, actor_list, folder_output, transform):
        self.rotation_lidar = rotation_carla(transform.rotation)
        self.rotation_lidar_transpose = self.rotation_lidar.T
        self.queue = queue.PriorityQueue()
        self.bp = self.set_attributes(world.get_blueprint_library())
        self.sensor = world.spawn_actor(self.bp, transform, attach_to=vehicle)
        actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.queue.put((data.timestamp, data)))
        self.sensor_id = self.__class__.sensor_id_glob
        self.__class__.sensor_id_glob += 1
        self.folder_output = folder_output

        self.i_packet = 0
        self.i_frame = 0

        self.initial_loc = np.zeros(3)
        self.initial_rot = np.identity(3)
        self.calib_output = folder_output
        self.frame_output = folder_output+"/frames"
        os.makedirs(self.frame_output) if not os.path.exists(self.frame_output) else [
            os.remove(f) for f in glob.glob(self.frame_output+"/*") if os.path.isfile(f)]

        settings = world.get_settings()
        self.packet_per_frame = 1 / \
            (self.bp.get_attribute('rotation_frequency').as_float()
             * settings.fixed_delta_seconds)
        self.packet_period = settings.fixed_delta_seconds

        self.pt_size = 4*4  # (4 float32 with x, y, z, timestamp)

        header = ['ply']
        header.append('format binary_little_endian 1.0')
        header.append('element vertex ')
        self.begin_header = '\n'.join(header)
        header = ["property float x"]
        header.append("property float y")
        header.append("property float z")
        header.append("property float timestamp")
        header.append('end_header')
        self.end_header = '\n'.join(header)+'\n'

        self.list_pts = []
        self.list_semantic = []
        self.list_ts = []
        self.list_trajectory = []

        self.ts_tmp = 0.0
        print('created %s' % self.sensor)

    def init(self):
        self.initial_loc = translation_carla(self.sensor.get_location())
        self.initial_rot_transpose = (rotation_carla(
            self.sensor.get_transform().rotation).dot(self.rotation_lidar)).T
        with open(self.calib_output+"/full_poses_lidar.txt", 'w') as posfile:
            posfile.write(
                "# R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2) timestamp\n")

    def save(self):
        while not self.queue.empty():
            data = self.queue.get()[1]
            ts = data.timestamp - Sensor.initial_ts
            if (ts - self.ts_tmp) > self.packet_period * 1.5 or (ts - self.ts_tmp < 0):
                print("[Error in timestamp] HDL64E: previous_ts %f -> ts %f" %
                      (self.ts_tmp, ts))
                sys.exit()

            self.ts_tmp = ts

            nbr_pts = len(data.raw_data)//24  # 4 float32 and 2 uint
            self.list_ts.append(np.broadcast_to(ts, nbr_pts))
            buffer = np.frombuffer(data.raw_data, dtype=np.dtype(
                [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos', 'f4'), ('index', 'u4'), ('semantic', 'u4')]))

            # We're negating the y to correctly visualize a world that matches what we see in Unreal since we uses a right-handed coordinate system
            self.list_pts.append(
                np.array([buffer[:]['x'], -buffer[:]['y'], buffer[:]['z'], buffer[:]['cos']]))
            self.list_semantic.append(
                np.array([buffer[:]['index'], buffer[:]['semantic']]))

            self.i_packet += 1
            if self.i_packet % self.packet_per_frame == 0:
                pts_all = np.hstack(self.list_pts)
                pts_all[0:3, :] = self.rotation_lidar_transpose.dot(
                    pts_all[0:3, :])
                pts_all = pts_all.T
                semantic_all = np.hstack(self.list_semantic).T
                ts_all = np.concatenate(self.list_ts)
                self.list_pts = []
                self.list_semantic = []
                self.list_ts = []

                ply_file_path = self.frame_output+"/frame_%04d.ply" % self.i_frame

                if ply.write_ply(ply_file_path, [np.float32(pts_all), np.float32(ts_all), np.uint32(semantic_all)], ['x', 'y', 'z', 'cos_angle_lidar_surface', 'timestamp', 'instance', 'semantic']):
                    print("Export : "+ply_file_path)
                else:
                    print('ply.write_ply() failed')

                self.i_frame += 1

            R_W = self.initial_rot_transpose.dot(rotation_carla(
                data.transform.rotation).dot(self.rotation_lidar))
            T_W = self.initial_rot_transpose.dot(translation_carla(
                data.transform.location) - self.initial_loc)

            with open(self.calib_output+"/full_poses_lidar.txt", 'a') as posfile:
                posfile.write(
                    " ".join(map(str, [r for r in R_W[0]]))+" "+str(T_W[0])+" ")
                posfile.write(
                    " ".join(map(str, [r for r in R_W[1]]))+" "+str(T_W[1])+" ")
                posfile.write(
                    " ".join(map(str, [r for r in R_W[2]]))+" "+str(T_W[2])+" ")
                posfile.write(str(ts)+"\n")

            self.list_trajectory.extend([struct.pack('f', T) for T in T_W])
            self.list_trajectory.append(struct.pack('f', ts))

        return self.i_frame

    def save_poses(self):
        ply_file_path = self.calib_output+"/poses_lidar.ply"
        trajectory_bytes = b''.join(self.list_trajectory)
        with open(ply_file_path, 'w') as posfile:
            posfile.write(
                self.begin_header+str(len(trajectory_bytes)//self.pt_size)+'\n'+self.end_header)
        with open(ply_file_path, 'ab') as posfile:
            posfile.write(trajectory_bytes)
        print("Export : "+ply_file_path)

    def set_attributes(self, blueprint_library):
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '80.0')    # 80.0 m
        lidar_bp.set_attribute('points_per_second', str(64/0.00004608))
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('upper_fov', str(2))
        lidar_bp.set_attribute('lower_fov', str(-24.8))
        return lidar_bp
