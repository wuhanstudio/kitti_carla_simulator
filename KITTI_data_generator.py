import os
import time
import glob
import argparse

import carla
import queue

from modules.sensors import Sensor, RGB, Depth, SS, HDL64E
from modules.utils import follow, spawn_npc, transform_lidar_to_camera

nbr_frame = 1_000    # MAX = 10_000
nbr_walkers = 0
nbr_vehicles = 50

spawn_points = [23, 46, 0, 125, 53, 257, 62]

def screenshot(vehicle, world, actor_list, folder_output, transform):
    sensor = world.spawn_actor(RGB.set_attributes(
        RGB, world.get_blueprint_library()), transform, attach_to=vehicle)
    actor_list.append(sensor)
    screenshot_queue = queue.Queue()
    sensor.listen(screenshot_queue.put)
    print('created %s' % sensor)

    while screenshot_queue.empty():
        world.tick()

    file_path = folder_output+"/screenshot.png"
    screenshot_queue.get().save_to_disk(file_path)
    print("Export : "+file_path)
    actor_list[-1].destroy()
    print('destroyed %s' % actor_list[-1])
    del actor_list[-1]

def main(i_map, use_depth_camera, use_sem_seg_camera, use_lidar):

    start_record_full = time.time()

    time_stop = 2.0
    fps_simu = 1000.0

    actor_list = []
    vehicles_list = []
    all_walkers_id = []

    init_settings = None

    try:
        # Connect to CARLA simulator
        client = carla.Client('localhost', 2000)
        init_settings = carla.WorldSettings()
        client.set_timeout(100.0)

        # Load the map
        print("Map Town0"+str(i_map))
        world = client.load_world("Town0"+str(i_map))

        folder_output = "KITTI_Dataset_CARLA_v%s/%s/generated" % (
            client.get_client_version(), world.get_map().name)
        os.makedirs(folder_output) if not os.path.exists(folder_output) else [
            os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]

        client.start_recorder(os.path.dirname(os.path.realpath(
            __file__))+"/"+folder_output+"/recording.log")

        # Weather
        world.set_weather(carla.WeatherParameters.WetCloudyNoon)

        # Set Synchronous mode
        print("Applying world settings")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps_simu
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        # Create KITTI vehicle
        print("Creating the ego vehicle")
        blueprint_library = world.get_blueprint_library()
        bp_KITTI = blueprint_library.find('vehicle.tesla.model3')
        bp_KITTI.set_attribute('color', '228, 239, 241')
        bp_KITTI.set_attribute('role_name', 'KITTI')
        start_pose = world.get_map().get_spawn_points()[spawn_points[i_map-1]]
        KITTI = world.spawn_actor(bp_KITTI, start_pose)
        actor_list.append(KITTI)
        print('Created %s' % KITTI)

        # Spawn vehicles and walkers
        spawn_npc(client, nbr_vehicles, nbr_walkers,
                  vehicles_list, all_walkers_id)

        # Wait for KITTI to stop
        start = world.get_snapshot().timestamp.elapsed_seconds
        print("Waiting for KITTI to stop ...")
        while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop:
            world.tick()
        print("KITTI stopped")

        # Set sensors transformation from KITTI
        lidar_transform = carla.Transform(carla.Location(
            x=0, y=0, z=1.80), carla.Rotation(pitch=0, yaw=180, roll=0))
        cam0_transform = carla.Transform(carla.Location(
            x=0.30, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))
        cam1_transform = carla.Transform(carla.Location(
            x=0.30, y=0.50, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))

        # Take a screenshot
        screenshot(KITTI, world, actor_list, folder_output, carla.Transform(
            carla.Location(x=0.0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)))

        # Create our sensors
        RGB.sensor_id_glob = 0
        SS.sensor_id_glob = 10
        Depth.sensor_id_glob = 20
        HDL64E.sensor_id_glob = 100

        cam0 = RGB(KITTI, world, actor_list, folder_output, cam0_transform)
        cam1 = RGB(KITTI, world, actor_list, folder_output, cam1_transform)

        if use_lidar:
            VelodyneHDL64 = HDL64E(
                KITTI, world, actor_list, folder_output, lidar_transform)

        if use_depth_camera:
            cam0_depth = Depth(KITTI, world, actor_list,
                               folder_output, cam0_transform)
            cam1_depth = Depth(KITTI, world, actor_list,
                               folder_output, cam1_transform)

        if use_sem_seg_camera:
            cam0_ss = SS(KITTI, world, actor_list,
                         folder_output, cam0_transform)
            cam1_ss = SS(KITTI, world, actor_list,
                         folder_output, cam1_transform)

        # Export LiDAR to cam0 transformation
        tf_lidar_cam0 = transform_lidar_to_camera(
            lidar_transform, cam0_transform)
        with open(folder_output+"/lidar_to_cam0.txt", 'w') as posfile:
            posfile.write(
                "#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
            posfile.write(str(tf_lidar_cam0[0][0])+" "+str(tf_lidar_cam0[0][1])+" "+str(
                tf_lidar_cam0[0][2])+" "+str(tf_lidar_cam0[0][3])+" ")
            posfile.write(str(tf_lidar_cam0[1][0])+" "+str(tf_lidar_cam0[1][1])+" "+str(
                tf_lidar_cam0[1][2])+" "+str(tf_lidar_cam0[1][3])+" ")
            posfile.write(str(tf_lidar_cam0[2][0])+" "+str(tf_lidar_cam0[2][1])+" "+str(
                tf_lidar_cam0[2][2])+" "+str(tf_lidar_cam0[2][3]))

        # Export LiDAR to cam1 transformation
        tf_lidar_cam1 = transform_lidar_to_camera(
            lidar_transform, cam1_transform)
        with open(folder_output+"/lidar_to_cam1.txt", 'w') as posfile:
            posfile.write(
                "#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
            posfile.write(str(tf_lidar_cam1[0][0])+" "+str(tf_lidar_cam1[0][1])+" "+str(
                tf_lidar_cam1[0][2])+" "+str(tf_lidar_cam1[0][3])+" ")
            posfile.write(str(tf_lidar_cam1[1][0])+" "+str(tf_lidar_cam1[1][1])+" "+str(
                tf_lidar_cam1[1][2])+" "+str(tf_lidar_cam1[1][3])+" ")
            posfile.write(str(tf_lidar_cam1[2][0])+" "+str(tf_lidar_cam1[2][1])+" "+str(
                tf_lidar_cam1[2][2])+" "+str(tf_lidar_cam1[2][3]))

        # Launch KITTI
        KITTI.set_autopilot(True)

        # Pass to the next simulator frame to spawn sensors and to retrieve first data
        world.tick()

        if use_lidar:
            VelodyneHDL64.init()

        follow(KITTI.get_transform(), world)

        # All sensors produce first data at the same time (this ts)
        Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds

        start_record = time.time()
        print("Start record : ")
        frame_current = 0
        while (frame_current < nbr_frame):

            # Save RGB camera data
            frame_current = cam0.save()
            cam1.save()

            # Save semantic segmentation camera data
            if use_sem_seg_camera:
                cam0_ss.save()
                cam1_ss.save()

            # Save depth camera data
            if use_depth_camera:
                cam0_depth.save()
                cam1_depth.save()

            # Save lidar data
            if use_lidar:
                VelodyneHDL64.save()

            # Update the spectator view
            follow(KITTI.get_transform(), world)
            world.tick()

        if use_lidar:
            VelodyneHDL64.save_poses()

        client.stop_recorder()
        print("Stop record")

        # Stop cameras
        cam0.sensor.stop()
        cam1.sensor.stop()

        if use_depth_camera:
            cam0_depth.sensor.stop()
            cam1_depth.sensor.stop()

        if use_sem_seg_camera:
            cam0_ss.sensor.stop()
            cam1_ss.sensor.stop()

        if use_lidar:
            VelodyneHDL64.sensor.stop()

        print('Destroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in vehicles_list])
        vehicles_list.clear()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        all_actors = world.get_actors(all_walkers_id)
        for i in range(0, len(all_walkers_id), 2):
            all_actors[i].stop()

        print('Destroying %d walkers' % (len(all_walkers_id)//2))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in all_walkers_id])
        all_walkers_id.clear()

        print('Destroying KITTI')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        actor_list.clear()

        print("Elapsed time : ", time.time()-start_record)
        print()

        time.sleep(time_stop)

    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        world.apply_settings(init_settings)

        time.sleep(time_stop)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='KITTI dataset generator')
    parser.add_argument(
        '--town',
        type=int,
        default=1,
        help='Map index: Town 1, 2, 3, ..., 7'
    )
    parser.add_argument('--use-depth-camera',
                        action='store_true', help='Use depth camera')
    parser.add_argument('--use-sem-seg-camera',
                        action='store_true', help='Use depth camera')
    parser.add_argument('--use-lidar', action='store_true', help='Use Lidar')
    args = parser.parse_args()

    main(args.town, args.use_depth_camera,
         args.use_sem_seg_camera, args.use_lidar)
