import carla
from carla import VehicleLightState as vls

import queue
import logging

import math
import random
import numpy as np


def rotation_carla(rotation):
    # Function to change rotations in CARLA from left-handed to right-handed reference frame
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    return np.array([[cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr], [-sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [sp, cp*sr, cp*cr]])


def translation_carla(location):
    # Function to change translations in CARLA from left-handed to right-handed reference frame
    if isinstance(location, np.ndarray):
        return location*(np.array([[1], [-1], [1]]))
    else:
        return np.array([location.x, -location.y, location.z])


def transform_lidar_to_camera(lidar_tranform, camera_transform):
    R_camera_vehicle = rotation_carla(camera_transform.rotation)
    # rotation_carla(lidar_tranform.rotation) #we want the lidar frame to have x forward
    R_lidar_vehicle = np.identity(3)
    R_lidar_camera = R_camera_vehicle.T.dot(R_lidar_vehicle)
    T_lidar_camera = R_camera_vehicle.T.dot(translation_carla(np.array([[lidar_tranform.location.x], [lidar_tranform.location.y], [
                                            lidar_tranform.location.z]])-np.array([[camera_transform.location.x], [camera_transform.location.y], [camera_transform.location.z]])))
    return np.vstack((np.hstack((R_lidar_camera, T_lidar_camera)), [0, 0, 0, 1]))


def spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id):
    world = client.get_world()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_random_device_seed(args.seed)

    traffic_manager.set_synchronous_mode(True)
    synchronous_master = True

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

    safe = True
    if safe:
        blueprints = [x for x in blueprints if int(
            x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    print("Number of spawn points : ", number_of_spawn_points)

    if nbr_vehicles <= number_of_spawn_points:
        random.shuffle(spawn_points)
    elif nbr_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, nbr_vehicles, number_of_spawn_points)
        nbr_vehicles = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= nbr_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        car_lights_on = False
        if car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                     .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    walkers_list = []
    percentagePedestriansRunning = 0.0            # how many pedestrians will run
    # how many pedestrians will walk through the road
    percentagePedestriansCrossing = 0.0
    # 1. take all the random locations to spawn
    spawn_points = []
    all_loc = []
    i = 0
    while i < nbr_walkers:
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if ((loc != None) and not (loc in all_loc)):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
            all_loc.append(loc)
            i = i + 1
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute(
                    'speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute(
                    'speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp,
                     carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_walkers_id.append(walkers_list[i]["con"])
        all_walkers_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_walkers_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_walkers_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(
            world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('Spawned %d vehicles and %d walkers' %
          (len(vehicles_list), len(walkers_list)))

    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)


def follow(transform, world):
    # Transforme carla.Location(x,y,z) from sensor to world frame
    rot = transform.rotation
    rot.pitch = -25
    world.get_spectator().set_transform(carla.Transform(
        transform.transform(carla.Location(x=-15, y=0, z=5)), rot))
