import sapien.core as sapien
import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import cv2
import torch
import yaml
import trimesh
import math
from .._GLOBAL_CONFIGS import CONFIGS_PATH
import os
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()


class Camera:

    def __init__(self, bias=0, random_head_camera_dis=0, **kwags):
        """ """
        self.random_head_camera_dis = random_head_camera_dis

        self.static_camera_config = []
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")

        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)

        # embodiment = kwags.get('embodiment')
        # embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
        # with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        #     embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        # robot_file = embodiment_types[embodiment]['file_path']
        # if robot_file is None:
        #     raise "No embodiment files"

        # robot_config_file = os.path.join(robot_file, 'config.yml')
        # with open(robot_config_file, 'r', encoding='utf-8') as f:
        #     embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        # TODO
        self.static_camera_info_list = kwags["left_embodiment_config"]["static_camera_list"]
        self.static_camera_num = len(self.static_camera_info_list)

    def load_camera(self, scene):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """
        near, far = 0.1, 100
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        # camera_args = get_camera_config()
        def create_camera(camera_info, random_head_camera_dis=0):
            if camera_info["type"] not in camera_args.keys():
                raise ValueError(f"Camera type {camera_info['type']} not supported")

            camera_config = camera_args[camera_info["type"]]
            cam_pos = np.array(camera_info["position"])
            vector = np.random.randn(3)
            random_dir = vector / np.linalg.norm(vector)
            cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
            cam_forward = np.array(camera_info["forward"]) / np.linalg.norm(np.array(camera_info["forward"]))
            cam_left = np.array(camera_info["left"]) / np.linalg.norm(np.array(camera_info["left"]))
            up = np.cross(cam_forward, cam_left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
            mat44[:3, 3] = cam_pos

            # ========================= sensor camera =========================
            # sensor_config = StereoDepthSensorConfig()
            # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

            camera = scene.add_camera(
                name=camera_info["name"],
                width=camera_config["w"],
                height=camera_config["h"],
                fovy=np.deg2rad(camera_config["fovy"]),
                near=near,
                far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))

            # ========================= sensor camera =========================
            # sensor_camera = StereoDepthSensor(
            #     sensor_config,
            #     sensor_mount_actor,
            #     sapien.Pose(mat44)
            # )
            # camera.entity.set_pose(sapien.Pose(camera_info['position']))
            # return camera, sensor_camera, camera_config
            return camera, camera_config

        # ================================= wrist camera =================================
        if self.collect_wrist_camera:
            wrist_camera_config = camera_args[self.wrist_camera_type]
            self.left_camera = scene.add_camera(
                name="left_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

            self.right_camera = scene.add_camera(
                name="right_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

        # ================================= sensor camera =================================
        # sensor_config = StereoDepthSensorConfig()
        # sensor_config.rgb_resolution = (wrist_camera_config['w'], wrist_camera_config['h'])
        # self.left_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # self.right_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # ================================= static camera =================================
        self.head_camera_id = None
        self.static_camera_list = []
        # self.static_sensor_camera_list = []
        self.static_camera_name = []
        # static camera list
        for i, camera_info in enumerate(self.static_camera_info_list):
            if camera_info.get("forward") == None:
                camera_info["forward"] = (-1 * np.array(camera_info["position"])).tolist()
            if camera_info.get("left") == None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                ] + [0]

            if camera_info["name"] == "head_camera":
                if self.collect_head_camera:
                    self.head_camera_id = i
                    camera_info["type"] = self.head_camera_type
                    # camera, sensor_camera, camera_config = create_camera(camera_info)
                    camera, camera_config = create_camera(camera_info,
                                                          random_head_camera_dis=self.random_head_camera_dis)
                    self.static_camera_list.append(camera)
                    self.static_camera_name.append(camera_info["name"])
                    # self.static_sensor_camera_list.append(sensor_camera)
                    self.static_camera_config.append(camera_config)
                    # ================================= sensor camera =================================
                    # camera_config = get_camera_config(camera_info['type'])
                    # cam_pos = np.array(camera_info['position'])
                    # cam_forward = np.array(camera_info['forward']) / np.linalg.norm(np.array(camera_info['forward']))
                    # cam_left = np.array(camera_info['left']) / np.linalg.norm(np.array(camera_info['left']))
                    # up = np.cross(cam_forward, cam_left)
                    # mat44 = np.eye(4)
                    # mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
                    # mat44[:3, 3] = cam_pos
                    # sensor_config = StereoDepthSensorConfig()
                    # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

                    # self.head_sensor = StereoDepthSensor(
                    #     sensor_config,
                    #     sensor_mount_actor,
                    #     sapien.Pose(mat44)
                    # )
            else:
                # camera, sensor_camera, camera_config = create_camera(camera_info)
                camera, camera_config = create_camera(camera_info)
                self.static_camera_list.append(camera)
                self.static_camera_name.append(camera_info["name"])
                # self.static_sensor_camera_list.append(sensor_camera)
                self.static_camera_config.append(camera_config)

        # observer camera
        self.observer_camera = scene.add_camera(
            name="observer_camera",
            width=640,
            height=480,
            fovy=np.deg2rad(93),
            near=near,
            far=far,
        )
        observer_cam_pos = np.array([0.0, 0.23, 1.33])
        observer_cam_forward = np.array([0, -1, -1.02])
        # observer_cam_left = np.array([1,-1, 0])
        observer_cam_left = np.array([1, 0, 0])
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))

        # world pcd camera
        self.world_camera1 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([0.4, -0.4, 1.6])
        world_cam_forward = np.array([-1, 1, -1.4])
        world_cam_left = np.array([-1, -1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera1.entity.set_pose(sapien.Pose(world_cam_mat44))

        self.world_camera2 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([-0.4, -0.4, 1.6])
        world_cam_forward = np.array([1, 1, -1.4])
        world_cam_left = np.array([-1, 1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera2.entity.set_pose(sapien.Pose(world_cam_mat44))

    def update_picture(self):
        # camera
        if self.collect_wrist_camera:
            self.left_camera.take_picture()
            self.right_camera.take_picture()

        for camera in self.static_camera_list:
            camera.take_picture()

        # ================================= sensor camera =================================
        # self.head_sensor.take_picture()
        # self.head_sensor.compute_depth()

    def update_wrist_camera(self, left_pose, right_pose):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.collect_wrist_camera:
            self.left_camera.entity.set_pose(left_pose)
            self.right_camera.entity.set_pose(right_pose)

    def get_config(self) -> dict:
        res = {}

        def _get_config(camera):
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            return {
                "intrinsic_cv": camera_intrinsic_cv,
                "extrinsic_cv": camera_extrinsic_cv,
                "cam2world_gl": camera_model_matrix,
            }

        if self.collect_wrist_camera:
            res["left_camera"] = _get_config(self.left_camera)
            res["right_camera"] = _get_config(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = _get_config(camera)
            else:
                res[camera_name] = _get_config(camera)
        # ================================= sensor camera =================================
        # res['head_sensor'] = res['head_camera']
        # print(res)
        return res

    def get_rgb(self) -> dict:
        rgba = self.get_rgba()
        rgb = {}
        for camera_name, camera_data in rgba.items():
            rgb[camera_name] = {}
            rgb[camera_name]["rgb"] = camera_data["rgba"][:, :, :3]  # Exclude alpha channel
        return rgb
    
    # Get Camera RGBA
    def get_rgba(self) -> dict:

        def _get_rgba(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
            return camera_rgba_img

        # ================================= sensor camera =================================
        # def _get_sensor_rgba(sensor):
        #     camera_rgba = sensor.get_rgb()
        #     camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:,:,:3]
        #     return camera_rgba_img

        res = {}

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["rgba"] = _get_rgba(self.left_camera)
            res["right_camera"]["rgba"] = _get_rgba(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["rgba"] = _get_rgba(camera)
            else:
                res[camera_name] = {}
                res[camera_name]["rgba"] = _get_rgba(camera)
        # ================================= sensor camera =================================
        # res['head_sensor']['rgb'] = _get_sensor_rgba(self.head_sensor)

        return res

    def get_observer_rgb(self) -> dict:
        self.observer_camera.take_picture()

        def _get_rgb(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgb_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            return camera_rgb_img

        return _get_rgb(self.observer_camera)

    # Get Camera Segmentation
    def get_segmentation(self, level="mesh") -> dict:

        def _get_segmentation(camera, level="mesh"):
            # visual_id is the unique id of each visual shape
            seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
            if level == "mesh":
                label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            elif level == "actor":
                label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            return color_palette[label0_image]

        res = {
            # 'left_camera':{},
            # 'right_camera':{}
        }

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"][f"{level}_segmentation"] = _get_segmentation(self.left_camera, level=level)
            res["right_camera"][f"{level}_segmentation"] = _get_segmentation(self.right_camera, level=level)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
            else:
                res[camera_name] = {}
                res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
        return res

    # Get Camera Depth
    def get_depth(self) -> dict:

        def _get_depth(camera):
            position = camera.get_picture("Position")
            depth = -position[..., 2]
            depth_image = (depth * 1000.0).astype(np.float64)
            return depth_image

        def _get_sensor_depth(sensor):
            depth = sensor.get_depth()
            depth = (depth * 1000.0).astype(np.float64)
            return depth

        res = {}
        rgba = self.get_rgba()

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["depth"] = _get_depth(self.left_camera)
            res["right_camera"]["depth"] = _get_depth(self.right_camera)
            res["left_camera"]["depth"] *= rgba["left_camera"]["rgba"][:, :, 3] / 255
            res["right_camera"]["depth"] *= rgba["right_camera"]["rgba"][:, :, 3] / 255
        
        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["depth"] = _get_depth(camera)
                    res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
            else:
                res[camera_name] = {}
                res[camera_name]["depth"] = _get_depth(camera)
                res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
        # res['head_sensor']['depth'] = _get_sensor_depth(self.head_sensor)

        return res

