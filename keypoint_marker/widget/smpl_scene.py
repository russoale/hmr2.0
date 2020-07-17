import numpy as np
from trimesh import Scene


class SmplScene(Scene):

    def calculate_regressor(self):
        mesh = self.geometry['geometry_0']
        joint = self.geometry['joint_0']




        ray_direction = np.eye(3)

        mesh.clos
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=joint,
            ray_directions=ray_direction)