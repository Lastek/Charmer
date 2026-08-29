import trimesh
import numpy as np

vertices = np.zeros((3, 3))
faces = np.array([[0, 1, 2]])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export('blank.glb')