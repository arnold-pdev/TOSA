import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

shapes = np.load('shapes.npy')
bc = np.load('boundary_conditions.npy', allow_pickle=True)
loads = np.load('loads.npy', allow_pickle=True)
vfs = np.load('vfs.npy')
topologies = np.load('topologies.npy', allow_pickle=True)

# For an example topology
# i = 3
for i in range(0, 20):

    design = topologies[i].reshape(shapes[i], order='C')  # "undo" the flattening done by the authors, according to the C convention they (seem to have) used
    # Plot using pyvista
    grid = pv.ImageData()  # Create a grid (and give it the same dimensions as the data)
    grid.dimensions = np.array(shapes[i]) + 1 # +1 because dimensions define points, not cells
    grid.cell_data["values"] = design.flatten(order="F") # Flatten in Fortran order for VTK
    # Threshold to remove 0s (voids)
    solid = grid.threshold(0.5)  # this keeps only cells where "values" >= 0.5 (the 1s)
    # solid.plot(show_edges=True, color="tan", opacity=0.5)

    bc_mesh = pv.PolyData(
                            bc[i][:, :3] * np.max(shapes[i]) # take the first three cols of the bc (all rows), since the first three cols corresp to bc locations
    )                                      # the np.max() here is as a result of Hongrui's clarification

    load_mesh = pv.PolyData(
                            (loads[i][0][:3] * np.max(shapes[i])).astype(float)
    )                                      # the np.max() here is as a result of Hongrui's clarification

    p = pv.Plotter()
    p.enable_depth_peeling()

    p.add_mesh(solid, show_edges=True, color="tan", opacity=0.4, label="Topology")

    p.add_mesh(bc_mesh, color="red", point_size=15, render_points_as_spheres=True, label="BCs")

    p.add_mesh(load_mesh, color="blue", point_size=15, render_points_as_spheres=True, label="Load")

    p.add_legend()

    p.add_axes_at_origin()
    p.show_grid()

    p.show()




# # Plot using ax.voxels
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# ax.voxels(design, edgecolor='k', alpha=0.8)  # ax.voxels expects a boolean array (True for solid, False for void); filled=data indicates which voxels to render
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# plt.show()


print('Done')
