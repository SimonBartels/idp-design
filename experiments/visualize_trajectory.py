import pickle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from align_point_clouds import PointCloudAlignment

f = open("../output/bla/ref_traj/iter44/traj.pos", "rb")
xyz = pickle.load(f)
f.close()

TARGET_RESIDUES = [297, xyz.shape[1] - 6]
#last_atom = 917
#TARGET_RESIDUES = [710, 895]

point_cloud_alignment = PointCloudAlignment(xyz[0, :348, :])



def update_graph(num):
    xyz_ = point_cloud_alignment.align(xyz[num, :348], xyz[num, :])
    graph._offsets3d = (xyz_[:, 0], xyz_[:, 1], xyz_[:, 2])
    contact_points._offsets3d = (xyz_[TARGET_RESIDUES, 0], xyz_[TARGET_RESIDUES, 1], xyz_[TARGET_RESIDUES, 2])
    linker._offsets3d = (xyz_[-20:-10, 0], xyz_[-20:-10, 1], xyz_[-20:-10, 2])
    title.set_text('time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(70, 80)
# ax.set_ylim3d(70, 80)
# ax.set_zlim3d(70, 80)
title = ax.set_title('time=0')

graph = ax.scatter(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], label="domain")
contact_points = ax.scatter(xyz[0, TARGET_RESIDUES, 0], xyz[0, TARGET_RESIDUES, 1], xyz[0, TARGET_RESIDUES, 2], c='red', label="desired contact residues", marker="s", s=50)
linker = ax.scatter(xyz[0, -20:-10, 0], xyz[0, -20:-10, 1], xyz[0, -20:-10, 2], c='green', label="linker", marker="s", s=50)
plt.legend()

ani = FuncAnimation(fig, update_graph, xyz.shape[0], interval=500, blit=False)
#writer = PillowWriter(fps=30)
#ani.save('./figures/animation.gif', writer=writer)
plt.show()
