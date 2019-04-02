import utils
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
superstrip_locs = utils.find_top_hyperstrips(500)


ax = Axes3D( plt.figure(figsize=(20,20)))
Xs = []
Ys = []
Zs = []
colors = []
#hard coding layers
layers = {2:'C0',4:'C1', 6:'C2', 8:'C3', 10:'C4', 12:'C5', 14:'C6'}

for superstrip in superstrip_locs:
    vol, lay, mod = superstrip
    x, y, z = superstrip_locs[superstrip]
    Xs.append(x)
    Ys.append(y)
    Zs.append(z)
    colors.append(layers[lay])

ax.scatter(Zs, Xs, Ys, alpha=1, c=colors)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
#ax.legend()
ax.scatter(150,150,150, s=0)
ax.scatter(-150,-150,-150, s=0)
plt.show()
