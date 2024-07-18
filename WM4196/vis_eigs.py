import napari
import mrcfile
import numpy as np

volumes = [f'eigenvolumes/eigenvolume_{i:04d}.mrc' for i in range(1, 51)]
volumes = [mrcfile.open(vol).data for vol in volumes]

viewer = napari.Viewer()
for volume in volumes[12:0:-1]:
    viewer.add_image(np.sum(volume[24:28], axis=0), interpolation='spline36')

viewer.grid.enabled = True
viewer.grid.shape = (3, 4)
viewer.grid.stride = 1
napari.run()