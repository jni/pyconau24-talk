---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Explore, annotate, and analyze multi-dimensional images in Python with napari

+++

## 1.1 – a *fast* 2D viewer

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import napari
```

```{code-cell} ipython3
import tifffile

image = tifffile.imread(
    '/Users/jni/projects/demos/spatialdata-sandbox/'
    'visium_io/data/Visium_Mouse_Olfactory_Bulb_image.tif'
)
```

```{code-cell} ipython3
image.shape
```

```{code-cell} ipython3
%matplotlib qt
```

```{code-cell} ipython3
plt.imshow(image)
```

```{code-cell} ipython3
viewer, layer = napari.imshow(image)
```

## 1.2 – a *multidimensional* viewer

+++

### 3D multichannel cells

```{code-cell} ipython3
from skimage import data

cells = data.cells3d()

cells.shape
```

```{code-cell} ipython3
data.cells3d?
```

```{code-cell} ipython3
viewer, layer = napari.imshow(
        cells,
        channel_axis=1,
        scale=[0.29, 0.26, 0.26],
        )
```

### CryoET Dynamo PCA analysis

Credit: Alister Burt (currently at Genentech)

```{code-cell} ipython3
from pathlib import Path
from functools import cached_property

import pandas as pd
import numpy as np
import mrcfile


class DPCAA:
    def __init__(self, eigenvolumes_dir, eigentable_file):
        self.eigenvolumes_dir = Path(eigenvolumes_dir)
        self.eigentable_file = eigentable_file
    
    @cached_property
    def eigenvolumes(self):
        volume_files = list(self.eigenvolumes_dir.glob('*.mrc'))
        df = pd.DataFrame({'path' : volume_files}).sort_values(by='path')
        df['eigenvolume'] = df['path'].apply(lambda x: mrcfile.open(x).data)
        eigenvolumes = np.stack(df['eigenvolume'])
        return eigenvolumes

    @cached_property
    def eigentable(self):
        return np.loadtxt(self.eigentable_file, delimiter=',')

    def spectral_average_from_coefficients(self, coefficients, normalise=True):
        coefficients = coefficients.squeeze()[..., np.newaxis, np.newaxis, np.newaxis]
        spectral_average = (coefficients * self.eigenvolumes).sum(axis=-4)

        if normalise:
            spectral_average = self._normalise_volume(spectral_average)

        return spectral_average

    def spectral_average_from_idx(self, idx):
        """generate spectral average from particles at idx
        """
        coefficients = self._coefficients_from_idx(idx)
        return self.spectral_average_from_coefficients(coefficients)

    def _coefficients_from_idx(self, idx):
        """generate coefficients from a set of particles at idx
        """
        return self.eigentable[idx, :].sum(axis=0)

    def _generate_volume_series(self, eig, n_bins=10, qcut=True):
        eig_coefficients = self.eigentable[:, eig]

        if qcut:
            cut = pd.qcut(eig_coefficients, n_bins)
        else:
            cut = pd.cut(eig_coefficients, n_bins)

        volumes = [self.spectral_average_from_idx(cut == subset) for subset in cut.categories]
        return np.stack(volumes)

    def _generate_volume_series_vectorised(self, eig, n_bins=10, qcut=True):
        eig_coefficients = self.eigentable[:, eig]

        if qcut:
            cut = pd.qcut(eig_coefficients, n_bins)
        else:
            cut = pd.cut(eig_coefficients, n_bins)

        coefficients = np.stack(
            [self._coefficients_from_idx(cut == subset) for subset in cut.categories]
        )
        volumes = self.spectral_average_from_coefficients(coefficients)
        return volumes

    def _normalise_volume(self, volume):
        """independently normalise a stack of volumes to mean 0 standard deviation 1
        """
        volume_axes = (-1, -2, -3)
        volume_mean = np.expand_dims(volume.mean(axis=volume_axes), axis=volume_axes)
        volume_std = np.expand_dims(volume.std(axis=volume_axes), axis=volume_axes)
        return (volume - volume_mean) / volume_std
```

```{code-cell} ipython3
from pathlib import Path

folder = Path('WM4196')
eigenvolumes = folder / 'eigenvolumes'
eigentable = folder / 'eigentable.csv'

pca = DPCAA(eigenvolumes_dir=eigenvolumes, eigentable_file=eigentable)

viewer = napari.Viewer()

n_bins = 10

volumes = np.stack([
        pca._generate_volume_series(comp, n_bins=n_bins, qcut=True)
        for comp in range(50)
        ])

viewer.add_image(volumes)
```

## 2 – a *layered* viewer

+++

### overlay images, segmentations, point detections, and more

```{code-cell} ipython3
from skimage import data

coins = data.coins()[50:-50, 50:-50]

viewer, im_layer = napari.imshow(coins)
```

```{code-cell} ipython3
from skimage import filters, measure, morphology, segmentation

thresholded = filters.threshold_otsu(coins) < coins
closed = morphology.closing(thresholded, morphology.square(4))
no_border = segmentation.clear_border(closed)
cleaned = morphology.remove_small_objects(no_border, 20)

segmented = measure.label(cleaned).astype(np.uint8)

label_layer = viewer.add_labels(segmented)
```

```{code-cell} ipython3
centroids = np.array([p.centroid for p in measure.regionprops(segmented)])
pts_layer = viewer.add_points(centroids, size=5)
```

(TODO: We need a cool shapes layer demo)

+++

### cryoET particle picking refinement

Credit: Alister Burt (currently at Genentech)

Code: https://github.com/alisterburt/napari-cryo-et-demo  
Data: https://www.ebi.ac.uk/empiar/EMPIAR-10164/

```{code-cell} ipython3
import mrcfile

# files containing data
tomogram_file = '/Users/jni/data/napari-cryo-et-demo/hiv/01_10.00Apx.mrc'
particles_file = '/Users/jni/data/napari-cryo-et-demo/hiv/01_10.00Apx_particles.star'

# loading data into memory
# tomogram is a numpy array containing image array data
with mrcfile.open(tomogram_file) as mrc:
    tomogram = mrc.data.copy()

viewer, tomo_layer = napari.imshow(
        tomogram,
        blending='translucent_no_depth',
        colormap='gray_r',
        )
```

![sphere picking](static/sphere-annotator.gif)

+++

![sphere fitting](https://teamtomo.org/_images/hiv-oversampling1.png)

from https://teamtomo.org/walkthroughs/EMPIAR-10164/geometrical-picking.html

```{code-cell} ipython3
import starfile
from scipy.spatial.transform import Rotation as R

# df is a pandas DataFrame containing table of info from STAR file
# including positions and orientations
df = starfile.read(particles_file)

# get particle positions as (n, 3) numpy array from DataFrame
zyx = df[
        ['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX']
        ].to_numpy()

pts_layer = viewer.add_points(
        zyx,
        face_color='cornflowerblue',
        size=10,
        )
```

```{code-cell} ipython3
# get particle orientations as Euler angles from DataFrame
euler_angles = df[
        ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        ].to_numpy()

# turn Euler angles into a scipy 'Rotation' object, rotate Z vectors to see
# where they point for the aligned particle
rotations = R.from_euler(
        seq='ZYZ', angles=euler_angles, degrees=True
        ).inv()
direction_xyz = rotations.apply([0, 0, 1])
direction_zyx = direction_xyz[:, ::-1]

# set up napari vectors layer data
# (n, 2, 3) array
# dim 0: batch dimension
# dim 1: first row is start point of vector,
#        second is direction vector
# dim 2: components of direction vector e.g. (z, y, x)

vectors = np.stack((zyx, direction_zyx), axis=1)

vec_layer = viewer.add_vectors(
        vectors, length=10, edge_color='orange'
        )
```

![reconstruction](https://teamtomo.org/_images/result2.png)

from https://teamtomo.org/walkthroughs/EMPIAR-10164/m.html

```{code-cell} ipython3
pts_data = pts_layer.data
```

```{code-cell} ipython3
type(pts_data)
```

## 3 – an *annotation* and *proofreading* tool

+++

### interactive segmentation of 3D cells

Semi-automated methods in Python.

```{code-cell} ipython3
viewer, (membrane_layer, nuclei_layer) = napari.imshow(
        cells,
        channel_axis=1,
        name=['membrane', 'nuclei'],
        )
```

```{code-cell} ipython3
# grab individual channels and convert to float in [0, 1]

membranes = cells[:, 0, :, :] / np.max(cells)
nuclei = cells[:, 1, :, :] / np.max(cells)
```

```{code-cell} ipython3
from skimage import filters


edges = filters.farid(nuclei)

edges_layer = viewer.add_image(
        edges,
        blending='additive',
        colormap='yellow',
        )
```

```{code-cell} ipython3
from scipy import ndimage as ndi

denoised = ndi.median_filter(nuclei, size=3)
```

```{code-cell} ipython3
li_thresholded = denoised > filters.threshold_li(denoised)

threshold_layer = viewer.add_image(
        li_thresholded,
        opacity=0.3,
        )
```

```{code-cell} ipython3
from skimage import morphology

width = 20

holes_removed = morphology.remove_small_holes(
        li_thresholded, width ** 3
        )

speckle_removed = morphology.remove_small_objects(
        holes_removed, width ** 3
        )

viewer.layers[-1].visible = False

viewer.add_image(
        speckle_removed,
        name='cleaned',
        opacity=0.3,
        );
```

```{code-cell} ipython3
from skimage import measure

labels = measure.label(speckle_removed)

viewer.layers[-1].visible = False
viewer.add_labels(
        labels,
        opacity=0.5,
        blending='translucent_no_depth'
        )
```

```{code-cell} ipython3
# Sean's solution
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

spacing = [0.29, 0.26, 0.26]
distances = ndi.distance_transform_edt(
    speckle_removed, sampling=spacing
)
dt_smoothed = filters.gaussian(distances, sigma=5)
peaks = peak_local_max(dt_smoothed, min_distance=5)

pts_layer = viewer.add_points(
        peaks,
        name="sean's points",
        size=4,
        n_dimensional=True,  # points have 3D "extent"
        )
```

```{code-cell} ipython3
points_data = pts_layer.data
points_data
```

```{code-cell} ipython3
from skimage import segmentation, util

markers = util.label_points(points_data, nuclei.shape)
markers_big = morphology.dilation(markers, morphology.ball(5))

segmented = segmentation.watershed(
        edges, markers_big, mask=speckle_removed,
        )

seg_layer = viewer.add_labels(
        segmented, blending='translucent_no_depth',
        )

viewer.layers['labels'].visible = False
```

## 4.1 – a *lazy* viewer

+++

Tribolium castaneum light sheet microscopy data from the [Cell tracking challenge](http://celltrackingchallenge.net/3d-datasets/) contributed by Akanksha Jain, MPI-CBG Dresden.

```{code-cell} ipython3
import zarr

image = zarr.open('/Users/jni/data/Fluo-N3DL-TRIF/01.ome.zarr/0/')

print(f'{image.nbytes / 1e9:.0f}GB')
```

```{code-cell} ipython3
print(image.shape)
```

```{code-cell} ipython3
print(image.chunks)
```

```{code-cell} ipython3
viewer, layer = napari.imshow(image)
```

## 4.2 – lazy annotation 🦥🎨, thank you zarr! 🧊❤️🙏

```{code-cell} ipython3
type(image), image.shape, image.nbytes / 1e9
```

```{code-cell} ipython3
viewer = napari.Viewer()
layer_multi = viewer.add_image(
        image,
        rendering='attenuated_mip',
        name='tribolium',
        contrast_limits=(1000, 6000),
        )

labels = zarr.open(
        '/Users/jni/data/Fluo-N3DL-TRIF/01-labels.zarr',
        dtype=np.uint32,
        shape=image.shape,
        write_empty_chunks=False,
        chunks=image.chunks,
        )
```

```{code-cell} ipython3
!ls -a /Users/jni/data/Fluo-N3DL-TRIF/
```

```{code-cell} ipython3
!ls -a /Users/jni/data/Fluo-N3DL-TRIF/01-labels.zarr
```

```{code-cell} ipython3
labels.shape
```

```{code-cell} ipython3
layer = viewer.add_labels(labels)
```

```{code-cell} ipython3
!ls -a /Users/jni/data/Fluo-N3DL-TRIF/01-labels.zarr
```

```{code-cell} ipython3
!rm -rf /Users/jni/data/Fluo-N3DL-TRIF/01-labels.zarr
```

## 5 – plays well with others

+++

napariboard

```{code-cell} ipython3
!python /Users/jni/projects/napariboard-proto/napariboard.py
```

## 6 – extensible with plugins

+++

## napari-ome-zarr

```{code-cell} ipython3
viewer = napari.Viewer()
```

```{code-cell} ipython3
viewer.open(
        '/Users/jni/data/Fluo-N3DL-TRIF/01.ome.zarr',
        plugin='napari-ome-zarr',
        )
```

napari-pdf-reader (I shit you not 😂)

```{code-cell} ipython3
viewer = napari.Viewer()

pdf_layer, = viewer.open('data/project_jupyter.pdf', plugin='napari-pdf-reader')
```

```{code-cell} ipython3
from skimage import color

pdfbw = color.rgb2gray(pdf_layer.data)
pdf_layer.visible = False
pdfbw_layer = viewer.add_image(
        pdfbw[:, ::2, ::2],
        scale=(2, 2, 2),
        rendering='translucent',
        )
viewer.dims.ndisplay = 3
```

```{code-cell} ipython3
from magicgui import magicgui, widgets

@magicgui(
        shear={'widget_type': widgets.FloatSlider,
               'min': 0,
               'max': pdfbw.shape[1]},
        auto_call=True,
        )
def set_layer_xz_shear(shear: float):
    pdfbw_layer.affine = [
            [1    , 0, 0, 0],
            [0    , 1, 0, 0],
            [shear, 0, 1, 0],
            [0    , 0, 0, 1],
            ]

dw = viewer.window.add_dock_widget(set_layer_xz_shear);
```

### napari-segment-everthing

```{code-cell} ipython3
viewer = napari.Viewer()
layer = viewer.open_sample('napari', 'eagle')
widg = viewer.window.add_plugin_dock_widget('napari-segment-everything')
```

### a simple widget

```{code-cell} ipython3
import napari
import numpy as np

from magicgui import magicgui
from skimage.morphology import disk
from skimage.filters.rank import mean

@magicgui(
    auto_call=True,
    threshold={"widget_type": "FloatSlider", "max": 1}
)
def filter_and_threshold(
    layer: 'napari.layers.Image',
    disk_size: int,
    threshold: float
) -> 'napari.types.LayerDataTuple':
    
    layer_tuples = []
    filtered_image = layer.data
    image_meta = {}
    
    if disk_size > 0:
        filter_disk = disk(disk_size)
        filtered_image = mean(layer.data, footprint=filter_disk)
        image_meta['name'] = 'Filtered'
        image_meta['visible'] = True
        layer_tuples.append(
            (filtered_image, image_meta, 'image')
        )
        
    if threshold > 0:
        scaled_threshold = threshold * np.max(layer.data)
        thresholded_labels = (filtered_image > scaled_threshold).astype(np.uint8)
        labels_meta = {
            'name': 'Thresholded',
            'visible': True
        }
        image_meta['visible'] = False
        layer_tuples.append(
            (thresholded_labels, labels_meta, 'labels')
        )
    return layer_tuples

viewer = napari.Viewer()
viewer.open_sample('napari', 'human_mitosis')
viewer.window.add_dock_widget(filter_and_threshold)
```
