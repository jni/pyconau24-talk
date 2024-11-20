"""To generate a Python package with all the configuration for a napari plugin,
use our copier template!

https://github.com/napari/napari-plugin-template

The template contains example implementations for different plugin contributions,
as well as comments to help you get started.
"""

import numpy as np

from magicgui import magic_factory
from skimage.morphology import disk
from skimage.filters.rank import mean

@magic_factory(
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
