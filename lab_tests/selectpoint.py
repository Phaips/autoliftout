import napari
import numpy as np


def select_point(image):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image)

    layers = viewer.layers
    point_layers = [layer for layer in layers if isinstance(layer, napari.layers.points.points.Points)]
    assert len(point_layers) == 1
    points = point_layers[0]
    assert points.data.shape == (1, 2)
    points.data
    return points.data[0]  # y, x format


def main():
    image = np.random.random((100, 100))
    point_coords = select_point(image)
    print(point_coords)


if __name__=="__main__":
    main()
