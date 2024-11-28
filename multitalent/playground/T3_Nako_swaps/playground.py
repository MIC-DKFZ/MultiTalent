import numpy as np
from skimage.morphology import ball
from batchgenerators.augmentations.spatial_transformations import augment_spatial

if __name__ == '__main__':
    image_shape = (128, 128, 128)
    initial_seg = np.zeros(image_shape, dtype=np.uint8)

    # generate shpere
    ball_radius = np.random.randint(0, 32)
    ball_mask = ball(ball_radius)

    # place shpere somewhere in the image
    lbs = [np.random.randint(0, image_shape[i] - ball_mask.shape[i]) for i in range(3)]
    initial_seg[tuple([slice(i, i+j) for i, j in zip(lbs, ball_mask.shape)])] = ball_mask

    # augment_spatial does not work without an image. This image is just created to make
    # augment_spatial rum. It does not serve any purpose
    dummy_image = np.zeros_like(initial_seg)

    # let's generate 15 examples (we can then display those in a 4x4 grid with the original input.
    # We also need to add one more dummy dimension for compatibility with augment_spatial
    initial_seg = np.tile(initial_seg, (15, 1, 1, 1, 1))
    dummy_image = np.tile(dummy_image, (15, 1, 1, 1, 1))

    # now apply augmentation
    _, s = augment_spatial(dummy_image, initial_seg, image_shape, 0,
                           do_elastic_deform=True,
                           alpha=(250, 1500), # intensity of deformation. Unfortunately this is interdependent with sigma :-(
                           sigma=(3, 13), # scale of deformation. Larger values = larger areas
                           do_rotation=True,
                           do_scale=True, scale=(0.5, 2),
                           order_data=0, order_seg=0, # fastest and order is basically irrelevant here
                           random_crop=False, p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                           independent_scale_for_each_axis=True, p_independent_scale_per_axis=1)
    # get it from https://github.com/FabianIsensee/BatchViewer
    # you need to `pip install pyqtgraph==0.12.3 pyqt5` as well
    # you need to use older numpy version: `pip install numpy==1.23.3`
    from batchviewer import view_batch
    view_batch(initial_seg[0], s[:, 0])
