import matplotlib.pyplot as plt
import numpy as np
from patrick.utils import load_model, model_inference, detect_and_draw_lamella_and_needle, scale_invariant_coordinates, calculate_distance_between_points, parse_metadata


# select model
weights_file = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick\models\12_04_2021_10_32_23_model.pt"
model = load_model(weights_file=weights_file)

img_tiff = eb_lowres_ref_jcut.data

fname = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\patrick/test_image.tif"
#fname = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_20210419.113154\liftout000\step05_eb_highres_needle.tif"
# fname = r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_20210419.122944\liftout000\step04_eb_lowres_needle.tif"
fnames = [r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_20210419.152436\liftout000\step02_A_tiltAlign_ref_eb_lowres.tif",
r"C:\Users\Admin\MICROSCOPE\DeMarcoLab\liftout\liftout\cryo_20210419.152436\liftout000\step02_A_tiltAlign_ref_ib_lowres.tif"]

img_overlays = []
imgs = []
lamella_centres = []
for fname in fnames:
    # model inference + display
    img, rgb_mask = model_inference(model, fname)#, img=img_tiff)

    # detect and draw lamella centre, and needle tip
    (
        lamella_centre_px,
        rgb_mask_lamella,
        needle_tip_px,
        rgb_mask_needle,
        rgb_mask_combined,
    ) = detect_and_draw_lamella_and_needle(rgb_mask, cols_masks=None)
    # TODO: this col masks still needs to be extracted out

    # scale invariant coordinatess
    scaled_lamella_centre_px, scaled_needle_tip_px = scale_invariant_coordinates(
        needle_tip_px, lamella_centre_px, rgb_mask_combined
    )

    # calculate distance between features
    (
        distance,
        vertical_distance,
        horizontal_distance,
    ) = calculate_distance_between_points(scaled_lamella_centre_px, scaled_needle_tip_px)

    # prediction overlay
    img_overlay = show_overlay(img, rgb_mask_combined)

    from PIL import Image

    img_overlay_resized = Image.fromarray(img_overlay).resize((int(df["[Image].ResolutionX"].values[0]),
    int(df["[Image].ResolutionY"].values[0])))

    df = parse_metadata(fname)

    # print("Lamella Centre: ", scaled_lamella_centre_px, lamella_centre_px)
    # print("Needle Tip: ", scaled_needle_tip_px, needle_tip_px)
    # print("Distance (um):",  distance, vertical_distance, horizontal_distance)

    # # TODO: Check these values are coorect.
    # # TODO: Check these values are coorect.
    # rescaled_horizontal_distance = float(df["[Image].ResolutionY"].values[0]) * horizontal_distance
    # rescaled_vertical_distance = float(df["[Image].ResolutionX"].values[0]) * vertical_distance

    # horizontal_distance_microns = rescaled_horizontal_distance * float(df["[Scan].PixelWidth"].values[0])
    # vertical_distance_microns = rescaled_vertical_distance * float(df["[Scan].PixelHeight"].values[0])

    # print(f"horizontal distance:  {horizontal_distance_microns}m")
    # print(f"vertical distance:  {vertical_distance_microns}m")

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[1].imshow(img_overlay_resized)
    # plt.show()

    imgs.append(img)
    img_overlays.append(img_overlay_resized)
    lamella_centres.append(scaled_lamella_centre_px)

# distance from centre
lamella_x = lamella_centres[1][0]
distance_from_centre_x = 0.5 - lamella_x
distance_from_centre_metres = float(df["[Image].ResolutionX"].values[0]) * float(df["[Scan].PixelHeight"].values[0]) * distance_from_centre_x

print(f"Distance from middle: {distance_from_centre_x} ({distance_from_centre_metres}m)")
# move_relative(z=distance_from_centre_x * scaling_factor)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(imgs[0], cmap='Blues_r', alpha=1)
ax[0].imshow(img_overlays[0], cmap='Oranges_r', alpha=0.5)
ax[1].imshow(imgs[1], cmap='Blues_r', alpha=1)
ax[1].imshow(img_overlays[1], cmap='Oranges_r', alpha=0.5)
plt.show()

# for col in df.columns:
#     if "Image" in col:
#         print(col, df[col])