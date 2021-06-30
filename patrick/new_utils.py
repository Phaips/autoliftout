def take_electron_and_ion_reference_images(
    microscope,
    hor_field_width=50e-6,
    image_settings=None,
    __autocontrast=True,
    eb_brightness=None,
    ib_brightness=None,
    eb_contrast=None,
    ib_contrast=None,
    save=False,
    save_label="_default",
):
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        GrabFrameSettings,
    )

    # image_settings = GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6)
    #############

    # Take reference images with lower resolution, wider field of view
    microscope.beams.electron_beam.horizontal_field_width.value = hor_field_width
    microscope.beams.ion_beam.horizontal_field_width.value = hor_field_width
    microscope.imaging.set_active_view(1)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ELECTRON)
    eb_reference = new_electron_image(
        microscope, image_settings, eb_brightness, eb_contrast
    )
    microscope.imaging.set_active_view(2)
    if __autocontrast:
        autocontrast(microscope, beam_type=BeamType.ION)
    ib_reference = new_ion_image(microscope, image_settings, ib_brightness, ib_contrast)

    # save images
    if save:
        storage.SaveImage(eb_image, id=save_label + "_eb")
        storage.SaveImage(ib_image, id=save_label + "_ib")

    return eb_reference, ib_reference

# needle_eb_lowres_with_lamella_shifted, needle_ib_lowres_with_lamella_shifted = take_electron_and_ion_reference_images(microscope, hor_field_width=150e-6,
#                 image_settings=image_settings, save=True, save_label="test_image")



