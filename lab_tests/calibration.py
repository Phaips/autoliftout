

def ensure_eucentricity(microscope):
    pass
    # Input: requires a good electron & ion beam image as input
    # 0. Ensure the scan rotation is ZERO!
    # 1. User must identify location in SEM image
    # 2. Center location in SEM image field of view
    # 3. Take new ion beam image
    # 4. User must identify the same location on the ion beam image
    # 5. Measure distance between location and the center of the ion beam image FOV. This is delta_y_FIB.
    # 6. Calculate how much to move relative in z, according to this equation:
    #     delta_z = (0.5 * delta_y) / sin(tilt)
    #     (make sure this works for different pre-tilts)
    # 7. Move relative in z.
    # 8. Take new SEM image.
    # 9. Figure out where the location has moved to.
    # 10. Re-center location in SEM image.
    # 11. Calculate how much to move relative in y, accordinate to this equation:
    #        delta_y_sem = -0.5 * delta_y_FIB
    # 12. Move relative in y.
    # DONE!
    # Take new images in SEM and FIB to double check you are correct.
