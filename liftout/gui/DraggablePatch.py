
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class DraggablePatch:
    def __init__(self, patch):
        self.patch = patch
        self.press = None
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self.move_all = False
        self.movable = True
        self.center_x = None
        self.center_y = None
        self.pixel_size = None
        self.image_width = None
        self.image_height = None
        self.rotation = 0
        self.rotating = False

    def connect(self):
        self.cidpress = self.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def update_position(self):
        relative_center_x, relative_center_y = self.calculate_center()
        center_x_px = relative_center_x - self.image_width / 2
        center_y_px = relative_center_y - self.image_height / 2

        self.center_x = center_x_px * self.pixel_size
        self.center_y = - center_y_px * self.pixel_size  # centre coordinate systems

        self.width = self.patch._width * self.pixel_size
        self.height = self.patch._height * self.pixel_size
        self.rotation = self.patch.angle

    def on_press(self, event):
        # movement enabled check
        if not self.movable:
            self.press = None
            return

        # left click check
        if event.button != 1: return

        # if only moving what's under the cursor:
        if not self.move_all:
            # discard all changes if this isn't a hovered patch
            if event.inaxes != self.patch.axes: return
            contains, attrd = self.patch.contains(event)
            if not contains: return

        # get the top left corner of the patch
        x0, y0 = self.patch.xy
        self.press = x0, y0, event.xdata, event.ydata

        if self.rotation_check(event):
            self.rotating = True
        else:
            self.rotating = False
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_motion(self, event):
        """on motion we will move the rect if the mouse is over us"""
        if self.rotation_check(event):
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.OpenHandCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not self.press: return

        if self.rotating and not self.move_all:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.OpenHandCursor)
            center_x, center_y = self.calculate_center()

            angle_dx = event.xdata - center_x
            angle_dy = event.ydata - center_y
            angle = np.rad2deg(np.arctan2(angle_dy, angle_dx))
            self.rotate_about_center(angle+90)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
            x0, y0, x_intial_press, y_initial_press = self.press
            dx = event.xdata - x_intial_press
            dy = event.ydata - y_initial_press
            self.patch.set_x(x0+dx)
            self.patch.set_y(y0+dy)

        self.patch.figure.canvas.draw()

    def on_release(self, event):
        """on release we reset the press data"""
        QtWidgets.QApplication.restoreOverrideCursor()
        self.update_position()
        self.press = None
        self.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.patch.figure.canvas.mpl_disconnect(self.cidmotion)

    def toggle_move_all(self, onoff=False):
        self.move_all = onoff

    def calculate_center(self):
        x0, y0 = self.patch.xy
        w = self.patch._width/2
        h = self.patch._height/2
        theta = np.deg2rad(self.patch.angle)
        x_center = x0 + w * np.cos(theta) - h * np.sin(theta)
        y_center = y0 + w * np.sin(theta) + h * np.cos(theta)

        return x_center, y_center

    def calculate_corners(self):
        x0, y0 = self.patch.xy
        w = self.patch._width/2
        h = self.patch._height/2
        theta = np.deg2rad(self.patch.angle)

        x_shift = 2 * (w * np.cos(theta) - h * np.sin(theta))
        y_shift = 2 * (w * np.sin(theta) + h * np.cos(theta))

        top_left = x0, y0
        top_right = x0 + x_shift, y0
        bottom_left = x0, y0 + y_shift
        bottom_right = x0 + x_shift, y0 + y_shift

        return top_left, top_right, bottom_left, bottom_right

    def rotation_check(self, event):
        xpress = event.xdata
        ypress = event.ydata
        if xpress and ypress:
            ratio = 5
            abs_min = 30
            distance_check = max(min(self.patch._height / ratio, self.patch._width / ratio), abs_min)
            corners = self.calculate_corners()
            for corner in corners:
                dist = np.sqrt((xpress-corner[0]) ** 2 + (ypress-corner[1]) ** 2)
                if dist < distance_check:
                    return True
        return False

    def rotate_about_center(self, angle):
        # print(angle)
        # calculate the center position in the unrotated, original position
        old_x_center, old_y_center = self.calculate_center()

        # move the pattern to have x0, y0 at 0, 0
        self.patch.set_x(0)
        self.patch.set_y(0)

        # rotate by angle
        self.patch.angle = angle
        new_theta = np.deg2rad(self.patch.angle)

        # calculate new center position at the rotated, 0, 0 position
        w = self.patch._width/2
        h = self.patch._height/2
        new_x_center = w * np.cos(new_theta) - h * np.sin(new_theta)
        new_y_center = w * np.sin(new_theta) + h * np.cos(new_theta)

        # move the center to the 0, 0, position (can be removed as a nondebugging step)
        # self.patch.set_x(-new_x_center)
        # self.patch.set_y(-new_y_center)

        # move pattern back to centered on original center position
        self.patch.set_x(-new_x_center + old_x_center)
        self.patch.set_y(-new_y_center + old_y_center)

        self.update_position()