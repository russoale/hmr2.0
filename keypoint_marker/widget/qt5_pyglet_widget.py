"""
This module contains the definiton of a pyglet widget for a
PySide application: QPygletWidget

It also provides a basic usage example.
"""
import sys

import pyglet

pyglet.options['shadow_window'] = False
pyglet.options['debug_gl'] = False

from pyglet import gl

from PySide2 import QtCore, QtOpenGL
from PySide2.QtWidgets import QApplication

from widget.gl_helper import gl_set_background, gl_enable_depth, gl_enable_color_material, gl_enable_blending, \
    gl_enable_smooth_lines, gl_enable_lighting, gl_resize


#######################
#   Pyglet Context    #
#######################


class ObjectSpace(object):
    """ Object space mocker """

    def __init__(self):
        # Textures and buffers scheduled for deletion the next time this
        # object space is active.
        self._doomed_textures = []
        self._doomed_buffers = []


class Context(object):
    """
    pyglet.gl.Context mocker. This is used to make pyglet believe that a valid
    context has already been setup. (Qt takes care of creating the open gl
    context)

    _Most of the methods are empty, there is just the minimum required to make
    it look like a duck..._
    """
    # define the same class attribute as pyglet.gl.Context
    CONTEXT_SHARE_NONE = None
    CONTEXT_SHARE_EXISTING = 1
    _gl_begin = False
    _info = None
    _nscontext = None
    _workaround_checks = [
        ('_workaround_unpack_row_length', lambda info: info.get_renderer() == 'GDI Generic'),
        ('_workaround_vbo', lambda info: info.get_renderer().startswith('ATI Radeon X')),
        ('_workaround_vbo_finish',
         lambda info: ('ATI' in info.get_renderer() and info.have_version(1, 5) and sys.platform == 'darwin'))]

    def __init__(self, context_share=None):
        """
        Setup workaround attr and object spaces (again to mock what is done in
        pyglet context)
        """
        self.object_space = ObjectSpace()
        for attr, check in self._workaround_checks:
            setattr(self, attr, None)

    def __repr__(self):
        return '%s()' % self.__class__.__name__

    def set_current(self):
        pass

    def destroy(self):
        pass

    def delete_texture(self, texture_id):
        pass

    def delete_buffer(self, buffer_id):
        pass


class Qt5PygletWidget(QtOpenGL.QGLWidget):
    """
    A simple qt5 with pyglet widget.

    User can subclass this widget and implement the following methods:
        - on_init: called when open gl has been initialised
        - on_resize: called when resizeGL is executed
        - on_draw: called when paintGL is executed
        - on_key_pressed: called when key is pressed

    Parameters
    -----------
    background : None or np.array(4,) uint8
          Color for background in rgba
    z_near : float, default=0.01
          what is the closest to the camera it should render
    z_far : float, default=1000.0
          what is the farthest from the camera it should render
    """

    def __init__(self, parent=None):
        super(Qt5PygletWidget, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # noinspection PyAttributeOutsideInit
    def initialize_widget(self, lights=None, background=None, fov=[60., 45.], z_near=0.01, z_far=1000.0):
        self.lights = lights
        self.background = background
        self.fov = fov
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.pixel_ratio = self.devicePixelRatio()

    def on_init(self):
        pass

    def on_resize(self, w, h):
        pass

    def on_draw(self):
        pass

    def on_key_pressed(self, event):
        pass

    def on_mouse_press(self, x, y, buttons, modifiers):
        pass

    def on_mouse_drag(self, x, y):
        pass

    def on_mouse_double_click(self, x, y):
        pass

    def on_mouse_scrolling(self, dy):
        pass

    def initializeGL(self):
        """ Initialises open gl with mock context for pyglet"""
        gl.current_context = Context()

        gl_set_background(self.background)
        gl_enable_depth(self.z_near, self.z_far)
        gl_enable_color_material()
        gl_enable_blending()
        gl_enable_smooth_lines()
        gl_enable_lighting(self.lights)

        self.on_init()

    def resizeGL(self, w, h):
        """
        Resizes the gl camera to match the widget size.
        """
        gl_resize(w, h, self.fov, self.z_near, self.z_far)

        self.on_resize(w, h)

    def paintGL(self):
        """
        Clears the back buffer than calls the on_draw method
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        self.on_draw()

    def keyPressEvent(self, event):
        """
        Call appropriate functions given key presses.
        """
        self.on_key_pressed(event)
        self.updateGL()

    def mousePressEvent(self, event):
        """
        Call appropriate functions given mouse key presses.
        """
        x, y = event.x() * self.pixel_ratio, event.y() * self.pixel_ratio
        buttons = event.buttons()
        modifiers = QApplication.keyboardModifiers()
        self.on_mouse_press(x, y, buttons, modifiers)
        self.updateGL()

    def mouseMoveEvent(self, event):
        """
        Call appropriate function if mouse is moving
        """
        x, y = event.x() * self.pixel_ratio, event.y() * self.pixel_ratio
        self.on_mouse_drag(x, y)
        self.updateGL()

    def mouseDoubleClickEvent(self, event):
        """
        Call appropriate function if mouse double click
        """
        x, y = event.x() * self.pixel_ratio, event.y() * self.pixel_ratio
        self.on_mouse_double_click(x, y)
        self.updateGL()

    def wheelEvent(self, event):
        """
        Call appropriate function if mouse is scrolling
        """
        dy = event.pixelDelta().y() * self.pixel_ratio / 5.
        self.on_mouse_scrolling(dy)
        self.updateGL()
