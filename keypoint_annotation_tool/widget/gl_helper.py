import numpy as np
from pyglet import gl


#######################
#    OpenGl Helper    #
#######################

def gl_float(array):
    array = np.array(array)
    return (gl.GLfloat * len(array))(*array)


def gl_set_background(background):
    # if user passed a background color use it
    if background is None:
        background = np.ones(4)  # default background color is white
    else:
        background = background.astype(np.float64) / 255.0  # convert to 0.0-1.0 float

    gl.glClearColor(*background)


def gl_enable_depth(z_near, z_far):
    gl.glDepthRange(z_near, z_far)
    gl.glClearDepth(1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glEnable(gl.GL_CULL_FACE)


def gl_enable_color_material():
    # do some openGL things
    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
    gl.glEnable(gl.GL_COLOR_MATERIAL)
    gl.glShadeModel(gl.GL_SMOOTH)

    gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, gl_float([0.192250, 0.192250, 0.192250]))
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, gl_float([0.507540, 0.507540, 0.507540]))
    gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, gl_float([.5082730, .5082730, .5082730]))
    gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, .4 * 128.0)


def gl_enable_blending():
    gl.glEnable(gl.GL_BLEND)  # enable blending for transparency
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


def gl_enable_smooth_lines():
    gl.glEnable(gl.GL_LINE_SMOOTH)  # make the lines from Path3D objects less ugly
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
    gl.glLineWidth(0.5)  # set the width of lines to 2 pixels
    gl.glPointSize(12)  # set PointCloud markers to 8 pixels in size
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)  # set
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glEnable(gl.GL_BLEND)


def gl_enable_lighting(lights):
    gl.glEnable(gl.GL_LIGHTING)

    for i, light in lights.items():
        gl_color = gl_float(light[1])
        gl_position = gl_float(light[0][:3, 3])
        light_n = eval('gl.GL_LIGHT{}'.format(i))

        gl.glEnable(light_n)
        gl.glLightfv(light_n, gl.GL_POSITION, gl_position)
        gl.glLightfv(light_n, gl.GL_SPECULAR, gl_color)
        gl.glLightfv(light_n, gl.GL_DIFFUSE, gl_color)
        gl.glLightfv(light_n, gl.GL_AMBIENT, gl_color)


def gl_resize(w, h, fov, z_near, z_far):
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()

    aspect_ratio = float(w) / float(h)
    gl.gluPerspective(fov[1], aspect_ratio, z_near, z_far)
    gl.glMatrixMode(gl.GL_MODELVIEW)


def gl_set_culling(on):
    # backface culling on or off
    if on:
        gl.glEnable(gl.GL_CULL_FACE)
    else:
        gl.glDisable(gl.GL_CULL_FACE)


def gl_set_polygon_mode(on):
    # view mode, filled vs wireframe
    if on:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    else:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
