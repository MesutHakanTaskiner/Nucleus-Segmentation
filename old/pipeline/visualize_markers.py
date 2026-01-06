from skimage import color, util


def visualize_markers(marker_image):
    vis_rgb = color.label2rgb(marker_image, bg_label=0, bg_color=(0, 0, 0))
    vis_bgr = util.img_as_ubyte(vis_rgb)[..., ::-1]
    return vis_bgr
