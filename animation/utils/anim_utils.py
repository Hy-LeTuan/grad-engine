from manim import *


def pulse(node: VGroup, flash_color=ORANGE, base_color=BLUE, scale_factor=1.1, run_time=0.8):
    return Succession(
        AnimationGroup(
            # ApplyMethod(node.set_fill, flash_color, opacity=1),
            # Transform(circle, circle.copy().set_fill(flash_color, opacity=1)),
            ApplyMethod(node.scale, scale_factor),
            run_time=run_time / 2
        ),
        AnimationGroup(
            # ApplyMethod(node.set_fill, base_color, opacity=1),
            # Transform(circle, circle.copy().set_fill(base_color, opacity=1)),
            ApplyMethod(node.scale, 1 / scale_factor),
            run_time=run_time / 2
        )
    )


def get_darker_hex(hex_color: str, factor: float = 0.8) -> str:
    """
    Takes a hexadecimal color string and returns a new hex string
    that is a darker version of the original.

    Args:
        hex_color (str): The original color in hexadecimal format (e.g., "#FF0000").
        factor (float): The darkening factor. A value of 1.0 is no change,
                        and a value of 0.0 is black. Default is 0.7.

    Returns:
        str: The new, darker hexadecimal color string.
    """
    if not hex_color.startswith('#'):
        hex_color = '#' + hex_color

    rgb_tuple = hex_to_rgb(hex_color)

    darker_rgb = tuple(c * factor for c in rgb_tuple)

    darker_hex = rgb_to_hex(darker_rgb)

    return darker_hex
