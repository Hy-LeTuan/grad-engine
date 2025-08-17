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
