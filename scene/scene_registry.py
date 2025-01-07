from .primitives import (
    SDFSphere,
    SDFBox,
    SDFPlane,
    SDFLine,
    # SDFDisk,
    SDFTorus,
)
from .transformations import (
    SDFOnion,
    # SDFRounding,
    SDFSmoothUnion,
    SDFUnion,
    SDFAffineTransformation
)


def make_test_scene():
    return SDFSmoothUnion(
            sdfs=[
                SDFAffineTransformation(
                    orientation=[0.9014, 0.25, 0.25, 0.25],
                    translation=[0.0, 0.25, .25],
                    sdf=SDFOnion(
                        SDFBox(halfsides=(0.1, 0.2, 0.05)),
                        radius=0.1
                    ),
                ),
                SDFAffineTransformation(
                    orientation=[1.0, 0.0, 0.0, 0.0],
                    translation=[0.0, 0.0, 1.0],
                    sdf=SDFSphere(
                        radius=0.5,
                    ),
                ),
                SDFLine(
                    start=(-1.0, 1.0, 2.0),
                    end=(1.0, 1.0, 0.0),
                    radius=0.1,
                ),
                SDFAffineTransformation(
                    orientation=[0.0, 0.5**0.5, 0.5**0.5, 0.0],
                    translation=[0.0, 0.5, 1.0],
                    sdf=SDFTorus(
                        radius1=0.5,
                        radius2=0.1,
                    ),
                ),
            ],
            blend_k=22.0,
        )


def make_test_scene2():
    return SDFUnion(
        [
            SDFOnion(
                SDFBox(halfsides=(5.0, 5.0, 5.0)),
                radius=0.1,
            ),
            SDFUnion(
                sdfs=[
                    SDFSphere(
                        radius=0.5,
                    ),
                    SDFTorus(
                        radius1=1.0,
                        radius2=0.25,
                    ),
                    SDFLine(
                        start=(1.0, 0.0, 0.0),
                        end=(-1.0, 0.0, 0.0),
                        radius=0.1,
                    ),
                ],
                # blend_k=22.0,
            ),
        ],
    )
