from .primitives import SDFSpheres, SDFBoxes, SDFPlanes, SDFLine, SDFDisks, SDFTori
from .transformations import SDFOnion, SDFRounding, SDFSmoothUnion, SDFUnion, SDFAffineTransformation


def make_test_scene(dtype):
    return SDFSmoothUnion(
            sdfs=[
                SDFAffineTransformation(
                    orientation=[[0.9014, 0.25, 0.25, 0.25], [0.9014, -0.25, 0.25, 0.25]], 
                    translation=[[0.0, 0.25, .25], [0., 1.0, 0.5]],
                    sdf=SDFOnion(
                        SDFBoxes(
                            halfsides=[[0.1, 0.2, 0.05], [5., 5., 5.]],
                            dtype=dtype
                        ),
                        radii=[[0.05], [0.1]],
                        dtype=dtype,
                    ),
                    dtype=dtype
                ),
                SDFAffineTransformation(
                    orientation=[[1.0, 0.0, 0.0, 0.0]],
                    translation=[[0.0, 0.0, 1.0]],
                    sdf=SDFSpheres(
                        radii=[[0.5]],
                        dtype=dtype
                    ),
                    dtype=dtype,
                ),
                SDFLine(
                    starts=[[-1., 1., 2.]],
                    ends=[[1., 1., 0.]],
                    radii=[[0.1,]],
                    dtype=dtype,
                ),
                SDFAffineTransformation(
                    orientation=[[0.0, 0.5**0.5, 0.5**0.5, 0.0]],
                    translation=[[0.0, 0.5, 1.0]],
                    sdf=SDFTori(
                        radii1=[[0.5,]], 
                        radii2=[[0.1,]],
                        dtype=dtype
                    ),
                    dtype=dtype,
                ),
            ], 
            blend_k=22.0,
            dtype=dtype
        )

def make_simple_scene(dtype):
    return SDFTori(
        radii1=[[0.5,]], 
        radii2=[[0.1,]],
        dtype=dtype
    )
