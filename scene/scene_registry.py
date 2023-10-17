from .primitives import SDFSpheres, SDFBoxes, SDFPlanes, SDFLine, SDFDisks, SDFTori
from .transformations import SDFOnion, SDFRounding, SDFSmoothUnion, SDFUnion, SDFAffineTransformation


def make_test_scene():
    return SDFSmoothUnion(
            sdfs=[
                SDFOnion(
                    SDFAffineTransformation(
                        orientation=[[0.9014, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]], 
                        translation=[[0.0, 0.25, .25], [0., 1.0, 0.5]],
                        sdf=SDFBoxes(
                            halfsides=[[0.1, 0.2, 0.05], [5., 5., 5.]]
                        ),
                    ),
                    radii=[[0.01]]
                ),
                SDFAffineTransformation(
                    orientation=[[1.0, 0.0, 0.0, 0.0]],
                    translation=[[0.0, 0.0, 1.0]],
                    sdf=SDFSpheres(
                        radii=[[0.5]]
                    ),
                ),
                SDFLine(
                    starts=[[-1., 1., 2.]],
                    ends=[[1., 1., 0.]],
                    radii=[[0.1,]],
                ),
                SDFAffineTransformation(
                    orientation=[[0.0, 0.5**0.5, 0.5**0.5, 0.0]],
                    translation=[[0.0, 0.5, 1.0]],
                    sdf=SDFTori(
                        radii1=[[0.5,]], 
                        radii2=[[0.1,]]
                    ),
                ),
            ], 
            blend_k=22.0,
        )

