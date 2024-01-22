from .primitives import SDFSphere, SDFBox, SDFPlane, SDFLine, SDFDisk, SDFTorus
from .transformations import SDFOnion, SDFRounding, SDFSmoothUnion, SDFUnion, SDFAffineTransformation


# def make_test_scene():
#     return SDFSmoothUnion(
#             sdfs=[
#                 SDFAffineTransformation(
#                     orientation=[[0.9014, 0.25, 0.25, 0.25], [0.9014, -0.25, 0.25, 0.25]], 
#                     translation=[[0.0, 0.25, .25], [0., 1.0, 0.5]],
#                     sdf=SDFOnion(
#                         SDFBoxes(
#                             halfsides=[[0.1, 0.2, 0.05], [5., 5., 5.]],
#                         ),
#                         radii=[[0.1], [0.2]],
#                     ),
#                 ),
#                 SDFAffineTransformation(
#                     orientation=[[1.0, 0.0, 0.0, 0.0]],
#                     translation=[[0.0, 0.0, 1.0]],
#                     sdf=SDFSpheres(
#                         radii=[[0.5]],
#                     ),
#                 ),
#                 SDFLine(
#                     starts=[[-1., 1., 2.]],
#                     ends=[[1., 1., 0.]],
#                     radii=[[0.1,]],
#                 ),
#                 SDFAffineTransformation(
#                     orientation=[[0.0, 0.5**0.5, 0.5**0.5, 0.0]],
#                     translation=[[0.0, 0.5, 1.0]],
#                     sdf=SDFTori(
#                         radii1=[[0.5,]], 
#                         radii2=[[0.1,]],
#                     ),
#                 ),
#             ], 
#             blend_k=22.0,
#         )

def make_test_scene2():
    return SDFUnion(
        [
            SDFOnion(
                SDFBox(
                    halfsides=[5., 5., 5.],
                ),
                radii=[0.1],
            ),
            SDFUnion(
                sdfs=[
                    SDFSphere(
                        radii=[0.5,],
                    ),
                    SDFTorus(
                        radii1=[1.0,], 
                        radii2=[0.25,],
                    ),
                    SDFLine(
                        starts=[1., 0., 0.,],
                        ends=[-1., 0., 0.,],
                        radii=[0.1,],
                    ),
                ],
                # blend_k=22.0,
            )
        ],
    )

# def make_simple_scene():
#     return SDFTori(
#         radii1=[[0.5,]], 
#         radii2=[[0.1,]],
#     )
