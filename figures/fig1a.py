import fresnel
import PIL
import pdb
import numpy as onp

from jax_md import space

from catalyst.octahedron import shell_getter


# mode = "entire-shell"
# mode = "one-missing"
mode = "one-vertex"
possible_modes = set(["entire-shell", "one-missing", "one-vertex"])
assert(mode in possible_modes)


displacement_fn, shift_fn = space.free()
shell_info = shell_getter.ShellInfo(displacement_fn, shift_fn)
rb = shell_info.rigid_body
body_pos = shell_info.get_body_frame_positions(rb)
body_pos = onp.array(body_pos)
num_vertices = 6
num_patches = 4


if mode == "one-missing":
    # Process body_pos for one vertex being taken off
    vertex_to_delete = 0
    start_remove_idx = vertex_to_delete * 5
    end_remove_idx = start_remove_idx + 5
    body_pos = onp.concatenate([body_pos[:start_remove_idx], body_pos[end_remove_idx:]])
    num_vertices = 5
elif mode == "one-vertex":
    # Process body_pos for only one vertex
    body_pos = body_pos[:5]
    num_vertices = 1


assert(num_vertices * (num_patches+1) == body_pos.shape[0])
radii = list()
patch_radius = 0.5
vertex_radius = 2.0
for i in range(body_pos.shape[0]):
    if i % (num_patches + 1) == 0:
        radii.append(vertex_radius)
    else:
        radii.append(patch_radius)


device = fresnel.Device()
scene = fresnel.Scene(device)
geometry = fresnel.geometry.Sphere(scene, N=body_pos.shape[0], radius=radii)
geometry.position[:] = body_pos

geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.25,0.5,0.9]),
                                              roughness=0.8)
geometry.material.primitive_color_mix = 0.5
geometry.color[::(num_patches+1)] = fresnel.color.linear([0, 1, 0])

scene.camera = fresnel.camera.Orthographic.fit(scene)


## Change the lighting

tracer = fresnel.tracer.Path(device, w=450, h=450)

# scene.lights = fresnel.light.rembrandt(side="left")
# scene.lights = fresnel.light.rembrandt()
scene.lights = fresnel.light.butterfly()
# scene.lights = fresnel.light.ring()

tracer.sample(scene, samples=64, light_samples=10)


# Use pathtrace to account for indirect lighting
# fresnel.pathtrace(scene)
# fresnel.pathtrace(scene, light_samples=40)


# Add an outline
geometry.outline_width = 0.05
geometry.material.solid = 1.0
# geometry.material.specular = 0.0
# geometry.material.metal = 0.25
# geometry.material.roughness = 0.8
fresnel.pathtrace(scene, w=300, h=300, light_samples=40)


def check_pos(x=100, y=100, z=100):
    scene.camera.position = (x, y, z)
    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.save("single.png")
    # image.show()

# check_pos()
check_pos(50, 450, 50)
