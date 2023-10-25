import fresnel
import PIL
import pdb
import numpy as onp

from jax_md import space

from catalyst.octahedron import complex_getter
from figures.utils import spider_base_color, spider_leg_color, spider_head_color


shell_vertex_color = (0.9254901960784314, 0.9411764705882353, 0.9450980392156862, 1.0)
shell_patch_color = (0.8941176470588236, 0.9137254901960784, 0.9294117647058824, 1.0)


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 0


init_sep_coeff = 1.5
head_radius = 1.0

head_height = 6.5
base_radius = 6.0
base_particle_radius = 1.0

complex_info = complex_getter.ComplexInfo(
    initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
    displacement_fn=displacement_fn, shift_fn=shift_fn,
    spider_base_radius=base_radius, spider_head_height=head_height,
    spider_base_particle_radius=base_particle_radius, spider_head_particle_radius=head_radius,
    spider_point_mass=1.0, spider_mass_err=1e-6
)




shell_info = complex_info.shell_info

shell_rb = shell_info.rigid_body
shell_body_pos = shell_info.get_body_frame_positions(shell_rb)
shell_body_pos = onp.array(shell_body_pos)
num_vertices = 6
num_patches = 4


assert(num_vertices * (num_patches+1) == shell_body_pos.shape[0])
radii = list()
patch_radius = 0.5
vertex_radius = 2.0
for i in range(shell_body_pos.shape[0]):
    if i % (num_patches + 1) == 0:
        radii.append(vertex_radius)
    else:
        radii.append(patch_radius)


device = fresnel.Device()
scene = fresnel.Scene(device)
geometry_shell = fresnel.geometry.Sphere(scene, N=shell_body_pos.shape[0], radius=radii)
geometry_shell.position[:] = shell_body_pos

geometry_shell.material = fresnel.material.Material(color=fresnel.color.linear(shell_patch_color),
                                                    roughness=0.8)
geometry_shell.material.primitive_color_mix = 0.5
geometry_shell.color[::(num_patches+1)] = fresnel.color.linear(shell_vertex_color)

tracer = fresnel.tracer.Path(device, w=450, h=450)

geometry_shell.outline_width = 0.05
geometry_shell.material.solid = 1.0





spider_info = complex_info.spider_info
spider_rb = spider_info.rigid_body
spider_body_pos = spider_info.get_body_frame_positions(spider_rb)[0]

num_legs = 4
assert(spider_body_pos.shape[0] == num_legs+1)
leg_positions = spider_body_pos[:num_legs]
head_pos = spider_body_pos[-1]

radii = [base_particle_radius]*num_legs + [head_radius]


geometry1 = fresnel.geometry.Sphere(scene, N=spider_body_pos.shape[0], radius=radii)
geometry1.position[:] = spider_body_pos
geometry1.material = fresnel.material.Material(color=fresnel.color.linear(spider_base_color),
                                               roughness=0.8)
geometry1.material.primitive_color_mix = 0.5
geometry1.color[-1] = fresnel.color.linear(spider_head_color)


geometry2 = fresnel.geometry.Cylinder(scene, N=8)
geometry2.material = fresnel.material.Material(color=fresnel.color.linear(spider_leg_color),
                                               roughness=0.8)

geometry2.points[:] = [[leg_positions[0], leg_positions[1]],
                       [leg_positions[1], leg_positions[2]],
                       [leg_positions[2], leg_positions[3]],
                       [leg_positions[3], leg_positions[0]],
                       [leg_positions[0], head_pos],
                       [leg_positions[1], head_pos],
                       [leg_positions[2], head_pos],
                       [leg_positions[3], head_pos],

]
leg_radius = 0.2
geometry2.radius[:] = [leg_radius] * 8


scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.butterfly()

tracer.sample(scene, samples=64, light_samples=10)

fresnel.pathtrace(scene, w=300, h=300, light_samples=40)

default_height = scene.camera.height
default_look_at = scene.camera.look_at

def check_pos(x, y, z, height=default_height, look_at=default_look_at):

    scene.camera.height = height
    scene.camera.look_at = look_at

    # scene.camera.position = (50, 450, 50)
    scene.camera.position = (x, y, z)
    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.show()


check_pos(50, 450, 50, 15)
pdb.set_trace()

# check_pos(125, 450, 100, 15, default_look_at + onp.array([0.0, 3.0, 0.0]))

check_pos(50, 450, 50, 15, default_look_at + onp.array([0.0, 15.5, 0.0]))

print("done")
