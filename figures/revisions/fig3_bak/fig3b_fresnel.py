import fresnel
import PIL
import pdb
import numpy as onp
from pathlib import Path

import jax.numpy as jnp
from jax_md import space

from catalyst.icosahedron_ext_rigid_tagged.complex import Complex, PENTAPOD_LEGS, BASE_LEGS
from figures.revisions.utils import shell_patch_color, shell_vertex_color
from figures.revisions.utils import spider_base_color, spider_leg_color, spider_head_color


output_basedir = Path("figures/revisions/output/fig3/b")
assert(output_basedir.exists())


# grey_vertex_color = (0.9254901960784314, 0.9411764705882353, 0.9450980392156862, 1.0)
grey_vertex_color = (1.0, 1.0, 1.0)
# grey_vertex_color = (0.9254901960784314, 0.1411764705882353, 0.1450980392156862, 1.0)
grey_patch_color = (0.8941176470588236, 0.9137254901960784, 0.9294117647058824, 1.0)

spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])



spider_base_radius = 5.0
spider_head_height = 8.0
spider_base_particle_radius = 1.0
spider_attr_particle_pos_norm = 0.5
spider_attr_site_radius = 0.75
spider_head_particle_radius = 1.0

vertex_radius = 2.0
patch_radius = 0.5
leg_radius = 0.25

num_vertices = 12
num_patches = 5


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 10

init_sep_coeff = 3.5
move_coeff = 1.2

complex_info = Complex(
    initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
    displacement_fn=displacement_fn, shift_fn=shift_fn,
    spider_base_radius=spider_base_radius,
    spider_head_height=spider_head_height,
    spider_base_particle_radius=spider_base_particle_radius,
    spider_attr_particle_pos_norm=spider_attr_particle_pos_norm,
    spider_attr_site_radius=spider_attr_site_radius,
    spider_head_particle_radius=spider_head_particle_radius,
    spider_point_mass=1.0, spider_mass_err=1e-6,
    spider_bond_idxs=spider_bond_idxs
)


shell_info = complex_info.shell_info
shell_rb = shell_info.rigid_body
shell_body_pos = shell_info.get_body_frame_positions(shell_rb)
shell_body_pos = onp.array(shell_body_pos)

spider_info = complex_info.spider_info
spider_rb = spider_info.rigid_body
spider_body_pos = spider_info.get_body_frame_positions(spider_rb)[0]


assert(num_vertices * (num_patches+1) == shell_body_pos.shape[0])
radii = [vertex_radius] + [patch_radius] * num_patches


device = fresnel.Device()
scene = fresnel.Scene(device)
# geometry_shell = fresnel.geometry.Sphere(scene, N=shell_body_pos.shape[0], radius=radii)
# geometry_shell.position[:] = shell_body_pos
geometry_shell = fresnel.geometry.Sphere(scene, N=num_patches+1, radius=radii)
geometry_shell.position[:] = shell_body_pos[vertex_to_bind_idx*6:vertex_to_bind_idx*6+6]

geometry_shell.material = fresnel.material.Material(color=fresnel.color.linear(shell_patch_color),
                                                    roughness=0.8)
geometry_shell.material.primitive_color_mix = 0.5
# geometry_shell.material.primitive_color_mix = 1.0
geometry_shell.color[::(num_patches+1)] = fresnel.color.linear(shell_vertex_color)

tracer = fresnel.tracer.Path(device, w=1000, h=1000)

geometry_shell.outline_width = 0.05
geometry_shell.material.solid = 1.0


num_base_particles = 5
assert(spider_body_pos.shape[0] == (num_base_particles*2)+1)
base_particle_positions = spider_body_pos[:num_base_particles]
attr_site_particle_positions = spider_body_pos[num_base_particles:num_base_particles*2]
head_pos = spider_body_pos[-1]

radii = [spider_base_particle_radius]*num_base_particles + [spider_attr_site_radius]*num_base_particles + [spider_head_particle_radius]


geometry1 = fresnel.geometry.Sphere(scene, N=spider_body_pos.shape[0], radius=radii)
geometry1.position[:] = spider_body_pos
geometry1.material = fresnel.material.Material(color=fresnel.color.linear(spider_base_color),
                                               roughness=0.8)
geometry1.material.primitive_color_mix = 0.5
# geometry1.color[-1] = fresnel.color.linear(spider_head_color)
geometry1.color[num_base_particles:2*num_base_particles] = fresnel.color.linear(spider_head_color)


geometry2 = fresnel.geometry.Cylinder(scene, N=10)
geometry2.material = fresnel.material.Material(color=fresnel.color.linear(spider_leg_color),
                                               roughness=0.8)

geometry2.points[:] = [[base_particle_positions[0], base_particle_positions[1]],
                       [base_particle_positions[1], base_particle_positions[2]],
                       [base_particle_positions[2], base_particle_positions[3]],
                       [base_particle_positions[3], base_particle_positions[4]],
                       [base_particle_positions[4], base_particle_positions[0]],
                       [base_particle_positions[0], head_pos],
                       [base_particle_positions[1], head_pos],
                       [base_particle_positions[2], head_pos],
                       [base_particle_positions[3], head_pos],
                       [base_particle_positions[4], head_pos],
]
geometry2.radius[:] = [leg_radius] * 10


scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.butterfly()

tracer.sample(scene, samples=64, light_samples=10)

fresnel.pathtrace(scene, w=1000, h=1000, light_samples=32)


# scene.camera.position = (50, 450, 50)
out = fresnel.preview(scene, h=370*2, w=600*2)
image = PIL.Image.fromarray(out[:], mode='RGBA')


# image.show()
fname = "spider-and-vertex.png"
image.save(str(output_basedir / fname))
