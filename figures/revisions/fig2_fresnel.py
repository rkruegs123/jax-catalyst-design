import fresnel
import PIL
import pdb
import numpy as onp
from pathlib import Path

from jax_md import space

from catalyst.icosahedron import complex_getter
from figures.revisions.utils import shell_patch_color, shell_vertex_color
from figures.revisions.utils import spider_base_color, spider_leg_color, spider_head_color


# mode = "diffusive-iter0"
# mode = "diffusive-iter625"
# mode = "diffusive-iter1250"
# mode = "diffusive-iter2500"
# mode = "explosive-iter625"
# mode = "explosive-iter1250"
mode = "explosive-iter2500"


num_vertices = 12
num_patches = 5

output_basedir = Path("figures/revisions/output/fig2/")
assert(output_basedir.exists())

if mode == "diffusive-iter0":
    image_name = "diffusive_iter0.png"

    init_sep_coeff = 3.5

    spider_head_radius = 1.0
    spider_base_particle_radius = 1.0

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.0
    spider_base_radius = 5.0

    target_bound = False
elif mode == "diffusive-iter625":
    image_name = "diffusive_iter625.png"

    init_sep_coeff = 2.5

    spider_head_radius = 1.860629165698884
    spider_base_particle_radius = 1.037655549912664

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 4.1470396411829755
    spider_base_radius = 4.66054602488968

    move_coeff = 1.0
    target_bound = True
elif mode == "diffusive-iter1250":
    image_name = "diffusive_iter1250.png"

    init_sep_coeff = 3.5

    spider_head_radius = 1.562486817715836
    spider_base_particle_radius = 0.9953469539862306

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 4.46539377809863
    spider_base_radius = 4.608224556875475

    move_coeff = 1.1
    target_bound = True
elif mode == "diffusive-iter2500":
    image_name = "diffusive_iter2500.png"

    init_sep_coeff = 3.5

    spider_head_radius = 1.2594821525722162
    spider_base_particle_radius = 0.952295152322053

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 4.914202511358603
    spider_base_radius = 4.603617310300306

    move_coeff = 1.1
    target_bound = True
elif mode == "explosive-iter625":
    image_name = "explositve_iter625.png"

    init_sep_coeff = 0.5

    spider_head_radius = 0.6057113604025651
    spider_base_particle_radius = 0.906171639485045

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.3963949115948475
    spider_base_radius = 4.98488275170156

    move_coeff = 2.5
    target_bound = True
elif mode == "explosive-iter1250":
    image_name = "explositve_iter1250.png"

    init_sep_coeff = 0.5

    spider_head_radius = 0.5261052681119952
    spider_base_particle_radius = 0.9178163555826525

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.479316347694025
    spider_base_radius = 4.885368173360221

    move_coeff = 2.5
    target_bound = True
elif mode == "explosive-iter2500":
    image_name = "explositve_iter2500.png"

    init_sep_coeff = 3.5

    spider_head_radius = 0.3766722753497847
    spider_base_particle_radius = 1.0932485394595859

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.639956719373849
    spider_base_radius = 4.6024671547308165

    move_coeff = 1.25
    target_bound = True
else:
    raise RuntimeError(f"Invalid mode: {mode}")


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 10



complex_info = complex_getter.ComplexInfo(
    initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
    displacement_fn=displacement_fn, shift_fn=shift_fn,
    spider_base_radius=spider_base_radius, spider_head_height=spider_head_height,
    spider_base_particle_radius=spider_base_particle_radius,
    spider_head_particle_radius=spider_head_radius,
    spider_point_mass=1.0, spider_mass_err=1e-6
)


shell_info = complex_info.shell_info
shell_rb = shell_info.rigid_body
shell_body_pos = shell_info.get_body_frame_positions(shell_rb)
shell_body_pos = onp.array(shell_body_pos)


spider_info = complex_info.spider_info
spider_rb = spider_info.rigid_body


if target_bound:
    vtx_to_spider_head = displacement_fn(shell_info.rigid_body[vertex_to_bind_idx].center,
                                         complex_info.spider_info.rigid_body[-1].center)
    vtx_to_spider_head_dist = space.distance(vtx_to_spider_head)

    # move_vtx_vec = vtx_to_spider_head * (1 - head_radius/vtx_to_spider_head_dist)
    move_vtx_vec = vtx_to_spider_head * move_coeff

    start_bind_idx = vertex_to_bind_idx * (num_patches+1)
    end_bind_idx = start_bind_idx + (num_patches+1)
    shell_body_pos[start_bind_idx:end_bind_idx] -= move_vtx_vec



assert(num_vertices * (num_patches+1) == shell_body_pos.shape[0])
radii = list()
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

tracer = fresnel.tracer.Path(device, w=1000, h=1000)

geometry_shell.outline_width = 0.05
geometry_shell.material.solid = 1.0



spider_body_pos = spider_info.get_body_frame_positions(spider_rb)[0]

num_base_particles = 5
assert(spider_body_pos.shape[0] == num_base_particles+1)
base_particle_positions = spider_body_pos[:num_base_particles]
head_pos = spider_body_pos[-1]

radii = [spider_base_particle_radius]*num_base_particles + [spider_head_radius]



geometry1 = fresnel.geometry.Sphere(scene, N=spider_body_pos.shape[0], radius=radii)
geometry1.position[:] = spider_body_pos
geometry1.material = fresnel.material.Material(color=fresnel.color.linear(spider_base_color),
                                               roughness=0.8)
geometry1.material.primitive_color_mix = 0.5
geometry1.color[-1] = fresnel.color.linear(spider_head_color)


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


image_path = str(output_basedir / image_name)
# image.show()
image.save(image_path)
