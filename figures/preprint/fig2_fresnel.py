import fresnel
import PIL
import pdb
import numpy as onp

from jax_md import space

from catalyst.octahedron import complex_getter
from figures.preprint.utils import shell_patch_color, shell_vertex_color
from figures.preprint.utils import spider_base_color, spider_leg_color, spider_head_color


mode = "diffusive-init"
# mode = "diffusive-fin"
# mode = "abduction-mid"
# mode = "abduction-fin"

num_vertices = 6
num_patches = 4


if mode == "diffusive-init":
    image_path = "diffusive-init.png"

    init_sep_coeff = 3.5

    spider_head_radius = 1.0
    spider_base_particle_radius = 1.0

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 4.0
    spider_base_radius = 5.0

    target_bound = False
elif mode == "diffusive-fin":
    image_path = "diffusive-fin.png"

    init_sep_coeff = 3.5

    spider_head_radius = 1.585585329339908
    spider_base_particle_radius = 1.8063836054758438

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 3.9005051204550534
    spider_base_radius = 3.731331462052045

    move_coeff = 1.1
    target_bound = True
elif mode == "abduction-mid":
    image_path = "abduction-mid.png"

    init_sep_coeff = 0.5

    # spider_head_radius = 0.16020497320706556 # true value
    spider_head_radius = 0.16020497320706556*3 # for visualization
    spider_base_particle_radius = 1.3808429165344556

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.339756092469683
    spider_base_radius = 4.273203766405066

    move_coeff = 2.5
    target_bound = True
elif mode == "abduction-fin":
    image_path = "abduction-fin.png"

    init_sep_coeff = 3.5

    # spider_head_radius = 0.19001464409956886 # true value
    spider_head_radius = 0.19001464409956886*3 # for visualization
    spider_base_particle_radius = 1.9033441902705621

    patch_radius = 0.5
    vertex_radius = 2.0
    leg_radius = 0.25

    spider_head_height = 5.302795963030974
    spider_base_radius = 3.8489440283911924

    move_coeff = 1.2
    target_bound = True
else:
    raise RuntimeError(f"Invalid mode: {mode}")


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 0



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

    start_bind_idx = vertex_to_bind_idx * 5
    end_bind_idx = start_bind_idx + 5
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

num_base_particles = 4
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


geometry2 = fresnel.geometry.Cylinder(scene, N=8)
geometry2.material = fresnel.material.Material(color=fresnel.color.linear(spider_leg_color),
                                               roughness=0.8)

geometry2.points[:] = [[base_particle_positions[0], base_particle_positions[1]],
                       [base_particle_positions[1], base_particle_positions[2]],
                       [base_particle_positions[2], base_particle_positions[3]],
                       [base_particle_positions[3], base_particle_positions[0]],
                       [base_particle_positions[0], head_pos],
                       [base_particle_positions[1], head_pos],
                       [base_particle_positions[2], head_pos],
                       [base_particle_positions[3], head_pos],

]
geometry2.radius[:] = [leg_radius] * 8


scene.camera = fresnel.camera.Orthographic.fit(scene)
scene.lights = fresnel.light.butterfly()

tracer.sample(scene, samples=64, light_samples=10)

fresnel.pathtrace(scene, w=1000, h=1000, light_samples=32)


scene.camera.position = (50, 450, 50)
out = fresnel.preview(scene, h=370*2, w=600*2)
image = PIL.Image.fromarray(out[:], mode='RGBA')


# image.show()
image.save(image_path)
