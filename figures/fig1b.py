import fresnel
import PIL
import pdb
import numpy as onp

from jax_md import space

from catalyst.octahedron import complex_getter
from figures.utils import shell_patch_color, shell_vertex_color
from figures.utils import spider_base_color, spider_leg_color, spider_head_color


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 0


mode = "just-entire-shell"
# mode = "complex-bound"
# mode = "just-spider"
# mode = "complex-unbound"

init_sep_coeff = 3.5
head_radius = 1.0

complex_info = complex_getter.ComplexInfo(
    initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
    displacement_fn=displacement_fn, shift_fn=shift_fn,
    spider_base_radius=5.0, spider_head_height=4.0,
    spider_base_particle_radius=0.5, spider_head_particle_radius=head_radius,
    spider_point_mass=1.0, spider_mass_err=1e-6
)

if mode == "just-entire-shell":

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
    geometry = fresnel.geometry.Sphere(scene, N=shell_body_pos.shape[0], radius=radii)
    geometry.position[:] = shell_body_pos

    geometry.material = fresnel.material.Material(color=fresnel.color.linear(shell_patch_color),
                                                  roughness=0.8)
    geometry.material.primitive_color_mix = 0.5
    geometry.color[::(num_patches+1)] = fresnel.color.linear(shell_vertex_color)

    scene.camera = fresnel.camera.Orthographic.fit(scene)

    tracer = fresnel.tracer.Path(device, w=450, h=450)

    scene.lights = fresnel.light.butterfly()

    tracer.sample(scene, samples=64, light_samples=10)


    geometry.outline_width = 0.05
    geometry.material.solid = 1.0

    fresnel.pathtrace(scene, w=300, h=300, light_samples=40)

    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.show()


elif mode == "just-spider":


    device = fresnel.Device()
    scene = fresnel.Scene(device)


    spider_info = complex_info.spider_info
    spider_rb = spider_info.rigid_body
    spider_body_pos = spider_info.get_body_frame_positions(spider_rb)[0]

    num_legs = 4
    assert(spider_body_pos.shape[0] == num_legs+1)
    leg_positions = spider_body_pos[:num_legs]
    head_pos = spider_body_pos[-1]

    leg_radius = 1.0
    radii = [leg_radius]*num_legs + [head_radius]


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
    geometry2.radius[:] = [0.25] * 8


    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.butterfly()

    tracer = fresnel.tracer.Path(device, w=450, h=450)

    tracer.sample(scene, samples=64, light_samples=10)

    fresnel.pathtrace(scene, w=300, h=300, light_samples=40)


    scene.camera.position = (50, 450, 50)
    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.show()


elif mode == "complex-bound":

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

    leg_radius = 1.0
    radii = [leg_radius]*num_legs + [head_radius]


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
    geometry2.radius[:] = [0.25] * 8


    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.butterfly()

    tracer.sample(scene, samples=64, light_samples=10)

    fresnel.pathtrace(scene, w=300, h=300, light_samples=40)


    scene.camera.position = (50, 450, 50)
    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.show()

elif mode == "complex-unbound":


    shell_info = complex_info.shell_info
    shell_rb = shell_info.rigid_body
    shell_body_pos = shell_info.get_body_frame_positions(shell_rb)
    shell_body_pos = onp.array(shell_body_pos)
    patch_radius = 0.5
    vertex_radius = 2.0


    spider_info = complex_info.spider_info
    spider_rb = spider_info.rigid_body
    leg_radius = 1.0


    vtx_to_spider_head = displacement_fn(shell_info.rigid_body[vertex_to_bind_idx].center,
                                         complex_info.spider_info.rigid_body[-1].center)
    vtx_to_spider_head_dist = space.distance(vtx_to_spider_head)

    # move_vtx_vec = vtx_to_spider_head * (1 - head_radius/vtx_to_spider_head_dist)
    move_vtx_vec = vtx_to_spider_head * 1.2

    start_bind_idx = vertex_to_bind_idx * 5
    end_bind_idx = start_bind_idx + 5
    shell_body_pos[start_bind_idx:end_bind_idx] -= move_vtx_vec


    num_vertices = 6
    num_patches = 4


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

    tracer = fresnel.tracer.Path(device, w=450, h=450)

    geometry_shell.outline_width = 0.05
    geometry_shell.material.solid = 1.0



    spider_body_pos = spider_info.get_body_frame_positions(spider_rb)[0]

    num_legs = 4
    assert(spider_body_pos.shape[0] == num_legs+1)
    leg_positions = spider_body_pos[:num_legs]
    head_pos = spider_body_pos[-1]

    radii = [leg_radius]*num_legs + [head_radius]


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
    geometry2.radius[:] = [0.25] * 8


    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.butterfly()

    tracer.sample(scene, samples=64, light_samples=10)

    fresnel.pathtrace(scene, w=300, h=300, light_samples=40)


    scene.camera.position = (50, 450, 50)
    out = fresnel.preview(scene)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    image.show()
