import fresnel
import PIL
import pdb
import numpy as onp

from jax_md import space

from catalyst.octahedron import complex_getter
from figures.preprint.utils import shell_patch_color, shell_vertex_color
from catalyst.octahedron import shell_getter
from figures.preprint.utils import spider_base_color, spider_leg_color, spider_head_color


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 0


# mode = "initial-fat-spider"
mode = "final-good-spider"
# mode = "just-entire-shell"


if mode == "initial-fat-spider":

    init_sep_coeff = 3.5
    head_radius = 0.9
    base_radius = 6.0
    head_height = 3.75
    base_particle_radius = 1.25

    complex_info = complex_getter.ComplexInfo(
        initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=base_radius, spider_head_height=head_height,
        spider_base_particle_radius=base_particle_radius, spider_head_particle_radius=head_radius,
        spider_point_mass=1.0, spider_mass_err=1e-6
    )


    device = fresnel.Device()
    scene = fresnel.Scene(device)


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
    leg_radius = 0.25
    geometry2.radius[:] = [leg_radius] * 8


    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.butterfly()

    tracer = fresnel.tracer.Path(device, w=1500, h=1500)

    tracer.sample(scene, samples=32, light_samples=64)

    fresnel.pathtrace(scene, w=1000, h=1000, light_samples=64)


    def render_pos(x, y, z):
        scene.camera.position = (x, y, z)
        out = fresnel.preview(scene, w=1500, h=1500)
        image = PIL.Image.fromarray(out[:], mode='RGBA')

        # image.show()
        image.save("just_spider_fat.png")
    render_pos(50, 250, 50)
    pdb.set_trace()
    print("done")

elif mode == "final-good-spider":

    init_sep_coeff = 3.5
    head_radius = 1.0
    base_radius = 5.0
    head_height = 7.0
    base_particle_radius = 0.8

    complex_info = complex_getter.ComplexInfo(
        initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=base_radius, spider_head_height=head_height,
        spider_base_particle_radius=base_particle_radius, spider_head_particle_radius=head_radius,
        spider_point_mass=1.0, spider_mass_err=1e-6
    )


    device = fresnel.Device()
    scene = fresnel.Scene(device)


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
    leg_radius = 0.25
    geometry2.radius[:] = [leg_radius] * 8


    scene.camera = fresnel.camera.Orthographic.fit(scene)
    scene.lights = fresnel.light.butterfly()

    tracer = fresnel.tracer.Path(device, w=1500, h=1500)

    tracer.sample(scene, samples=32, light_samples=64)

    fresnel.pathtrace(scene, w=1000, h=1000, light_samples=64)


    def render_pos(x, y, z):
        scene.camera.position = (x, y, z)
        out = fresnel.preview(scene, w=1500, h=1500)
        image = PIL.Image.fromarray(out[:], mode='RGBA')

        # image.show()
        image.save("just_spider_good.png")
    render_pos(50, 250, 50)
    pdb.set_trace()
    print("done")

elif mode == "just-entire-shell":


    shell_info = shell_getter.ShellInfo(displacement_fn, shift_fn)
    rb = shell_info.rigid_body
    body_pos = shell_info.get_body_frame_positions(rb)
    body_pos = onp.array(body_pos)
    num_vertices = 6
    num_patches = 4

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

    geometry.material = fresnel.material.Material(color=fresnel.color.linear(shell_patch_color),
                                                  roughness=0.8)
    geometry.material.primitive_color_mix = 0.5
    geometry.color[::(num_patches+1)] = fresnel.color.linear(shell_vertex_color)

    scene.camera = fresnel.camera.Orthographic.fit(scene)


    ## Change the lighting

    tracer = fresnel.tracer.Path(device, w=1500, h=1500)

    scene.lights = fresnel.light.butterfly()

    tracer.sample(scene, samples=32, light_samples=64)



    # Add an outline
    geometry.outline_width = 0.05
    geometry.material.solid = 1.0
    fresnel.pathtrace(scene, w=1500, h=1500, light_samples=64)


    def check_pos(x=100, y=100, z=100):
        scene.camera.position = (x, y, z)
        out = fresnel.preview(scene, w=1500, h=1500)
        image = PIL.Image.fromarray(out[:], mode='RGBA')

        image.save("just_shell_1d.png")
        # image.show()

    # check_pos()
    check_pos(50, 450, 50)



else:
    raise RuntimeError(f"Mode not ipmlemented: {mode}")
