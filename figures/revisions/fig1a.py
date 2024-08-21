import fresnel
import PIL
import pdb
import numpy as onp
from pathlib import Path

import jax.numpy as jnp
from jax_md import space

from catalyst.icosahedron import complex_getter
from figures.revisions.utils import shell_patch_color, shell_vertex_color
from figures.revisions.utils import spider_base_color, spider_leg_color, spider_head_color


displacement_fn, shift_fn = space.free()
vertex_to_bind_idx = 10

output_basedir = Path("figures/revisions/output/fig1/a")
assert(output_basedir.exists())


one_missing = False
one_vertex = False

for one_missing, one_vertex in [(False, False), (True, False), (False, True)]:

    init_sep_coeff = 1.0
    head_radius = 1.0

    complex_info = complex_getter.ComplexInfo(
        initial_separation_coeff=init_sep_coeff, vertex_to_bind_idx=vertex_to_bind_idx,
        displacement_fn=displacement_fn, shift_fn=shift_fn,
        spider_base_radius=5.0, spider_head_height=4.0,
        spider_base_particle_radius=0.5, spider_head_particle_radius=head_radius,
        spider_point_mass=1.0, spider_mass_err=1e-6
    )

    num_vertices = 12
    num_patches = 5
    num_legs = 5

    assert(not (one_missing and one_vertex))
    img_name = "just_shell.png"

    shell_info = complex_info.shell_info

    shell_rb = shell_info.rigid_body
    shell_body_pos = shell_info.get_body_frame_positions(shell_rb)
    shell_body_pos = onp.array(shell_body_pos)

    if one_missing:
        img_name = "one_missing.png"
        start_remove_idx = vertex_to_bind_idx * 6
        end_remove_idx = start_remove_idx + 6
        shell_body_pos = onp.concatenate([shell_body_pos[:start_remove_idx],
                                          shell_body_pos[end_remove_idx:]])
        num_vertices -= 1
    elif one_vertex:
        img_name = "one_vertex.png"
        start_remove_idx = vertex_to_bind_idx * 6
        end_remove_idx = start_remove_idx + 6
        shell_body_pos = shell_body_pos[start_remove_idx:end_remove_idx]
        num_vertices = 1


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

    tracer = fresnel.tracer.Path(device, w=1000, h=1000)

    scene.lights = fresnel.light.butterfly()

    tracer.sample(scene, samples=64, light_samples=16)


    geometry.outline_width = 0.05
    geometry.material.solid = 1.0

    fresnel.pathtrace(scene, w=1000, h=1000, light_samples=16)

    out = fresnel.preview(scene, h=370*2, w=600*2)
    image = PIL.Image.fromarray(out[:], mode='RGBA')

    # image.show()
    image.save(str(output_basedir / img_name))
