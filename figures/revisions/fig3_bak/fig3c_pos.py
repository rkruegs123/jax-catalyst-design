import fresnel
import PIL
import pdb
import numpy as onp
from pathlib import Path

from figures.revisions.utils import shell_patch_color, shell_vertex_color
from figures.revisions.utils import spider_base_color, spider_leg_color, spider_head_color





patch_radius = 0.5
vertex_radius = 2.0

fig3_data_basedir = Path("figures/revisions/data/fig3/")

for state_idx in [19, 400]:

    f_rel_path = f"wham-op1/rigid/eq_state{state_idx}.pos"
    # f_rel_path = "wham-op1/rigid/eq_state400.pos"
    fpath = fig3_data_basedir / f_rel_path

    """
    spider_leg_radius = 0.25
    spider_base_radius = 1.0
    spider_head_radius = 1.0
    spider_attr_site_radius = 0.75
    leg_radius = 0.25
    """

    spider_leg_radius = 0.25
    spider_base_radius = 1.4979135216810637
    spider_head_radius = 1.0
    spider_attr_site_radius = 1.4752831792315242
    leg_radius = 0.25


    output_basedir = Path("figures/revisions/output/fig3/c")
    assert(output_basedir.exists())


    with open(fpath, "r") as f:
        state_lines = f.readlines()[6:-1]
    state_shell_lines = state_lines[:-11]
    state_spider_lines = state_lines[-11:]

    shell_body_pos = list()
    for shell_line in state_shell_lines:
        elts = shell_line.strip().split()
        shell_body_pos.append(onp.array(elts[1:], dtype=onp.float64))
    shell_body_pos = onp.array(shell_body_pos, dtype=onp.float64)

    spider_body_pos = list()
    for spider_line in state_spider_lines:
        elts = spider_line.strip().split()
        spider_body_pos.append(onp.array(elts[1:], dtype=onp.float64))
    spider_body_pos = onp.array(spider_body_pos, dtype=onp.float64)


    def rotate(axis, theta, shell_body_pos, spider_body_pos):
        if axis == "x":
            rot = onp.array([
                [1, 0, 0],
                [0.0, onp.cos(theta), -onp.sin(theta)],
                [0.0, onp.sin(theta), onp.cos(theta)]
            ])
        elif axis == "y":
            rot = onp.array([
                [onp.cos(theta), 0.0, onp.sin(theta)],
                [0, 1, 0],
                [-onp.sin(theta), 0.0, onp.cos(theta)]
            ])
        elif axis == "z":
            rot = onp.array([
                [onp.cos(theta), -onp.sin(theta), 0.0],
                [onp.sin(theta), onp.cos(theta), 0.0],
                [0, 0, 1],
            ])

        shell_body_pos = onp.dot(shell_body_pos, rot.T)
        spider_body_pos = onp.dot(spider_body_pos, rot.T)
        return shell_body_pos, spider_body_pos


    # shell_body_pos, spider_body_pos = rotate("x", -onp.pi / 2, shell_body_pos, spider_body_pos)
    shell_body_pos, spider_body_pos = rotate("x", -onp.pi / 1.75, shell_body_pos, spider_body_pos)

    num_vertices = 1
    num_patches = 5

    assert(num_vertices * (num_patches+1) == shell_body_pos.shape[0])
    radii = list()
    for i in range(shell_body_pos.shape[0]):
        if i % (num_patches + 1) == 0:
            radii.append(vertex_radius)
        else:
            radii.append(patch_radius)


    device = fresnel.Device()
    scene = fresnel.Scene(device)
    tracer = fresnel.tracer.Path(device, w=1000, h=1000)


    geometry_shell = fresnel.geometry.Sphere(scene, N=shell_body_pos.shape[0], radius=radii)
    geometry_shell.position[:] = shell_body_pos

    geometry_shell.material = fresnel.material.Material(color=fresnel.color.linear(shell_patch_color),
                                                  roughness=0.8)
    geometry_shell.material.primitive_color_mix = 0.5
    # geometry_shell.material.primitive_color_mix = 1.0
    geometry_shell.color[::(num_patches+1)] = fresnel.color.linear(shell_vertex_color)



    geometry_shell.outline_width = 0.05
    geometry_shell.material.solid = 1.0




    num_base_particles = 5
    assert(spider_body_pos.shape[0] == (num_base_particles*2)+1)
    head_pos = spider_body_pos[0]
    base_particle_positions = spider_body_pos[1::2]
    attr_site_particle_positions = spider_body_pos[2::2]


    # radii = [spider_base_radius]*num_base_particles + [spider_attr_site_radius]*num_base_particles + [spider_head_radius]
    radii = [spider_head_radius] + [spider_base_radius, spider_attr_site_radius]*num_base_particles


    geometry1 = fresnel.geometry.Sphere(scene, N=spider_body_pos.shape[0], radius=radii)
    geometry1.position[:] = spider_body_pos
    geometry1.material = fresnel.material.Material(color=fresnel.color.linear(spider_base_color),
                                                   roughness=0.8)
    geometry1.material.primitive_color_mix = 0.5
    # geometry1.color[-1] = fresnel.color.linear(spider_head_color)
    # geometry1.color[num_base_particles:2*num_base_particles] = fresnel.color.linear(spider_head_color)
    geometry1.color[2::2] = fresnel.color.linear(spider_head_color)


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
    save_fname = f"eq_state{state_idx}.png"
    save_fpath = str(output_basedir / save_fname)
    image.save(save_fpath)
