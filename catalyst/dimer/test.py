import pdb
import unittest
import numpy as onp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from jax import random, jit
import jax.numpy as jnp
from jax_md import space, simulate, energy

import catalyst.rigid_body as rigid_body


def simulate_dimers(eps, sigma, rc):

    alpha = 2 * (rc / sigma)**2 * (3 / (2 * ((rc/sigma)**2 - 1)))**3
    rmin = rc * (3 / (1 + 2 * (rc/sigma)**2))**(1/2)

    init_dimer_dist = rmin # start bonded
    box_size = 15.0

    """
    monomer1_rb = rigid_body.RigidBody(
        center=jnp.array([0.0, 0.0]),
        orientation=jnp.array([0.0]))
    monomer1_rb = rigid_body.RigidBody(
        center=jnp.array([0.0, init_dimer_dist]),
        orientation=jnp.array([0.0]))
    monomers_rb = rigid_body.RigidBody(
        center=jnp.array([monomer1_rb.center, monomer2_rb]),
        orientation=jnp.array([monomer1_rb.orientation, monomer2_rb.orientation]))
    """

    monomers = jnp.array([[box_size / 2, box_size / 2],
                          [box_size / 2, box_size / 2 + init_dimer_dist]])

    displacement_fn, shift_fn = space.periodic(box_size)

    @jit
    def energy_fn(positions):
        m1 = positions[0]
        m2 = positions[1]
        dr = displacement_fn(m1, m2)
        r = space.distance(dr)

        val = eps * alpha * ((sigma/r)**2 - 1) * ((rc/r)**2 - 1)**2
        return jnp.where(r <= rc, val, 0.0)

    init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt=1e-4, kT=1.0, gamma=12.5)

    key = random.PRNGKey(0)
    state = init_fn(key, monomers, mass=1.0)

    do_step = lambda state, t: (apply_fn(state), state.position)
    do_step = jit(do_step)



    n_steps = int(1e4)
    save_every = 100

    trajectory = [state.position]
    energies = [energy_fn(state.position)]
    distances = [space.distance(displacement_fn(state.position[0], state.position[1]))]
    start = time.time()
    
    for t in tqdm(range(n_steps)):
        state, pos_t = do_step(state, t)
        if t % save_every == 0:
            trajectory.append(pos_t)
            energies.append(energy_fn(pos_t))
            distances.append(space.distance(displacement_fn(pos_t[0], pos_t[1])))
    end = time.time()

    total = onp.round(end - start, 2)
    print(f"Total time: {total}")

    plt.plot(energies)
    plt.axhline(y=-eps, linestyle='--')
    plt.title("Energies")
    plt.show()

    plt.plot(distances)
    plt.axhline(y=rc, linestyle='--')
    plt.title("Distances")
    plt.show()

    # Write to injavis file
    color="4fb06d"

    radius = rmin / 2 # note: rmin is a terrible name
    box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 0"
    head_def = f"def M \"sphere {radius*2} {color}\""

    all_lines = list()
    for pos in trajectory:
        all_lines += [box_def, head_def]
        m1_pos = pos[0]
        m1_line = f"M {m1_pos[0] - box_size / 2} {m1_pos[1] - box_size / 2} 0.0"

        m2_pos = pos[1]
        m2_line = f"M {m2_pos[0] - box_size / 2} {m2_pos[1] - box_size / 2} 0.0"

        all_lines += [m1_line, m2_line, "eof"]

    with open("traj.pos", 'w+') as of:
        of.write('\n'.join(all_lines))

    print("done!")


def plot_daan_frenkel_not_lj(eps, sigma, rc, k):


    alpha = 2 * (rc / sigma)**2 * (3 / (2 * ((rc/sigma)**2 - 1)))**3
    rmin = rc * (3 / (1 + 2 * (rc/sigma)**2))**(1/2)

    # note: assumes mu and nu are 1
    def daan_frenkel_not_lj(r):
        val = eps * alpha * ((sigma/r)**2 - 1) * ((rc/r)**2 - 1)**2
        return jnp.where(r <= rc, val, 0.0)

    rs_to_plot_min = 1
    rs = jnp.linspace(rs_to_plot_min, rc + 2, 100)
    not_lj_energies = daan_frenkel_not_lj(rs)

    plt.plot(rs, not_lj_energies, label="not_lj")
    plt.ylabel("Energy")
    plt.xlabel("r")
    plt.axvline(x=rmin, linestyle='--', label="rmin")
    plt.axhline(y=-eps, linestyle='--', label="-eps")
    plt.legend()

    plt.show()


    def harmonic_potential(r):
        return 1/2 * k * (r - rmin)**2

    harmonic_rs = jnp.linspace(rs_to_plot_min, rmin, 50)
    harmonic_energies = harmonic_potential(harmonic_rs)
    plt.plot(harmonic_rs, harmonic_energies, label="harmonic")

    plt.show()


    """
    Note: they only have the harmonic thing for valence reasons. We can't do this as they did (note: this is probalby one of the reasons why they wrote a custom integrate). We can mimic this by having four distinct species
    """
def get_first_dissociation_time(key, eps, sigma, rc, n_steps=int(1e5)):

    alpha = 2 * (rc / sigma)**2 * (3 / (2 * ((rc/sigma)**2 - 1)))**3
    rmin = rc * (3 / (1 + 2 * (rc/sigma)**2))**(1/2)

    init_dimer_dist = rmin # start bonded
    box_size = 15.0

    """
    monomer1_rb = rigid_body.RigidBody(
        center=jnp.array([0.0, 0.0]),
        orientation=jnp.array([0.0]))
    monomer1_rb = rigid_body.RigidBody(
        center=jnp.array([0.0, init_dimer_dist]),
        orientation=jnp.array([0.0]))
    monomers_rb = rigid_body.RigidBody(
        center=jnp.array([monomer1_rb.center, monomer2_rb]),
        orientation=jnp.array([monomer1_rb.orientation, monomer2_rb.orientation]))
    """

    monomers = jnp.array([[box_size / 2, box_size / 2],
                          [box_size / 2, box_size / 2 + init_dimer_dist]])

    displacement_fn, shift_fn = space.periodic(box_size)

    @jit
    def energy_fn(positions):
        m1 = positions[0]
        m2 = positions[1]
        dr = displacement_fn(m1, m2)
        r = space.distance(dr)

        val = eps * alpha * ((sigma/r)**2 - 1) * ((rc/r)**2 - 1)**2
        return jnp.where(r <= rc, val, 0.0)

    init_fn, apply_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt=1e-4, kT=1.0, gamma=12.5)

    state = init_fn(key, monomers, mass=1.0)

    do_step = lambda state, t: (apply_fn(state), state.position)
    do_step = jit(do_step)

        
    for t in tqdm(range(n_steps)):
        state, pos_t = do_step(state, t)
        dist = space.distance(displacement_fn(pos_t[0], pos_t[1]))
        if dist > rc:
            return t
    return -1

def get_dissociation_distribution(key, batch_size, eps, sigma, rc, n_steps=int(1e5)):
    diss_times = []
    dt = 1e-4
    for b in tqdm(range(batch_size)):
        key, split = random.split(key)
        t = get_first_dissociation_time(split, eps, sigma, rc, n_steps)
        if t != -1:
            diss_times += [t*dt]
        else:
            print('Warning: no dissociation')
    return diss_times



if __name__ == "__main__":
    # plot_daan_frenkel_not_lj(eps=1.0, sigma=1.0, rc=10.0, k=2.0)

    sigma = 1.0
    rc = 1.1
    eps = 5.0
    batch_size=10
    key = random.PRNGKey(0)
    assert(rc / sigma == 1.1)
    #simulate_dimers(eps=1.0, sigma=sigma, rc=rc)
    diss_times = get_dissociation_distribution(key, batch_size, eps, sigma, rc, n_steps=int(1e6))
    onp.save('diss_times.npy', diss_times, allow_pickle=False)
    expected_avg_diss_time = -0.91 * eps + 2.2
    ln_k_measured = jnp.log( 1 / jnp.mean(jnp.array(diss_times)))
    print('expected average ln k: ',  expected_avg_diss_time)
    print('measured average ln k: ', ln_k_measured)
    pdb.set_trace()
