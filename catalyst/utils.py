from tqdm import tqdm

from jax import vmap

# from jax_md import rigid_body
import catalyst.rigid_body as rigid_body

from jax.config import config
config.update('jax_enable_x64', True)


def get_body_frame_positions(rb, shape):
    body_pos = vmap(rigid_body.transform, (0, None))(rb, shape)
    return body_pos

def traj_to_pos_file(traj, complex_info, traj_path, box_size=30.0):
    assert(len(traj.center.shape) == 3)
    n_states = traj.center.shape[0]

    traj_injavis_lines = list()
    for i in tqdm(range(n_states), desc="Generating injavis output"):
        s = traj[i]
        traj_injavis_lines += complex_info.body_to_injavis_lines(s, box_size=box_size)[0]
    with open(traj_path, 'w+') as of:
        of.write('\n'.join(traj_injavis_lines))
    
