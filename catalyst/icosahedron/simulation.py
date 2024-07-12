import pdb
import functools
import unittest
from tqdm import tqdm

from jax import jit, random, vmap, lax

from jax_md.util import *
from jax_md import space, smap, energy, minimize, quantity, simulate, partition
# from jax_md import rigid_body
from jax_md import dataclasses
from jax_md import util

import catalyst.icosahedron.rigid_body as rigid_body
from catalyst.checkpoint import checkpoint_scan
from catalyst.icosahedron.complex_getter import ComplexInfo, PENTAPOD_LEGS, BASE_LEGS
from catalyst.icosahedron.shell_getter import ShellInfo
from catalyst.icosahedron.utils import get_body_frame_positions, traj_to_pos_file
from catalyst.icosahedron.loss import get_loss_fn

from jax.config import config
config.update('jax_enable_x64', True)


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def simulation(complex_info, complex_energy_fn, num_steps, gamma, kT, shift_fn, dt, key):

    gamma_rb = rigid_body.RigidBody(jnp.array([gamma]), jnp.array([gamma/3]))
    init_fn, step_fn = simulate.nvt_langevin(complex_energy_fn, shift_fn, dt, kT, gamma=gamma_rb)
    step_fn = jit(step_fn)

    mass = complex_info.shape.mass(complex_info.shape_species)
    state = init_fn(key, complex_info.rigid_body, mass=mass)

    do_step = lambda state, t: (step_fn(state), state.position)
    do_step = jit(do_step)

    state, traj = scan(do_step, state, jnp.arange(num_steps))
    return state.position, traj


class TestSimulate(unittest.TestCase):

    init_sep_coeff = 0.0

    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.467912900697836,
        "morse_shell_center_spider_head_alpha": 1.2654897136989913,
        "spider_base_particle_radius": 0.5328196552783585,
        "spider_base_radius": 4.965124458025015,
        "spider_head_height": 4.764709630665588,
        "spider_head_particle_radius": 0.1828697409842395,
    }
    """

    # abduction-limit-min-head-no-stable-shell-long, iteration 3500
    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.408032049462529,
        "morse_r_cutoff": 11.56057750337057,
        "morse_r_onset": 9.742170316460735,
        "morse_shell_center_spider_head_alpha": 2.054583554380889,
        "spider_base_particle_radius": 0.4129907309836615,
        "spider_base_radius": 2.2604892727572756,
        "spider_head_height": 6.274483061746559,
        "spider_head_particle_radius": 0.09520242485086461
    }
    """

    # production-sep0.2-eps9.2-alph1.5-no-ss, iteration 150
    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 9.005651628019526,
        "morse_r_cutoff": 11.802113964947548,
        "morse_r_onset": 9.83342386479687,
        "morse_shell_center_spider_head_alpha": 1.7454608172250263,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 4.569387777290548,
        "spider_head_height": 5.282185267636781,
        "spider_head_particle_radius": 0.22747195656686792,
    }
    """


    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 4950
    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 9.221475407506961,
        "morse_r_cutoff": 10.546210091935214,
        "morse_r_onset": 10.513544408974829,
        "morse_shell_center_spider_head_alpha": 1.8757224643886063,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 4.830050255104434,
        "spider_head_height": 5.664320569392129,
        "spider_head_particle_radius": 0.7632605079210569,
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 0
    """
    init_sep_coeff = 0.2
    sim_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 0.5,
        "spider_head_particle_radius": 0.5,
        "log_morse_shell_center_spider_head_eps": 5.5,
        "morse_shell_center_spider_head_alpha": 1.5,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 2500
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 7.07010915780864,
        "morse_r_cutoff": 10.446287469377683,
        "morse_r_onset": 9.286000948885864,
        "morse_shell_center_spider_head_alpha": 2.009179455510144,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 5.04261216365621,
        "spider_head_height": 4.92472225940584,
        "spider_head_particle_radius": 1.1673952824606546
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 4750
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 9.11817907909598,
        "morse_r_cutoff": 10.546210091935214,
        "morse_r_onset": 10.513544408974829,
        "morse_shell_center_spider_head_alpha": 1.8899509179640157,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 4.767557802985431,
        "spider_head_height": 5.62326990937787,
        "spider_head_particle_radius": 0.8048606609317089
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 350
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.856055409600398,
        "morse_r_cutoff": 11.06205688820389,
        "morse_r_onset": 9.259549799745674,
        "morse_shell_center_spider_head_alpha": 2.0295389507816135,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 5.172024778586832,
        "spider_head_height": 4.5071973539472046,
        "spider_head_particle_radius": 0.9053049966101149
    }
    """

    # production-sep0.2-eps11-alph1.5-no-ss, iteration 0
    """
    init_sep_coeff = 0.2
    sim_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 0.5,
        "spider_head_particle_radius": 0.5,
        "log_morse_shell_center_spider_head_eps": 11.0,
        "morse_shell_center_spider_head_alpha": 1.5,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0,
    }
    """

    # production-sep0.2-eps11-alph1.5-no-ss, iteration 1700
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.5729055725228,
        "morse_r_cutoff": 11.761658454863811,
        "morse_r_onset": 9.849567656720794,
        "morse_shell_center_spider_head_alpha": 1.95369930385077,
        "spider_base_particle_radius": 0.5241586402941958,
        "spider_base_radius": 4.7068378064708485,
        "spider_head_height": 5.5381367409541085,
        "spider_head_particle_radius": 0.08740544592241241
    }
    """

    # production-sep0.2-eps11-alph1.5-no-ss, iteration 850
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.663925142281387,
        "morse_r_cutoff": 11.761947663360134,
        "morse_r_onset": 9.849393971871713,
        "morse_shell_center_spider_head_alpha": 1.8764266998889352,
        "spider_base_particle_radius": 0.5241586402941958,
        "spider_base_radius": 4.840503520020092,
        "spider_head_height": 5.381967092965107,
        "spider_head_particle_radius": 0.11457184974371708
    }
    """

    # production-sep0.2-eps11-alph1.5-no-ss, iteration 600
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.677680530185057,
        "morse_r_cutoff": 11.762650290524169,
        "morse_r_onset": 9.849412237413233,
        "morse_shell_center_spider_head_alpha": 1.855672926758726,
        "spider_base_particle_radius": 0.5241586402941958,
        "spider_base_radius": 4.862781843797175,
        "spider_head_height": 5.357382480123232,
        "spider_head_particle_radius": 0.13945912142582137
    }
    """

    # production-sep0.2-eps11-alph1.5-no-ss, iteration 425
    """
    init_sep_coeff = 0.2
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.702638890542561,
        "morse_r_cutoff": 11.764529498601846,
        "morse_r_onset": 9.849538414176202,
        "morse_shell_center_spider_head_alpha": 1.8296436234025493,
        "spider_base_particle_radius": 0.5241586402941958,
        "spider_base_radius": 4.890507002056248,
        "spider_head_height": 5.322118007162969,
        "spider_head_particle_radius": 0.17426078866532183
    }
    """


    # production-sep0.2-eps5.5-alph1.5-no-ss, iteration 360
    """
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.860381021234434,
        "morse_r_cutoff": 11.044329521123666,
        "morse_r_onset": 9.25568992413066,
        "morse_shell_center_spider_head_alpha": 2.029099822995265,
        "spider_base_particle_radius": 0.5,
        "spider_base_radius": 5.181132757773662,
        "spider_head_height": 4.503305867446094,
        "spider_head_particle_radius": 0.9097058678968004,
    }
    """

    """
    sim_params = {
        # catalyst shape
        'spider_base_radius': 5.0,
        'spider_head_height': 5.0,
        'spider_base_particle_radius': 0.5,
        'spider_head_particle_radius': 0.5,

        # catalyst energy
        'log_morse_shell_center_spider_head_eps': 8.0,
        'morse_shell_center_spider_head_alpha': 1.0,
        'morse_r_onset': 10.0,
        'morse_r_cutoff': 12.0
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0, iteration 0
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0-iter0.pos"
    sim_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 1.0,
        "spider_head_particle_radius": 1.0,
        "log_morse_shell_center_spider_head_eps": 5.5,
        "morse_shell_center_spider_head_alpha": 1.5,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0,
    }
    """


    # production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0, iteration 1650
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0-iter1650.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 8.255769563520754,
        "morse_r_cutoff": 10.918875161052911,
        "morse_r_onset": 9.807425043287981,
        "morse_shell_center_spider_head_alpha": 1.9211678141769546,
        "spider_base_particle_radius": 1.0091104500054886,
        "spider_base_radius": 4.503958535895628,
        "spider_head_height": 5.571283961936115,
        "spider_head_particle_radius": 0.9459919094039012,
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0, iteration 110
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0-iter110.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.499446260578564,
        "morse_r_cutoff": 11.755415112238243,
        "morse_r_onset": 9.805069526073947,
        "morse_shell_center_spider_head_alpha": 1.7633505786108319,
        "spider_base_particle_radius": 0.9135636326739526,
        "spider_base_radius": 5.356420749474068,
        "spider_head_height": 4.945954413825034,
        "spider_head_particle_radius": 1.0072897388031377,
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0, iteration 120
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0-iter120.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.6112574202513095,
        "morse_r_cutoff": 11.736376028650179,
        "morse_r_onset": 9.792991001055297,
        "morse_shell_center_spider_head_alpha": 1.7693261932457551,
        "spider_base_particle_radius": 0.8855083207324053,
        "spider_base_radius": 5.436772715679406,
        "spider_head_height": 4.818328299137688,
        "spider_head_particle_radius": 1.124451543196626
    }
    """

    # production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0, iteration 130
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps5.5-alph1.5-no-ss-leg0.25-r1.0-iter130.pos"
    
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.719720538898884,
        "morse_r_cutoff": 11.687934442412624,
        "morse_r_onset": 9.785795093914599,
        "morse_shell_center_spider_head_alpha": 1.7005649155980287,
        "spider_base_particle_radius": 0.8661662086070747,
        "spider_base_radius": 5.5419246797161215,
        "spider_head_height": 4.717428055723606,
        "spider_head_particle_radius": 1.2251309134124795
    }
    """

    # production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0, iteration 0
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0-iter0.pos"

    sim_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 1.0,
        "spider_head_particle_radius": 1.0,
        "log_morse_shell_center_spider_head_eps": 10.5,
        "morse_shell_center_spider_head_alpha": 1.5,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0
    }
    """

    # production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0, iteration 4990
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0-iter4990.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.255644333492606,
        "morse_r_cutoff": 11.785951723604809,
        "morse_r_onset": 9.846117964853981,
        "morse_shell_center_spider_head_alpha": 1.878093078677359,
        "spider_base_particle_radius": 0.9080776013560445,
        "spider_base_radius": 4.221519558881762,
        "spider_head_height": 5.847105147265167,
        "spider_head_particle_radius": 0.1826102000517613
    }
    """

    # production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0, iteration 2500
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0-iter2500.pos"
    
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.042849714309009,
        "morse_r_cutoff": 11.785851382152478,
        "morse_r_onset": 9.830155245885992,
        "morse_shell_center_spider_head_alpha": 1.967606420058759,
        "spider_base_particle_radius": 1.0932485394595859,
        "spider_base_radius": 4.6024671547308165,
        "spider_head_height": 5.639956719373849,
        "spider_head_particle_radius": 0.3766722753497847
    }
    """

    # production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0, iteration 1250
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0-iter1250.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.096981748062712,
        "morse_r_cutoff": 11.786106463528439,
        "morse_r_onset": 9.830083884523114,
        "morse_shell_center_spider_head_alpha": 1.9210738267172196,
        "spider_base_particle_radius": 0.9178163555826525,
        "spider_base_radius": 4.885368173360221,
        "spider_head_height": 5.479316347694025,
        "spider_head_particle_radius": 0.5261052681119952
    }
    """


    # production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0, iteration 625
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps10.5-alph1.5-no-ss-leg0.25-r1.0-iter625.pos"

    sim_params = {
        "log_morse_shell_center_spider_head_eps": 10.143456441752551,
        "morse_r_cutoff": 11.786688555289338,
        "morse_r_onset": 9.830335785187273,
        "morse_shell_center_spider_head_alpha": 1.8710924541101144,
        "spider_base_particle_radius": 0.906171639485045,
        "spider_base_radius": 4.98488275170156,
        "spider_head_height": 5.3963949115948475,
        "spider_head_particle_radius": 0.6057113604025651
    }
    """

    # test init eps 3.0
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"test-init-eps3.0.pos"
    sim_params = {
        "spider_base_radius": 5.0,
        "spider_head_height": 5.0,
        "spider_base_particle_radius": 1.0,
        "spider_head_particle_radius": 1.0,
        "log_morse_shell_center_spider_head_eps": 3.0,
        "morse_shell_center_spider_head_alpha": 1.5,
        "morse_r_onset": 10.0,
        "morse_r_cutoff": 12.0,
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 4999
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter4999.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 9.21046449853169,
        "morse_r_cutoff": 10.874268486287589,
        "morse_r_onset": 10.654390451225392,
        "morse_shell_center_spider_head_alpha": 1.7869950218884967,
        "spider_base_particle_radius": 0.9355884893225339,
        "spider_base_radius": 4.271572338219849,
        "spider_head_height": 5.927486460509821,
        "spider_head_particle_radius": 0.7470556643444934
    }
    """


    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 2500
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter2500.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 6.5351670422416275,
        "morse_r_cutoff": 10.248170139927717,
        "morse_r_onset": 9.223346874149664,
        "morse_shell_center_spider_head_alpha": 1.9000214958370414,
        "spider_base_particle_radius": 0.952295152322053,
        "spider_base_radius": 4.603617310300306,
        "spider_head_height": 4.914202511358603,
        "spider_head_particle_radius": 1.2594821525722162
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 1250
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter1250.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.2169519599496565,
        "morse_r_cutoff": 10.237073451645829,
        "morse_r_onset": 8.619385956704138,
        "morse_shell_center_spider_head_alpha": 2.0834987594796726,
        "spider_base_particle_radius": 0.9953469539862306,
        "spider_base_radius": 4.608224556875475,
        "spider_head_height": 4.46539377809863,
        "spider_head_particle_radius": 1.562486817715836
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 625
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter625.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 5.022829172033229,
        "morse_r_cutoff": 10.75670075458311,
        "morse_r_onset": 8.785070532648472,
        "morse_shell_center_spider_head_alpha": 2.4482688678927222,
        "spider_base_particle_radius": 1.037655549912664,
        "spider_base_radius": 4.66054602488968,
        "spider_head_height": 4.1470396411829755,
        "spider_head_particle_radius": 1.860629165698884,
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 300
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter300.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476420526029521,
        "morse_r_cutoff": 11.035408428019027,
        "morse_r_onset": 9.039624312749842,
        "morse_shell_center_spider_head_alpha": 2.8103173326208184,
        "spider_base_particle_radius": 1.495081834823034,
        "spider_base_radius": 4.338533077523172,
        "spider_head_height": 4.198277629304025,
        "spider_head_particle_radius": 1.8012083137568093
    }
    """


    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 400
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter400.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.6625304511985775,
        "morse_r_cutoff": 10.985728485599719,
        "morse_r_onset": 8.97691543425139,
        "morse_shell_center_spider_head_alpha": 2.7123595244522583,
        "spider_base_particle_radius": 1.1873857147451576,
        "spider_base_radius": 4.591936717323716,
        "spider_head_height": 4.096556332145659,
        "spider_head_particle_radius": 1.9041228412649722
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 500
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter500.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.7774221030138,
        "morse_r_cutoff": 10.835983677251356,
        "morse_r_onset": 8.832617441289635,
        "morse_shell_center_spider_head_alpha": 2.6861197781605215,
        "spider_base_particle_radius": 1.1069414510186815,
        "spider_base_radius": 4.590898489654481,
        "spider_head_height": 4.171030421839727,
        "spider_head_particle_radius": 1.8310504619521966
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 350
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter350.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.504747881751074,
        "morse_r_cutoff": 11.033931782823595,
        "morse_r_onset": 9.038628154567853,
        "morse_shell_center_spider_head_alpha": 2.726367192979255,
        "spider_base_particle_radius": 1.3077600742288402,
        "spider_base_radius": 4.524423205409979,
        "spider_head_height": 4.159950584895944,
        "spider_head_particle_radius": 1.8397094660971374
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 375
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter375.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.54141925664745,
        "morse_r_cutoff": 11.01325356725435,
        "morse_r_onset": 9.012456496716501,
        "morse_shell_center_spider_head_alpha": 2.663733775360324,
        "spider_base_particle_radius": 1.1534243478382646,
        "spider_base_radius": 4.656822351562876,
        "spider_head_height": 4.162331829587933,
        "spider_head_particle_radius": 1.8379280918460044
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 325
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter325.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.477905441833658,
        "morse_r_cutoff": 11.033814350380274,
        "morse_r_onset": 9.038679365922249,
        "morse_shell_center_spider_head_alpha": 2.800334558961143,
        "spider_base_particle_radius": 1.4737836677540248,
        "spider_base_radius": 4.359648485332705,
        "spider_head_height": 4.195989978367066,
        "spider_head_particle_radius": 1.803516162001804
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 320
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 4
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter320-k4.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.477412966522471,
        "morse_r_cutoff": 11.034143065383951,
        "morse_r_onset": 9.038876532388898,
        "morse_shell_center_spider_head_alpha": 2.8031397769503825,
        "spider_base_particle_radius": 1.4789575324466533,
        "spider_base_radius": 4.354326557707576,
        "spider_head_height": 4.19670207700281,
        "spider_head_particle_radius": 1.802803039962012
    }
    """


    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 315
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter315.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476551638157115,
        "morse_r_cutoff": 11.034459890098796,
        "morse_r_onset": 9.03906745395958,
        "morse_shell_center_spider_head_alpha": 2.8073194629603675,
        "spider_base_particle_radius": 1.4890898729408901,
        "spider_base_radius": 4.344905792435479,
        "spider_head_height": 4.198049100608667,
        "spider_head_particle_radius": 1.8014532096260982
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 310
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 4
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter310-k4.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476469561373039,
        "morse_r_cutoff": 11.034782008852913,
        "morse_r_onset": 9.039259275392006,
        "morse_shell_center_spider_head_alpha": 2.808366843174372,
        "spider_base_particle_radius": 1.494781351058766,
        "spider_base_radius": 4.339445653263561,
        "spider_head_height": 4.198166823921447,
        "spider_head_particle_radius": 1.8013289254327418
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 305
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 0
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter305.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476576454198253,
        "morse_r_cutoff": 11.03510182325282,
        "morse_r_onset": 9.039447931708105,
        "morse_shell_center_spider_head_alpha": 2.8087382465847908,
        "spider_base_particle_radius": 1.4962963405996859,
        "spider_base_radius": 4.337798946003916,
        "spider_head_height": 4.198001556708442,
        "spider_head_particle_radius": 1.8014885467682005
    }
    """

    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 300
    """
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 4
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter300-k4.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476420526029521,
        "morse_r_cutoff": 11.035408428019027,
        "morse_r_onset": 9.039624312749842,
        "morse_shell_center_spider_head_alpha": 2.8103173326208184,
        "spider_base_particle_radius": 1.495081834823034,
        "spider_base_radius": 4.338533077523172,
        "spider_head_height": 4.198277629304025,
        "spider_head_particle_radius": 1.8012083137568093
    }
    """


    # production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0, iteration 250
    init_sep_coeff = 0.2
    spider_leg_radius = 0.25
    key = 4
    traj_fname = f"production-sep0.2-eps3.0-alph1.5-no-ss-leg0.25-r1.0-iter250-k4.pos"
    sim_params = {
        "log_morse_shell_center_spider_head_eps": 4.476509364495032,
        "morse_r_cutoff": 11.038233247310973,
        "morse_r_onset": 9.04130517696154,
        "morse_shell_center_spider_head_alpha": 2.8169209597314255,
        "spider_base_particle_radius": 1.5171523313685982,
        "spider_base_radius": 4.3184937944005615,
        "spider_head_height": 4.198054318168296,
        "spider_head_particle_radius": 1.801382346087309
    }

    

    def test_simulate_complex(self):

        displacement_fn, shift_fn = space.free()
        min_head_radius = 0.1

        spider_bond_idxs = jnp.concatenate([PENTAPOD_LEGS, BASE_LEGS])

        complex_info = ComplexInfo(
            initial_separation_coeff=self.init_sep_coeff, vertex_to_bind_idx=5,
            displacement_fn=displacement_fn, shift_fn=shift_fn,
            spider_base_radius=self.sim_params["spider_base_radius"],
            spider_head_height=self.sim_params["spider_head_height"],
            spider_base_particle_radius=self.sim_params["spider_base_particle_radius"],
            spider_head_particle_radius=jnp.max(jnp.array([min_head_radius, self.sim_params["spider_head_particle_radius"]])),
            spider_point_mass=1.0, spider_mass_err=1e-6,
            spider_bond_idxs=spider_bond_idxs,
            spider_leg_radius=self.spider_leg_radius
        )
        energy_fn = complex_info.get_energy_fn(
            morse_shell_center_spider_head_eps=jnp.exp(self.sim_params["log_morse_shell_center_spider_head_eps"]),
            morse_shell_center_spider_head_alpha=self.sim_params["morse_shell_center_spider_head_alpha"]
        )

        n_steps = 100000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(self.key)
        # key = random.PRNGKey(1)
        fin_state, traj = simulation(
            complex_info, energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)

        # Write final states to file -- visualize with `java -Xmx4096m -jar injavis.jar <name>.pos`

        ## Shell
        # fin_shell_rb = fin_state[:12]
        # shell_lines = complex_info.shell_info.body_to_injavis_lines(fin_shell_rb, box_size=30.0)
        # with open('shell_state.pos', 'w+') as of:
        #     of.write('\n'.join(shell_lines))

        ## Spider
        # fin_spider_rb = fin_state[-1]
        # spider_lines = complex_info.spider_info.body_to_injavis_lines(fin_spider_rb, box_size=30.0)
        # with open('spider_state.pos', 'w+') as of:
        #     of.write('\n'.join(spider_lines))

        ## Complex
        """
        complex_lines, _, _, _ = complex_info.body_to_injavis_lines(fin_state, box_size=30.0)
        with open('complex_state.pos', 'w+') as of:
            of.write('\n'.join(complex_lines))
        """

        # Compute the loss of the final state
        """
        complex_loss_fn, _ = get_loss_fn(
            displacement_fn, complex_info.vertex_to_bind_idx,
            use_abduction=False,
            use_stable_shell=False, stable_shell_k=20.0,
            use_remaining_shell_vertices_loss=True, remaining_shell_vertices_loss_coeff=1.0
        )
        fin_state_loss = complex_loss_fn(fin_state, self.sim_params, complex_info)
        print(f"Loss: {fin_state_loss}")
        """


        # Write trajectory to file

        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        traj = traj[vis_traj_idxs]
        
        traj_to_pos_file(traj, complex_info, self.traj_fname, box_size=30.0)


    def _test_simulate_shell(self):

        displacement_fn, shift_fn = space.free()

        shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
        shell_energy_fn = shell_info.get_energy_fn()

        n_steps = 5000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)

        fin_state, traj = simulation(
            shell_info, shell_energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        vis_traj = traj[vis_traj_idxs]
        traj_to_pos_file(vis_traj, shell_info, "traj_shell.pos", box_size=30.0)

    def _test_simulate_shell_remainder(self):

        displacement_fn, shift_fn = space.free()

        shell_info = ShellInfo(displacement_fn=displacement_fn, shift_fn=shift_fn)
        shell_info.rigid_body = shell_info.rigid_body[:-1]
        shell_energy_fn = shell_info.get_energy_fn()

        n_steps = 5000
        assert(n_steps % 100 == 0)
        key = random.PRNGKey(0)

        fin_state, traj = simulation(
            shell_info, shell_energy_fn, num_steps=n_steps,
            gamma=10.0, kT=1.0, shift_fn=shift_fn, dt=1e-3, key=key)


        # Write trajectory to file
        vis_traj_idxs = jnp.arange(0, n_steps+1, 100)
        vis_traj = traj[vis_traj_idxs]
        traj_to_pos_file(vis_traj, shell_info, "traj_shell.pos", box_size=30.0)




if __name__ == "__main__":
    unittest.main()
