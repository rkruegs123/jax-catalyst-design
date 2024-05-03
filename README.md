# jax-catalyst-design

Code for designing **spider catalysts**

# May 3, 2024

Stuff that we need:

Figure 1 will be overview, as we have it now

Figure 2 will combine current Fig. 2 and Fig 3. Need t oshow it working in high and low initiailization limits, also releasing the particle

Figure 3 will show similarity of FE energy diagram (OP as the extraction distance) w/ partial solutions. Need those partial solutions and those free energy diagrams. Kinetics matter!

Fig 4 wlil show the bneefit of flexibility. Compare FE diagrams with OP as release distance. Need those free energy diagams. Both tagged in this case. Compute entropy differences.

Fig 5 will show how one can add/change terms to the loss function to affect the optimized free energy diagram in an informed way. E.g. for flexibile version, add realeas term, show how this pushes FE towards release.


# April 2, 2024

- We decided to do the following for the WHAM energy function: make it all a part of complex and just have separate helpe rfunctions for e.g. constructing th combined eneryg, and also getting the energy function. Then all the logic stays in the same place. We do this next time.


# February 20, 2024
- added the "cheating" by only interacting with the vertex to bind
- played around - we can still extract, can't get it to release
- next: formulate optimization to get it to extract and release



# February 16, 2024
- did lots of experimenting: we're having lots of issues with interactions with other vertices
- plan (next week): 
    --add loose springs between non-bonded spider base particles so that the spider stays sort of together/can't fully flatten out
    --cheat by only having the spider attract the vertex to bind, the physical analogy is adding
      a tag to the particle we want to pull off at a later time



# February 6, 2024

- Added interaction energy between shell and spider (just morse attraction and general soft sphere repulsion)
  - next: leg repulsion, maybe add repulsion between head vertex and extracted vertex
- Then, optimize for extraction
- Then go from there...
  - e.g. add an energy penalty for the vertex to bind interaction. Want to minimize this.

# September 7, 2023

There are really three things wrong at th emoment
- we don't think the legs are working
- we thought we had the initialization right, but we don't really -- at least tha's what it looks like in th eoptimization
- wea re coloring the wrong ertex to bind in the trajectories. That would help for checking the above.

So, we shouuld do the following:
- fix (3)
- have some hepler function in complex getter that gets is the energy terms so we can look at all them .e.g. the shell energy, spider energy, interaction energy (and maybe its subcomponents) and the leg energy. Then, we can evalutae on an initialization and see (i) what's wrong with th eorienation and (ii) what's wrong with th eleg energy, if anything for both

Note that the initiailization is currently wrong in suc a way that the leg energy should be high, so we should look at the leg energy term before we fix the initializaation

Also note that one thing that could be wrong with the initailziation is init_fn from simulate. So, we should visualizae both pre- and post- state initialization. Because we thought things were working, but we weren't initailizing a siulation, whereas the ifrst state of a trajectory (which is what currently looks wrong) will be post-initialization of state, and we know that initialization involves some diagonzliation, etc.
- if that's the case, easiest thing to try would be different vertex to bind. ha. otherwise we can try different eigenvalue stuff, etc ,etc.


# August 29, 2023

So, we realized why we are having all these weird orientation things

When you create a shape in JAX-MD, it changes the default body frame such that the moment of inertia is diagonal. When we say the default body frame, we mean the relative space frame positions when the orientation is [1, 0, 0, 0]. This is typically fine for the psider as the longest axis of ortation wlil be from head to base, but for the vertex shape this can certainl8y vary.

So, we set the spider orientation in the complex similar to how we se the vertex shape -- by measuring the current body frame vector (i.e. head to base center) of the spider, as well a the target vector (i.e. the vector from the vertex to bind to the center of the shell), and doing the same trig. trick to reorient the thing. This works just fine.

Priorities going forwrad:
- Get octahedron optimization working
- Do the same for icosahedron. Can then probably get rid of all this orig\_rigid\_body nonsense



# August 24, 2023

We have been evaluating the Zorana dimer thing. They only found the effect they cared about at high epsilon, so things take really long timescales, eg 1e9, 1e10 (note: we don't know if they did ensembles or not, and we also don't know why the efefect can't be observed at lower enerygy scales). This raised some question marks for us in terms of how to do the optimzation.

As a first way to address these question marks, we just wanted to see if we could take stable gradients through long simulations with their gamma and dt. We did this for 1e6, 1e7, and 1e8 (note that 1e8 is sitll running). We did see stable (i.e. non-exploding gradients). With rigid bodies, 1e6 took like 220 seconds per gradient update and 1e7 took 10x that, so compiling didn't relaly get a speedup. Note that we are using a foriloop and just computing a runing average (of sorts) so that we don't have to store all the distances. It runs out of memory for 1e9 -- we could probably address this by doing a scan with some rematting, but is it really worth it...? For 1e9 steps, we expect a single gradient update (without any rematting) to take ~220*1e3 seconds, which is way too much.

So, looking forward, we see Zorana's caes as a great test bed for optimizing w.r.t. enhanced sampling but we don't want to do that right now.

We'd rather just prove that we can do the icosahedron thing for other shells, close that loop, then try to deal with enhanced sampling for different reaction types (i.e. substitution, addition, elimination).

So, we are going to go forth and try to do what we did for the icosahedron for the octahedron.



# August 18, 2023

We agree that we will follow one of two courses:
- (i) get optimization over long simulations to work either by enhanced sampling (e.g. metadynamics of DNEB) or some other method
  - to give us confidence that we need this, we are running some gradient checks over long simulatoins. We set this up without rigid bodies, and we want to add rigid bodies later to see how gradient stability is affected
- (ii) Somethign else -- e.g. relax the problem definition

Note that we also still want to email Zoarana at some point, something including the following three points:
- why didn't lower epsilons work?
- did you simulate things in ensembles?
- why did you use such a small time step?

Another tihng we want to do: run dimer dissociation tests at higher dt and see if we can rpeproduce the same expected rates

# August 14, 2023

Goal for today:
- set up eval for non-RB vs RB for different epsilons for dissociation time
- reread paper
- email zorana and first author with questions
  - e.g. whether or not their simulations were in bulk, simulation times, why they did such a large epsilon for the substrate

# August 10, 2023

Gameplan:

Need to scan the dissociation times thing to do higher throughoupt. Then, confir mwith larger batch sizes for eps 4, and 5. Could also try 15

Then, do the same for the monomers as rigid bodies, confirm that we get same results.

Then do with catalyst. Have to be careful with valence considerations...

Also should probably read the paper...

# August 8, 2023

We finally are simulating a catalyst and some monomers. Next, need tro continue validating them in the forward case with each other (e.g. setting the right epsilons and measuring rates, also measuring bulk properties)

Then, before we do any optimization, we need to make a rigid body for the dimer that uses the right counts (i.e. not 6 and 6). Note that this versio nof rigid body will only be valid for transfomriong/simulating the complex (i.e. monomers + catalyst). We will also have to be careful with species for monomers and catalyst.

(Notes Part 2):

Nose-Hoover takes forever to run. We want to validate the catalyst + substrate simulations, but we are just going to wait until Langevin is put in.

In the meantime, ew wrote the componets for the loss. While we could directly optimize for the monomers falling off, we are going to target the energies instead. That term will just be porportional to epsilon. the other term will just be the distances betwen the monomers.


# August 2, 2023

Reconvening on icosahedron stuff before we move onto Zorana dimer stuff:

Before we left off, we learned many things. The first was that at kT of 2.0, the shell falls apart by itself for simulations > 1000 timesteps. So, we set kT to  1.0 and confirmed that the icosahedron was stable at this temperature.

Once ew fixed kT at 1.0, we sought to optimize such that the thing fell off. We tried to achieve byis by adding a loss term that minimized the remianing energy bewteen the spider/catalyst and the *remianing* 11 vertices (i.e. those that you aren't trying to pull off).

However, for any optimizion to work, we learned something (obvious) -- that you need a signal to start. This means that you can't start with the catalyst not interacting *at all* with the icosahedron. For all optimizations, we started with the following set of parameters:
```
params = {
    # catalyst shape
    'spider_base_radius': 5.0,
    'spider_head_height': 5.0,
    'spider_base_particle_radius': 0.5,
    'spider_head_particle_radius': 0.5,

    # catalyst energy
    'log_morse_shell_center_spider_head_eps': FIXME,
    'morse_shell_center_spider_head_alpha': 1.5,
    'morse_r_onset': 10.0,
    'morse_r_cutoff': 12.0
}
```
Note the ``FIXME'' for `log_morse_shell_center_spider_head_eps`. This is the one parameter that we varied across runs. We found that, for there to be any gradient signal (i.e. any interaction between the icosahedron and the catalyst), we needed a log head eps of at least 5.5-6.0 ish. Note that at this lower end, there is no vertex abduction to begin with. So, falling off (i.e. the catalyst diffusin gaway) is "easy", but there is no abduction.

The opposite scenario is to start with *too strong* of an interaction between the catalyst and the icosahedron. Empirically, this will manifest as the shell blowing paart/being attracted to the catalyst (either within 1000 steps or for longer trajectories). This is th eopposite end of the spectrum, where abduction/pulling off is easy, but falling off is hard.

So, we sought to do two kinds of optimizations -- one where we begin with falling off/diffusion and try to balance that with abduction, and another where abduction is easy and we try to balance that with falling off/diffusion/not interacting too strongly with the rest of the shell. These are characterized by a low and high starting log head eps, respectively.

Note that in the following, we record the parameters from the last iteration, even toiugh for each optimization run we produce a summary file that logs th ebest iteration.

Limit 1: low starting head eps, start with falling off due to diffusion, need to get abduciton. (i.e. abduction starts hard).
- Starting with a starting log head eps of 5.5
- Ran the following command: `python3 -m catalyst.optimize --use-abduction-loss --batch-size 10 --n-iters 250 --n-steps 1000 -g 10 -kT 1.0 --vis-frame-rate 100 --lr 0.01 -k 0 --leg-mode both --use-stable-shell-loss --stable-shell-k 20.0 --use-remaining-shell-vertices-loss --remaining-shell-vertices-loss-coeff 1.0 --run-name first-try-kt1-coeff1-eps55`
- Parameters of final iteration:
```
Iteration 249:
- log_morse_shell_center_spider_head_eps: 6.564301325813623
- morse_r_cutoff: 10.588998628881034
- morse_r_onset: 8.948694316574155
- morse_shell_center_spider_head_alpha: 1.87168885282763
- spider_base_particle_radius: 0.600637705989551
- spider_base_radius: 4.840478096384897
- spider_head_height: 4.6262113602045805
- spider_head_particle_radius: 0.7778571364301861
```
- We checked this with a 20k step simulation using `python3 -m catalyst.simulation`. Note that you have to set the parameters of the simulation by hand.
- Note that we named the directory "first-try-kt1-coeff1-eps55". We have created a directory called `good-icos-optimizations` and have copied this run as `good-icos-optimizations/init-diffusive-limit`
  - We have copied the sanity check simulation using the parameters from the final iteratoin and 20k steps in this directory as `iter249_20k_traj.pos`


Limit 2: high starting head eps, start with abduction, need to get falling off due to diffusion (and no shell explosion). (i.e. falling off/diffusion/not exploding starts hard).
- Starting with a starting log head eps of 9.2
- Ran the following command: `python3 -m catalyst.optimize --use-abduction-loss --batch-size 10 --n-iters 250 --n-steps 1000 -g 10 -kT 1.0 --vis-frame-rate 100 --lr 0.01 -k 0 --leg-mode both --use-stable-shell-loss --stable-shell-k 20.0 --use-remaining-shell-vertices-loss --remaining-shell-vertices-loss-coeff 1.0 --run-name first-try-kt1-coeff1-eps92`
- Parameters of final iteration:
```
Iteration 249:
- log_morse_shell_center_spider_head_eps: 8.933010009519583
- morse_r_cutoff: 11.742239951761889
- morse_r_onset: 9.803544376687315
- morse_shell_center_spider_head_alpha: 1.7910141878127712
- spider_base_particle_radius: 0.6394784533750115
- spider_base_radius: 4.787588005279043
- spider_head_height: 5.331054263095673
- spider_head_particle_radius: 0.17076660748048453
```
- We checked this with a 20k step simulation using `python3 -m catalyst.simulation`. Note that you have to set the parameters of the simulation by hand.
- Note that we named the directory "first-try-kt1-coeff1-eps92". We have copied this run as `good-icos-optimizations/init-abduction-limit`
  - We have copied the sanity check simulation using the parameters from the final iteratoin and 20k steps in this directory as `iter249_20k_traj.pos`



## July 17, 2023

We coul doptimize for things including this wide spring potential, but it blows up for longer trajectories. This made us look at the interaction energy, because we thought that maybe we could betterrestrict the range of the pairwise morse potential (note that morse_pair uses multiplicaitve isotropic cutofF)

We have a couple of takewaay notes from this:
- we should really be using the papropriate sigma for the morse potential, and not rely on soft sphere as the repuslve component. Currently, we have a frustraed system (with sigma=0 for the morse potential) and perhaps this could affect gradient stability
- one potential way to mitigate "explosion" for timescales longer than those of the optimization is to optimize for 2k steps, with the loss requiring abduction at 1k steps and non-explosion the whole way thorugh
  - but, to get a sense of this, we should really be able to track the value of the explosion term/wide spring term throughout the simulation, up to 2k steps


## July 14, 2023

We switched to 64 bit precision, as well as the correct eigen calculation (becauese ew are already using the new rigid body stuff) to get more stability because our loss funcitonw as a bit all ove rthe place. Then, we ran wit ha bunch of differen tparametres, like a lower agmma, higher kt, highe rlearning rate. What ee found is that increasing kT allowed us to retrievee our lod loss/behavior in a stable fashion.

We also visualized the final state, so the last frame in an injavis trajectory will be off by one because we enforce that the number of steps is divisible by th eframe rate

The parameters that seem to work are the following:
- batch_size: 10
- n_iters: 100
- key_seed: 0
- n_steps: 1000
- vertex_to_bind: 5
- lr: 0.01
- init_separate: 0.0
- data_dir: data/
- temperature: 2.0
- dt: 0.001
- gamma: 10.0
- use_abduction_loss: True
- use_stable_shell_loss: False
- vis_frame_rate: 100

## July 14, 2023

We set up legls but it doesn't look like it's working bsaed on simulations. As a first test, we are going to add it as a fourth term in the energy components function and check the value of that term. If that doesn't work, we'll probably have to do the calculation manually on some final state -- we'll do the postiion transformation, etc.


## July 13, 2023

Part 1:
- accomplished lots of logging, like the trajectory per iteration (with a sufficiently low number of frames to make injavis conversion onot crazy expensive) and the average gradients. Also better formatting...
- we need to experiment a bit with forward simulation tot hceck if things are even interacting in our initla parameters and if we can get abduction by hand turning the paremetress i.e. just to fheck that there *is* a solution). Some things to checK: strong head eps, lower head alpha. Right now alpha is 4.5 and 1/4.5 is quite small. Also, we may want to set kT t o1.0 and vary gamma. Maybe we also want to vary the head height


## July 11, 2023

We are getting 0 gradients. This is probably because of not using mod rigid body -- we think we vaguely remember something in the shape definition that is discontinuous, and that we specialized for rigid bodies.

Should start next time by doing a diff on mod_rigid_bodey and rigid_body, and then trying to use it where appropriate. Should be careful about this.

## June 29, 2023

Note for today: we are not adding any leg energy function for now

## May 16, 2023
Have the forwrad pass working iwth a simple graph neural network. Next time, we are gonig to optimize over it. The following are some basic notes re. what we'll have to change to optimiez over it:
- The graph network must be initizlied so that it has the right input shape. We should only do this once. So, unlike the previous neergy function case, we will need some type of "getter"/factory for the run dynamics function that will be responsible for initializing the network with the appropriate shape
- in our previous optimizations, we were optimizing over both the energy parameters as well as the parameters describing the spider shape. In part, the latter was because the action of the morse potential is fairly sensitive to the shap eof the catalyst. But, given the expressiveness of NNs, we think it'd be best to just fix the shape. Otherwise, we'd have to deal with compatibilities between the `RigidPointUnion` class in our version of `rigid_body` vs. `jax_md.rigid_body`. So, this can also be in the getter/factory.


## April 12, 2023

Taking stock and next steps:
- we can currently optimize for abducting and stayin gtogether for really and gamma -- this is from treating the morse eps as a log. (With autodiff, maybe with GA).
- we still can't get it to etach
- we computed the energy diffetrence between the attached and detached state. It's lik e200... whereas kt=2.0. This is a problme.
- so, we want to do the following:
  - first,want too understand this difference. Is it the heads interaction with the other vertices? Is it the legs interaction withthe vertices? If the former, how narrow can it be?
  - then, run a couple things by hand and see if we can get away wit ha lower head eps and higher alpha -- or, is there some other (differentiable) potential that would be steeper and allow for a stricter interactoin range, and therefore a better difference -- for example, maybe we can do this with an NN and not a Mors epotential (proof of existence)
  - then, try to optimize using information from the above. probably looks somethin glike penalizing against the energy difference betweeen bound and unbound

## mar 20, 2023

Maybe todo tomorrow?
- maybe setup an optimization that uses the abducting parameters?
- maybe do a stepwise optimization?

TODO:
- could think about adding reinforce term
- fix that eig thing
- could maybe experimeent with higher gammas


## mar. 13, 2023

Weirdly, the gradients are 0 for head_eps and head_alpha on the discret elgs branch. So, we are starting on this new branch to iteratively add things.

We are starting with a key of 10000 with no legs just to get a baseline for the gradients.


## feb 28, 2023
- key 1000 is d5 1e-4 10k steps, legs and both loss terms (no coefficient). legs have same alpha.
- key 1001 is 2 * icos_stays_together (dt1e-3, legs have same alpha, with legs, 1000 steps)
- key 1002 is leg_alpha always 2 for the legs. otherwise same -- dt1e-3, legs, 1000 steps, no coefficient
- key 1003 is no legs
- key 1004 is 1001 wit hbatch size 10 (and redo to confirm we are actually using the coeff)
- key 2000 is with langevin integrator. dt 1e-3. no nlegs. 1k steps. no coeff in loss, but both terms
- key 2001 is above but with legs
- key 2002 is above but also with coeff

## feb 22, 2023
- we can get the catalyst to fall off with an initial separation of 0.1, temp of 2.0, morse_leg_eps=0.0, and running for 5k steps with hand-designed parameters
- added coefficient on the non-explosion part of the loss because it was lower for something that was exploding than good results
- running optimization for varying initial separation, random vs fixed
- possible next steps: batch over different initial separation coefficients, try optimization with different temperatures/energy scales, think about how to get the abducted particle to leave the catalyst, add repulsive cap to spider, try catalyst on different shells (octahedron etc)



## feb 20, 2023
- Added term to the loss function to get the catalyst to detach from the icosahedron
- fixed the leg diameter to be 1.5 throughout the optimization so it couldn't cheat
- fixed issue with leg repulsion: wasn't counting the radius of the vertex
- getting degenerate solution at the moment, may be due to initial separation coefficient
- Played with increasing the temperature; may be causing the shell to fall apart. Should play with the morse_ii_eps vs temperature, though note that increasing morse_ii_eps will make it harder for the catalyst to get the vertex to detach



## Jan. 17, 2023

For next time:
- test on GPU
  - note that we'll need a local version of the jax-md changes that we did. Will likely want to make a conda environment that has its wn install of jax-md
  - changes are `union_to_points` and in `RigidPointUnion.__getitem__`
- setup optimization
  - note that an initial thing we can do to avoid dummy local minima (i.e. having a huge spider catalyst to splay apart the shell vertices via excluded volume) is to make the initialized height of the catalyst a function of its two radii (well, really the max of something like (head_radius - height, leg_radius), as well as a function of base radius
  - note: could setup two optimizations -- one that only optimizes over energy, another that optimizes over both shape and energy


## Jan. 18, 2023

From today
- Tried to setup a training loop but didn't actually run it
  - need to add a batch size (at the least)
- were getting nans for some spider shape stuff. added some noise to msases and got non-nan grads. but then re-ran on cluster and got 0 gradients for some more things. Maybe we didn't originally save the flie or osmetihng? Maybe we should run again locally? Maybe we didn't run for long enough? Some stocahsticity there? Have to experiment
  - maybe should install the github rpeo on a colab and visualize with the new masses. unequal masses may be messing pu the initialization
- once we resolve the above, just setup a training loop and hit go
  -


## Jan. 29, 2023

today we banged our heads against the wall and reminimized the icosahedron and wokred out that we got both really luckky by choosing the wrong vertex to bind that was the problem one, and also unlucky in the sense that our minimization hadn't worked

so, we reminimized. and changed some +'s to -'s, which we are still confused about. This had to be done when doing the dispalcmenets for (i) the vertex shape (so that the patches weren't facing outwards) *and* for the spider head (so that the spider head wasn't facing inwards). This suggests that our optimization from before had the spider head inside and was just splaying everything evyerwhere. It also suggests that our minimized icosahedron from beforee was wrong (i.e. the one that we generated on the cluster)

For next time:
- going forward, esp. as we set up these optimizations, we want to be careful that tarjectories on the cluster correspond with trajectores in colabs/when we visualize them. So, we propose doing the following:
  - go on the colab. Crank up interaction parameters between the spider and the icosahedron. If nothing really happens, that could explain the 0 gradients
  - once we get some interactions, do the same thing on the cluster, pickle th etrajcetory, upload it to colab, and visualize it
  - total pain, but has to be done

## Feb 2, 2023

dear diary, today we tested the gradients some more. we found out that it makes sense for some gradients to be zero depending on the initial parameters. for example, say the legs and head of the spider aren't touchin anything  --  then ofcourse the grad w.r.t. their diameters will be 0.

given this, we don't understand why we got 0s for some of the optmization yesterday. to debug this, we propose the following steps:
- continue the test in __main__ in `simulation.py` to take the grad of th esimulatoin rather than of a single energy evaluatoin. use the parameters we used for ht energy evaluation. if this is fine (i.e. all grads that we expect to be non-zero are non-zero), gthen redo optimization. If optimization doesn't work, probably juts soething wrong there.

Once we are ready (ie.e. aoptimization grads work), should fiddle around in the colab to find a reasonable set of starting parameters.

## Feb 3, 2032

We got some stuff working today. First, we confirmed that gradients (w.r.t. the energy function) propagate through the optimization. However, we realized that the grads w.r.t. the *loss* function are 0 -- we suspect this is due to the loss function being in a local minimum. One cause for beign ina  local minimum could be that the nearest local minimum is this the "explosion", for which the loss is similar/identical (maybe) as in thefully bound state

Some things to address are (i) constraints on the spider/UFO/unidenfied shape, and (ii) constraints on the loss function. These include the following:
- the leg/bond of the spider should have excluded volume. This would prevent the explosion thing because the leg couldn't overlap with the rest of the shell
- we should include additional constraints in the loss function to descibe waht we want. te most immediate is that the rest of the shell should stay together (e.g. the sum of the pairwise distances for the remainiing 11 things). this will also miitigate the explosoin thing, potentially in a better way
constraints of this form could also smooth the landscape

### Part 2

An interesting night -- we found that we had hardcoded all of the values in run_dynamics, but not in run_dynamics_helper. This explained our 0.0 gradients.

We ran our working parameter set in the colab with several keys and initial separation coefficients to midlly evaluate its robustness.

We just launched a bunch of jobs.

Next time, we want to:
- look at what the loss function is telling us about what we're asking for
  - change the loss function accordingly
- consider making the legs repulsive/having some excluded volume


papers to read for next week: diffussion model paper from Baker lab, ICLR paper


## Feb 5, 2032

we want to see random parmeters converge

we have been running some with fixed parameters to see if it could solv eht eproblem of recombining. it oculdn't, so far.

if this doesn't work, a couple things we could try:
- random parameters w/o the extra loss term. confirm that we can converge there
- try differetn optimizers (note: make a flag for this)
