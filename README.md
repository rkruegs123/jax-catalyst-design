# jax-catalyst-design

Code for designing **spider catalysts**

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

