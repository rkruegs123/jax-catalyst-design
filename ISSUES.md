# Known issues

1. When doing the dodecahedron, we noticed that we were doing the legs wrong -- the sigma should really be `leg_radius + vertex_radius` rather than `2 * leg_radius`. Even though this was potentially correct, for the wrong reasons, in the octahedron and icosahedron (if vertex radius and leg radius were the same), we should fix it in the code for correctness
2. We should do the icosahedron orientation in the same way that we do it for the octahedron
