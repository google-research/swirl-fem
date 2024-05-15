// Mesh on [-1, 1] x [-π, π] with a periodic connection the y direction

lc = 0.55;
pi = 3.14159265358979323846;

Point(1) = {-1, -pi, 0, lc};
Point(2) = {1, -pi,  0, lc};
Point(3) = {1, pi, 0, lc};
Point(4) = {-1,  pi, 0, lc};

Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {4, 1, -2, 3};
Plane Surface(1) = {1};

Physical Curve("left") = {4};
Physical Curve("right") = {2};
Physical Curve("top") = {3};
Physical Curve("bottom") = {1};
Physical Surface("mesh") = {1};

Periodic Curve {3} = {-1} Translate {0, 2 * pi, 0};

Mesh.Algorithm = 8;

Mesh.RecombinationAlgorithm = 3;

Mesh 2;
RecombineMesh;
Save "kovasznay.msh";
