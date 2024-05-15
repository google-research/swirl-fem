// Unit interval with 16 elements

lc = 0.066;

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0,  0, lc};

Line(1) = {1, 2};

Mesh.Algorithm = 8;

Mesh 1;
Save "line1d.msh";
