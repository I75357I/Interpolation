#include "NearestSimplexInterpolationMD.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    // 2D test: f(x,y) = x+2y over a triangle
    {
        std::vector<std::array<double, 3>> train2d = {
            std::array<double,3>{0.0, 0.0, 0.0},
            std::array<double,3>{1.0, 0.0, 1.0},
            std::array<double,3>{0.0, 1.0, 2.0}
        };
        NearestSimplexInterpolationMD::Interpolator<2, double> interp2d(train2d);
        std::array<double, 2> pt2d = { 0.25, 0.75 };
        double val2d = interp2d.interpolate(pt2d);
        std::cout << "2D interpolation at (0.25,0.75): " << val2d << std::endl;
    }

    // 3D test: f(x,y,z) = x+4y+z over a tetrahedron
    {
        std::vector<std::array<double, 4>> train3d = {
            std::array<double,4>{0.0, 0.0, 0.0, 0.0},
            std::array<double,4>{1.0, 0.0, 0.0, 1.0},
            std::array<double,4>{0.0, 1.0, 0.0, 4.0},
            std::array<double,4>{0.0, 0.0, 1.0, 1.0}
        };
        NearestSimplexInterpolationMD::Interpolator<3, double> interp3d(train3d);
        std::array<double, 3> pt3d = { 0.5, 0.2, 0.3 };
        double val3d = interp3d.interpolate(pt3d);
        std::cout << "3D interpolation at (0.5,0.2,0.3): " << val3d << std::endl;
    }

    // 5D test: f(x1,..,x5) = x1+2x2+x3 over a 5-simplex
    {
        std::vector<std::array<double, 6>> train5d = {
            std::array<double,6>{0,0,0,0,0, 0.0},
            std::array<double,6>{1,0,0,0,0, 1.0},
            std::array<double,6>{0,1,0,0,0, 2.0},
            std::array<double,6>{0,0,1,0,0, 1.0},
            std::array<double,6>{0,0,0,1,0, 0.0},
            std::array<double,6>{0,0,0,0,1, 0.0}
        };
        NearestSimplexInterpolationMD::Interpolator<5, double> interp5d(train5d);
        std::array<double, 5> pt5d = { 0.72, 0.13, 0.05, 0.05, 0.05 };
        double val5d = interp5d.interpolate(pt5d);
        std::cout << "5D interpolation at (0.72,0.13,0.05,0.05,0.05): " << val5d << std::endl;
    }
    return 0;
}