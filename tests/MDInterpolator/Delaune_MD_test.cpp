#include "DelauneInterpolation_MD.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    // 2D: f(x, y) = x + y
    std::vector<std::array<double, 3>> train2d = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    DelauneInterpolationMD::Interpolator<2, double> interp2(train2d);
    std::array<double, 2> pt2 = { 0.3, 0.7 };
    double val2 = interp2.interpolate(pt2);
    std::cout << "2D interpolation at (0.3,0.7): " << val2 << " (expected 1.0)\n";
    assert(std::fabs(val2 - 1.0) < 1e-6);

    // 3D: f(x, y, z) = x + y + z
    std::vector<std::array<double, 4>> train3d = {
        {0.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 1.0},
		{0.0, 0.0, 1.0, 1.0}

    };
    DelauneInterpolationMD::Interpolator<3, double> interp3(train3d);
    std::array<double, 3> pt3 = { 0.2, 0.3, 0.5 };
    double val3 = interp3.interpolate(pt3);
    std::cout << "3D interpolation at (0.2,0.3,0.5): " << val3 << " (expected 1.0)\n";
    assert(std::fabs(val3 - 1.0) < 1e-6);

    // 4D: f(x, y, z, w) = x + y + z + w
    std::vector<std::array<double, 5>> train4d = {
        {0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0, 0.0, 1.0},
        {0.0, 0.0, 0.0, 1.0, 1.0},
    };
    DelauneInterpolationMD::Interpolator<4, double> interp4(train4d);
    std::array<double, 4> pt4 = { 0.1, 0.2, 0.3, 0.4 };
    double val4 = interp4.interpolate(pt4);
    std::cout << "4D interpolation at (0.1,0.2,0.3,0.4): " << val4 << " (expected 1.0)\n";
    assert(std::fabs(val4 - 1.0) < 1e-6);

    // 5D: f(x, y, z, w, u) = x + y + z + w + u
    std::vector<std::array<double, 6>> train5d = {
        {0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 1},
        {0, 0, 1, 0, 0, 1},
        {0, 0, 0, 1, 0, 1},
        {0, 0, 0, 0, 1, 1}
    };
    DelauneInterpolationMD::Interpolator<5, double> interp5(train5d);
    std::array<double, 5> pt5 = { 0.1, 0.2, 0.2, 0.2, 0.2 };
    double val5 = interp5.interpolate(pt5);
    std::cout << "5D interpolation at (0.1,0.2,0.2,0.2,0.2): " << val5 << " (expected 0.9)\n";
    assert(std::fabs(val5 - 0.9) < 1e-6);

    std::cout << "All dimensional tests passed." << std::endl;
    return 0;
}