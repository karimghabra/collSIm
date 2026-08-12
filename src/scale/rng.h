#pragma once
#include <cstdint>
#include <cmath>

// Counter-based RNG for the scale ladder. Deliberately the same PCG family the
// GPU kernels use, so a CPU-side rung and a GPU-side rung fed identical seeds
// produce comparable noise statistics and a replicate can be reproduced from
// (seed, step) alone rather than from a serialised generator state.

namespace scale {

class Rng {
public:
    explicit Rng(uint64_t seed = 1234u) : s_(seed * 6364136223846793005ull + 1442695040888963407ull) {}

    uint64_t next() {
        s_ = s_ * 6364136223846793005ull + 1442695040888963407ull;
        uint64_t x = s_;
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return x;
    }

    // (0,1) open at both ends: the log in the Box-Muller transform must never
    // see an exact zero.
    double uniform() { return (double(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

    int index(int n) { return int(uniform() * n) % n; }

    double normal() {
        if (haveSpare_) { haveSpare_ = false; return spare_; }
        double u1 = uniform(), u2 = uniform();
        double m = std::sqrt(-2.0 * std::log(u1));
        spare_ = m * std::sin(6.283185307179586 * u2);
        haveSpare_ = true;
        return m * std::cos(6.283185307179586 * u2);
    }

private:
    uint64_t s_;
    double spare_ = 0;
    bool haveSpare_ = false;
};

}  // namespace scale
