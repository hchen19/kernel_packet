// Shared by the C++ kernels: k!, parallel_for over std::thread, dispatch on q.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "../common.hpp"

namespace skp {

// k! correctly rounded, k = 0..60
constexpr double FACT[61] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0,
    6227020800.0, 87178291200.0, 1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
    1.21645100408832e+17, 2.43290200817664e+18, 5.109094217170944e+19, 1.1240007277776077e+21,
    2.585201673888498e+22, 6.204484017332394e+23, 1.5511210043330986e+25, 4.0329146112660565e+26,
    1.0888869450418352e+28, 3.0488834461171387e+29, 8.841761993739702e+30, 2.6525285981219107e+32,
    8.222838654177922e+33, 2.631308369336935e+35, 8.683317618811886e+36, 2.9523279903960416e+38,
    1.0333147966386145e+40, 3.7199332678990125e+41, 1.3763753091226346e+43, 5.230226174666011e+44,
    2.0397882081197444e+46, 8.159152832478977e+47, 3.345252661316381e+49, 1.40500611775288e+51,
    6.041526306337383e+52, 2.658271574788449e+54, 1.1962222086548019e+56, 5.502622159812089e+57,
    2.5862324151116818e+59, 1.2413915592536073e+61, 6.082818640342675e+62, 3.0414093201713376e+64,
    1.5511187532873822e+66, 8.065817517094388e+67, 4.2748832840600255e+69, 2.308436973392414e+71,
    1.2696403353658276e+73, 7.109985878048635e+74, 4.0526919504877214e+76, 2.3505613312828785e+78,
    1.3868311854568984e+80, 8.32098711274139e+81};


inline int num_workers(int64_t work, int64_t grain) {
    return static_cast<int>(std::clamp<int64_t>(work / grain, 1, kp::num_threads()));
}

// fn(lo, hi, t) on `workers` contiguous ranges of [0, total)
template <typename Fn>
void parallel_for(int64_t total, int workers, Fn&& fn) {
    workers = static_cast<int>(std::min<int64_t>(workers, total));
    if (workers <= 1) {
        fn(int64_t(0), total, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (int t = 1; t < workers; ++t)
        threads.emplace_back([&fn, total, workers, t]() { fn(total * t / workers, total * (t + 1) / workers, t); });
    fn(int64_t(0), total / workers, 0);
    for (auto& th : threads) th.join();
}

}  // namespace skp

#define DISPATCH(q, f, ...) (q == 1 ? f<1>(__VA_ARGS__) : q == 2 ? f<2>(__VA_ARGS__) : f<3>(__VA_ARGS__))
