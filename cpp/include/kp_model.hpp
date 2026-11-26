// Handle of the C interface: the stable KPs (default, stable/stable_kp.hpp) or the original KernelPacket
// (original/kernel_packets.hpp).
#pragma once
#include <memory>

#include "original/kernel_packets.hpp"
#include "stable/stable_kp.hpp"

namespace kp {

struct Model {
    std::unique_ptr<StableKP> stable;
    std::unique_ptr<KernelPacket> original;
};

}  // namespace kp
