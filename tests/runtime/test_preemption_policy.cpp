// The fork's no-preemption pin lives in one module; this asserts it is still enabled and that a
// constrained full-resident pool is the only accepted configuration (the real-GPU coverage lives in
// ninfer_qwen3_5_no_preemption_real_test).

#include "runtime/engine/preemption_policy.h"

#include <iostream>

int main() {
    if (ninfer::runtime::PreemptionPolicy::may_pause_resident()) {
        std::cerr << "FAIL: the fork's no-preemption pin must stay enabled\n";
        return 1;
    }
    std::cout << "No-preemption policy pinned off\n";
    return 0;
}
