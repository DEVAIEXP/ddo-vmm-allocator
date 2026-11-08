// Regression: run the inherited actual-pool and automatic-selection suites,
// which earlier candidates compiled but did not execute. CPU doubles only.
#include "auto_cpu.cpp"

int main() {
    if (const int pool = unused_inherited_pool_main()) return pool;
    reset();
    if (const int automatic = inherited_automatic_main()) return automatic;
    reset();
    std::cout << "PASS inherited pool and automatic CPU suites\n";
    return 0;
}
