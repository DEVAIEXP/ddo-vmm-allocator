// Actual segmented pool, CPU Driver doubles only. No hardware is initialized.
#include "auto_cpu.cpp"

void fault(int status, size_t count=SIZE_MAX) {
    native_fault="access"; native_error=status; native_fault_remaining=count;
}
void failed_target(int status, size_t accesses, size_t barriers) {
    check(native_accesses==accesses && auto_sync_calls==barriers,"exact access/barrier call counts");
    check(records.empty() && backings.size()==1 && backings.back()->retiring
          && backings.back()->segmented_error==status,"original/terminal status retained; no publication");
    check(native_va.size()==1 && native_physical.size()==1 && native_mapped.size()==1
          && !native_unmaps && !native_releases,"failed target VA/handle/map retained");
    uint64_t report[15]; std::fill(std::begin(report),std::end(report),99);
    check(ddo_vmm_pool_stats(report,15)!=0 && report[0]==99,"failed accounting is unavailable, never fake zero");
}

int main() {
    try {
        // Successful first assignment must remain free of any barrier.
        reset(); enable(); auto clean=allocate(fixture_page);
        check(clean && native_accesses==1 && !auto_sync_calls && !auto_recovery_used,"ordinary success unchanged");
        retire(clean); closed();
        std::cout << "PASS success without synchronization\n";

        reset(); enable(); fault(CUDA_ERROR_NOT_READY,1);
        std::vector<std::string> sequence;
        native_observer=[&](const char* operation) {
            if (std::string(operation)=="access") sequence.emplace_back("access");
        };
        CUdeviceptr failed_va=0; CUmemGenericAllocationHandle failed_handle=0;
        native_access_failure=[&] {
            check(native_mapped.size()==1 && records.empty(),"failed target mapped but unpublished");
            failed_va=native_mapped.begin()->first; failed_handle=native_mapped.begin()->second;
        };
        fixture_barrier_callback=[&] {
            sequence.emplace_back("barrier");
            check(native_mapped.at(failed_va)==failed_handle && !native_unmaps && !native_releases,
                  "same mapping/handle survives context barrier");
        };
        auto recovered=allocate(fixture_page);
        check(recovered && reinterpret_cast<uintptr_t>(recovered)==failed_va && !auto_recovery_used,
              "same VA published only after NOT_READY continuation");
        check(sequence==std::vector<std::string>{"access","barrier","access"}
              && native_creates==1 && native_accesses==2 && auto_sync_calls==1,"exact helper order; no recreation");
        check(native_mapped.at(failed_va)==failed_handle,"original handle preserved");
        native_observer={}; retire(recovered); closed();
        std::cout << "PASS NOT_READY barrier and same-mapping continuation\n";

        for (CUresult status : {CUDA_ERROR_ILLEGAL_ADDRESS,CUDA_ERROR_NOT_READY,CUDA_ERROR_INVALID_CONTEXT}) {
            reset(); enable(); fault(CUDA_ERROR_NOT_READY,1); fixture_barrier_status=status;
            check(!allocate(fixture_page),"failed barrier refuses allocation");
            failed_target(status,1,1);
        }
        std::cout << "PASS three barrier failures preserve ownership\n";

        reset(); enable(); fault(CUDA_ERROR_NOT_READY);
        check(!allocate(fixture_page),"persistent NOT_READY stops");
        failed_target(CUDA_ERROR_NOT_READY,2,1);
        std::cout << "PASS persistent NOT_READY bounded to one continuation\n";

        // NOT_READY -> OOM now enters the capacity transaction; that contract
        // lives in transaction_cpu. A non-capacity continuation stays terminal.
        {
            const int status = static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS);
            reset(); enable(); fault(CUDA_ERROR_NOT_READY,1);
            native_access_failure=[status] { native_error=status; native_fault_remaining=SIZE_MAX; };
            check(!allocate(fixture_page) && !auto_recovery_used,"no recovery after NOT_READY branch");
            failed_target(status,2,1);
        }
        std::cout << "PASS NOT_READY then illegal-address does not chain\n";

        reset(); enable(); fault(CUDA_ERROR_NOT_READY,1);
        native_access_failure=[] { fixture_current_context=reinterpret_cast<CUcontext>(uintptr_t{0x300}); };
        check(!allocate(fixture_page),"context switch before barrier refuses");
        fixture_current_context=fixture_context(); failed_target(CUDA_ERROR_INVALID_CONTEXT,1,0);
        std::cout << "PASS identity checked before barrier\n";

        reset(); enable(); fault(CUDA_ERROR_NOT_READY,1);
        fixture_barrier_callback=[] { fixture_current_context=reinterpret_cast<CUcontext>(uintptr_t{0x300}); };
        check(!allocate(fixture_page),"context switch after barrier refuses continuation");
        fixture_current_context=fixture_context(); failed_target(CUDA_ERROR_INVALID_CONTEXT,1,1);
        std::cout << "PASS identity rechecked before continuation\n";

        for (int status : {static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS),static_cast<int>(CUDA_ERROR_INVALID_CONTEXT)}) {
            reset(); enable(); fault(status);
            check(!allocate(fixture_page) && !auto_recovery_used,"unrelated failures remain terminal");
            failed_target(status,1,0);
        }
        std::cout << "PASS other failures remain terminal\n";

        reset();
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"),"no GPU library loaded");
        std::cout << "PASS 11 status-alignment CPU scenarios; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; reset(); return 1;
    }
}
