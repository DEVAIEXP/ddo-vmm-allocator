// Actual pool implementation; all Driver symbols are CPU doubles, never a DLL.
#define main unused_inherited_pool_main
#include "pool_cpu.cpp"
#undef main
#include <set>
#include <thread>

std::map<CUdeviceptr,size_t> native_va;
std::map<CUmemGenericAllocationHandle,size_t> native_physical;
std::map<CUdeviceptr,CUmemGenericAllocationHandle> native_mapped;
CUdeviceptr next_va = 0x1000000;
CUmemGenericAllocationHandle next_handle = 1;
size_t native_creates = 0;
size_t native_accesses = 0, native_unmaps = 0, native_releases = 0;
size_t native_fault_remaining = SIZE_MAX;
std::string native_fault;
int native_error = 600;
std::function<void()> native_callback;
std::function<void()> native_access_failure;
std::function<void(const char*)> native_observer;
int driver_step(const char* operation) {
    check(pool_mutex.try_lock(), "Driver work outside pool lock"); pool_mutex.unlock();
    if (native_observer) native_observer(operation);
    if (native_callback) { auto callback=std::move(native_callback); callback(); }
    if (native_fault == operation && native_fault_remaining) {
        if (native_fault_remaining != SIZE_MAX) --native_fault_remaining;
        return native_error;
    }
    return 0;
}
CUresult CUDAAPI cuMemGetAllocationGranularity(size_t* out,const CUmemAllocationProp*,CUmemAllocationGranularity_flags) {
    *out=fixture_page; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* out,size_t extent,size_t,CUdeviceptr,unsigned long long) {
    if (const auto error=driver_step("reserve")) return static_cast<CUresult>(error);
    *out=next_va; native_va[*out]=extent; next_va+=extent+fixture_page; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr address,size_t) {
    if (const auto error=driver_step("free_va")) return static_cast<CUresult>(error);
    native_va.erase(address); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* out,size_t bytes,const CUmemAllocationProp*,unsigned long long) {
    ++native_creates;
    if (const auto error=driver_step("create")) return static_cast<CUresult>(error);
    *out=next_handle++; native_physical[*out]=bytes; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemMap(CUdeviceptr address,size_t,size_t,CUmemGenericAllocationHandle handle,unsigned long long) {
    if (const auto error=driver_step("map")) return static_cast<CUresult>(error);
    native_mapped[address]=handle; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr,size_t,const CUmemAccessDesc*,size_t) {
    ++native_accesses;
    const auto status = driver_step("access");
    if (status && native_access_failure) {
        auto callback = std::move(native_access_failure); callback();
    }
    return static_cast<CUresult>(status);
}
CUresult CUDAAPI cuMemUnmap(CUdeviceptr address,size_t) {
    ++native_unmaps;
    if (const auto error=driver_step("unmap")) return static_cast<CUresult>(error);
    native_mapped.erase(address); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    ++native_releases;
    if (const auto error=driver_step("release")) return static_cast<CUresult>(error);
    native_physical.erase(handle); return CUDA_SUCCESS;
}
size_t physical() {
    size_t total=0; for (const auto& entry:native_physical) total+=entry.second; return total;
}
void reset() {
    // Fault-fixture isolation only: archive ownership, never invoke Driver cleanup.
    static auto* retained_backings=new std::vector<std::shared_ptr<Backing>>;
    static auto* retained_storage=new std::vector<std::shared_ptr<DriverStorage>>;
    for (auto& owner:backings) retained_backings->push_back(owner);
    for (auto& owner:driver_storage_owners) retained_storage->push_back(owner);
    records.clear(); backings.clear(); driver_storage_owners.clear();
    native_va.clear(); native_physical.clear(); native_mapped.clear();
    fixture_events.clear(); native_fault.clear(); native_callback={}; native_access_failure={}; native_observer={}; auto_pressure={};
    events_complete=false; query_failure=dependency_failure=false;
    region_attempts=pressure_calls=native_creates=0;
    native_accesses=native_unmaps=native_releases=0; native_fault_remaining=SIZE_MAX;
    pressure_status=DDO_VMM_SUCCESS; region_status=DDO_VMM_OUT_OF_MEMORY;
    auto_segmented=false; allocation_started=false; auto_recovery_used=false;
    auto_recovery_error=0; native_error=600;
    auto_sync_calls=0; fixture_barrier_status=CUDA_SUCCESS;
    fixture_barrier_callback={}; fixture_current_context=fixture_context();
    // Capacity-transaction fixtures.
    fixture_event_sync_status=CUDA_ERROR_NOT_READY; event_sync_calls=ctx_sync_calls=0; sync_lock_violation=false;
    fixture_capture_status=CU_STREAM_CAPTURE_STATUS_NONE; fixture_capture_query=CUDA_SUCCESS;
    arena_pages=arena_busy=arena_count_calls=arena_evict_calls=0; arena_page_bytes=fixture_page;
    arena_count_override=SIZE_MAX; arena_fail=arena_lock_violation=false;
    capacity_report_recorded=false; last_capacity_report={};
}
// Last capacity-transaction record through the experimental export.
std::array<uint64_t,31> refusal_record() {
    std::array<uint64_t,31> out{}; out.fill(0);
    check(ddo_exp_segmented_last_refusal(out.data(),out.size())==0,"refusal record readable");
    return out;
}
void* allocate(size_t bytes) { return ddo_vmm_pool_malloc(bytes,0,reinterpret_cast<cudaStream_t>(fixture_stream())); }
void retire(void* pointer) { ddo_vmm_pool_free(pointer,reinterpret_cast<cudaStream_t>(fixture_stream())); }
void enable() { check(ddo_exp_segmented_configure_auto(1)==0,"cold opt-in"); }
void closed() {
    events_complete=true;
    check(!ddo_vmm_pool_trim_idle(),"query-only explicit trim");
    check(backings.empty() && records.empty() && driver_storage_owners.empty()
          && native_va.empty() && native_physical.empty() && native_mapped.empty(),"zero final ownership");
}
std::array<uint64_t,15> stats() {
    std::array<uint64_t,15> out{};
    check(!ddo_vmm_pool_stats(out.data(),out.size()),"legacy stats available");
    check(out[1]==out[3]+out[4]+out[5]+out[6],"unique physical partition");
    return out;
}
struct Split {
    void *left,*middle,*right;
    Split() {
        auto whole=allocate(4*fixture_page); check(whole!=nullptr,"whole extent"); retire(whole);
        left=allocate(fixture_page); middle=allocate(2*fixture_page); right=allocate(fixture_page);
        check(left==whole && middle && right,"split original same VA");
    }
    void finish() { retire(left); retire(right); }
};
int inherited_automatic_main() {
    try {
        reset();
        check(!auto_segmented && !allocate(fixture_page) && region_attempts>0
            && native_va.empty(),"default off preserves monolithic route");
        check(ddo_exp_segmented_configure_auto(1)!=0,"failed first allocation freezes config");
        reset();
        check(ddo_exp_segmented_configure_auto(2)!=0,"invalid cold config"); enable();
        auto full=allocate(4*fixture_page);
        check(full && !region_attempts && native_va.size()==1 && physical()==4*fixture_page,"automatic miss uses native factory");
        check(ddo_exp_segmented_configure_auto(0)!=0,"late opt-out refused");
        retire(full); check(physical()==4*fixture_page && native_va.size()==1,"pending retained, no forced wait");
        closed();

        reset(); enable();
        auto zero=allocate(0); auto tail=allocate(fixture_page+1);
        check(zero && tail && physical()==3*fixture_page,"zero and tail normalization");
        check(stats()[1]==3*fixture_page,"physical excludes VA gaps");
        retire(zero); retire(tail); closed();
        reset(); enable();
        check(!allocate(4097*fixture_page) && native_va.empty() && !region_attempts,"owner page limit before Driver; no fallback");

        reset(); enable(); Split split; retire(split.middle); events_complete=true;
        auto recovered=recover_segmented_context(0,fixture_context());
        check(recovered.released==2 && physical()==2*fixture_page,"middle dead pages released");
        auto report=stats(); check(report[1]==2*fixture_page && report[3]==2*fixture_page && !report[4],"legacy physical partition excludes discarded middle");
        uint64_t detailed[10]{};
        check(!ddo_exp_segmented_stats(detailed,10) && detailed[2]==4*fixture_page && detailed[3]==2*fixture_page,"VA remains separate");
        auto fresh=allocate(2*fixture_page);
        check(fresh==split.middle && physical()==4*fixture_page,"automatic rematerialization same VA");
        split.finish(); retire(fresh); closed();

        reset(); enable();
        auto a=allocate(512),b=allocate(512); check(a && b && physical()==fixture_page,"small leases share one physical unit");
        retire(a); events_complete=true; recovered=recover_segmented_context(0,fixture_context());
        check(!recovered.released,"live intersection prevents unit release");
        report=stats(); check(report[1]==fixture_page && report[3]==512 && report[4]==512,"byte partition clipped to resident unit");
        retire(b); closed();

        reset(); enable(); Split pressure; retire(pressure.middle); events_complete=true;
        auto_pressure=[](size_t bytes) { return physical()+bytes>6*fixture_page ? DDO_VMM_OUT_OF_MEMORY : DDO_VMM_SUCCESS; };
        const auto calls=pressure_calls; auto extra=allocate(4*fixture_page);
        check(extra && pressure_calls==calls+2 && physical()==6*fixture_page
            && auto_recovery_used && !region_attempts,"one admission retry after actual other-owner release");
        check(records.size()==3,"live neighbors plus new lease only");
        auto_pressure={}; pressure.finish(); retire(extra); closed();

        reset(); enable(); auto live=allocate(fixture_page);
        auto_pressure=[](size_t) { return DDO_VMM_OUT_OF_MEMORY; };
        const auto no_release_calls=pressure_calls;
        check(!allocate(2*fixture_page) && pressure_calls==no_release_calls+1 && physical()==fixture_page,"no eligible release means no admission retry");
        auto_pressure={}; retire(live); closed();

        reset(); enable(); Split nonoom; retire(nonoom.middle); events_complete=true;
        pressure_status=DDO_VMM_CUDA_ERROR;
        check(!allocate(4*fixture_page) && !auto_recovery_used && physical()==4*fixture_page,"noncapacity admission failure never reclaims");
        pressure_status=DDO_VMM_SUCCESS; nonoom.finish(); closed();

        reset(); enable(); Split failed_recovery; retire(failed_recovery.middle); events_complete=true;
        native_fault="unmap"; pressure_status=DDO_VMM_OUT_OF_MEMORY;
        const auto failure_calls=pressure_calls;
        check(!allocate(4*fixture_page) && auto_recovery_error==600 && pressure_calls==failure_calls+1
            && physical()==4*fixture_page,"original recovery error retained; no retry or credit");
        check(backings.front()->retiring && backings.front()->segmented_error==600,"failed owner remains poisoned");

        for (const auto operation:{"create","map","access","unmap","release","free_va"}) {
            reset(); enable();
            if (std::string(operation)=="create" || std::string(operation)=="map" || std::string(operation)=="access") {
                native_fault=operation;
                check(!allocate(2*fixture_page) && records.empty(),"partial materialization publishes no tensor");
            } else {
                auto pointer=allocate(2*fixture_page); retire(pointer); events_complete=true; native_fault=operation;
                check(ddo_vmm_pool_trim_idle()!=0,"checked trim propagates original error");
            }
            check(!driver_storage_owners.empty() && !backings.empty()
                && backings.front()->retiring && backings.front()->segmented->error()==600,"partial failure retains registry and original error");
            std::array<uint64_t,15> unavailable; unavailable.fill(99);
            check(ddo_vmm_pool_stats(unavailable.data(),15)!=0 && unavailable[0]==99,"unavailable physical report never becomes zero");
        }

        reset(); enable(); auto big=allocate(80*fixture_page); retire(big);
        auto first70=allocate(70*fixture_page); events_complete=true;
        auto pass1=recover_segmented_context(0,fixture_context());
        auto pass2=recover_segmented_context(0,fixture_context());
        check(pass1.scanned==64 && !pass1.released && pass2.scanned==64 && pass2.released==10,"bounded page cursor reaches late dead units");
        retire(first70); closed();

        reset(); enable(); auto one=allocate(80*fixture_page),two=allocate(80*fixture_page);
        retire(one); retire(two); auto live1=allocate(70*fixture_page),live2=allocate(70*fixture_page); events_complete=true;
        size_t released=0;
        for (int i=0;i<4;++i) {
            const auto part=recover_segmented_context(0,fixture_context());
            check(part.scanned<=64 && part.released<=16,"aggregate recovery bounds unchanged"); released+=part.released;
        }
        check(released==20,"owner and page cursors prevent first-owner starvation");
        retire(live1); retire(live2); closed();

        reset(); enable();
        native_callback=[] { check(ddo_exp_segmented_configure_auto(0)!=0,"in-flight registration freezes configuration"); };
        auto pointer=allocate(fixture_page); check(pointer!=nullptr,"registration callback reentry"); retire(pointer); closed();
        reset(); enable();
        native_callback=[] {
            std::thread worker([] {
                check(ddo_exp_segmented_configure_auto(0)!=0,"concurrent cold config cannot change in-flight policy");
                check(ddo_exp_segmented_close(0)==0 && driver_storage_owners.size()==1,
                      "concurrent close retains unpublished registry ownership");
            }); worker.join();
        };
        pointer=allocate(fixture_page); check(pointer!=nullptr,"concurrent registration lifetime retained");
        retire(pointer); closed();
        reset(); enable(); pressure_status=DDO_VMM_OUT_OF_MEMORY;
        check(!allocate(fixture_page) && driver_storage_owners.size()==1,"refused growth retains empty VA owner");
        check(!trim_segmented_idle(true) && native_va.empty() && driver_storage_owners.empty(),
              "zero-physical idle owner does not leak VA below cache threshold");
        reset();
        {
            auto owner=std::make_shared<Backing>(fixture_base,fixture_page,0,fixture_context(),next_generation++);
            backings.push_back(owner);
            check(ddo_exp_segmented_configure_auto(1)!=0,"pre-existing monolithic owner refuses cold enable");
        }
        reset(); enable(); pointer=allocate(fixture_page); retire(pointer); query_failure=true;
        check(ddo_vmm_pool_trim_idle()!=0 && !native_va.empty() && physical()==fixture_page,
              "failed consumer query retains complete physical ownership");
        check(backings.front()->segmented_error==static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS),
              "original failed consumer code retained");
        reset();
        { AutoRequestScope outer; auto_recovery_used=true; AutoRequestScope inner;
          check(auto_recovery_used,"nested request cannot reset recovery quota"); }
        check(!auto_request_depth,"request scope released on return");
        check(!auto_sync_calls,"no segmented path synchronized pending consumers");
        SpanLedger range(2048,0,77);
        check(!range.range_snapshot(SIZE_MAX,1) && !range.range_snapshot(2047,2),"range observation validates bounds without overflow");
        auto reservation=range.reserve(513,0);
        check(reservation && range.publish(*reservation,true),"range observation fixture");
        const auto clipped=range.range_snapshot(512,1024);
        check(clipped && clipped->mapped==1024 && clipped->leased==512 && clipped->available==512,
              "range partition clips an aligned lease at both boundaries");
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"),"no CUDA library loaded");
        std::cout<<"PASS automatic pool CPU: "<<checks<<" checks; Driver symbols substituted, no GPU\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
