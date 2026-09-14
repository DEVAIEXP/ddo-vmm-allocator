from ddo_vmm_allocator import detect_cuda_vmm_capabilities


def test_capability_detection_is_side_effect_free_and_structured():
    capabilities = detect_cuda_vmm_capabilities()

    assert capabilities.operating_system
    assert isinstance(capabilities.supported_operating_system, bool)
    assert isinstance(capabilities.cuda_bindings_available, bool)
    assert isinstance(capabilities.cuda_core_available, bool)
    assert len(capabilities.python_version) == 2

