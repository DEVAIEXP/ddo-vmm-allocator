# Native PyTorch adapter

This directory is reserved for a small native shared library that will implement the PyTorch allocator integration.

It is intentionally absent in the scaffold phase.  The portable VMM core is first validated through CUDA Driver API probes on Windows and Linux.  The adapter will expose a narrow C ABI so that Python policy remains portable while only allocation interception is native.

The implementation must be authored from the CUDA Driver API documentation and this repository's own tests.  It must not derive from GPL allocator implementations.

