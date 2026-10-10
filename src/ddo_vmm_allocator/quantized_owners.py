"""Opt-in composite ownership using the existing native-view lifetime protocol.

This helper does not install hooks or choose residency. Caller supplies complete
payload views and the final-consumer event; CUDA execution remains unvalidated.
"""

import torch

from .quantized_payload import _supported_backend, describe_float8_weight_only_cpu
from .tensor_owners import TensorOwnerGroup


class Float8WeightOwnerGroup(TensorOwnerGroup):
    def __init__(self, weight, owners):
        self.weight_spec = describe_float8_weight_only_cpu(weight)
        super().__init__(weight, "parameter", owners)
        self.nbytes = self.weight_spec.physical_bytes

    def _validate_publication_view(self, view):
        if not isinstance(view, torch.Tensor) or view.device.type not in ("cpu", "cuda"):
            raise ValueError("Cannot publish retired meta payloads")
        self.weight_spec.validate_wrapper_view(view)

    def _make_retirement_placeholder(self):
        payloads = {item.name: torch.empty(item.shape, dtype=item.dtype, device="meta")
                    for item in self.weight_spec.payloads}
        view = self.weight_spec.reconstruct_views(payloads)
        return torch.nn.Parameter(view, requires_grad=False)

    def publish_payload_views(self, payloads):
        """Publish both views transactionally to all captured parameter aliases."""
        view = self.weight_spec.reconstruct_views(payloads)
        return self.publish(view)


def _capture_float8_component_owners_cpu(root):
    """Index aliases once during discovery; never retain this index at runtime."""
    cls = _supported_backend()
    if not isinstance(root, torch.nn.Module):
        raise ValueError("An eval component with exclusive lifecycle control is required")
    records, storages = {}, {}
    for path, module in root.named_modules(remove_duplicate=False):
        if module.training:
            raise ValueError("An eval component with exclusive lifecycle control is required")
        if torch.nn.utils.parametrize.is_parametrized(module):
            raise ValueError("Parametrized owners are unsupported")
        for kind, table in (("parameter", module._parameters), ("buffer", module._buffers)):
            for key, tensor in table.items():
                if tensor is None:
                    continue
                if kind == "buffer" and type(tensor) is not torch.Tensor:
                    raise ValueError("Subclass buffer owners are unsupported")
                if kind == "parameter" and type(tensor) not in (cls, torch.nn.Parameter):
                    raise ValueError("Unsupported native tensor subclass")
                identity = id(tensor)
                if identity not in records:
                    records[identity] = (tensor, kind, {})
                    if tensor.device.type == "cpu":
                        payloads = (tensor.qdata, tensor.scale) if type(tensor) is cls else (tensor,)
                        for payload in payloads:
                            storages.setdefault(payload.untyped_storage().data_ptr(), set()).add(identity)
                _, previous_kind, owners = records[identity]
                if kind != previous_kind:
                    raise ValueError("A buffer aliases composite payload storage")
                entry = owners.setdefault((id(module), key), [module, key, []])
                entry[2].append(f"{path}.{key}" if path else key)
    groups = {}
    for identity, (tensor, kind, owners) in records.items():
        if type(tensor) is cls:
            group = Float8WeightOwnerGroup(tensor, owners.values())
            for payload in group.weight_spec.payloads:
                pointer = getattr(tensor, payload.name).untyped_storage().data_ptr()
                if storages[pointer] != {identity}:
                    raise ValueError("Distinct parameter or buffer objects share composite payload storage")
        else:
            group = TensorOwnerGroup(tensor, kind, owners.values())
        groups[identity] = group
    return groups


def capture_float8_weight_owners_cpu(root, weight):
    """Capture explicit wrapper aliases, refusing other shared storage owners.

    No architecture-specific detection or automatic quantized component opt-in.
    The caller must include every live module owner in root and retain exclusive
    lifecycle control. Arbitrary escaped references cannot be discovered.
    """
    spec = describe_float8_weight_only_cpu(weight)
    if not isinstance(root, torch.nn.Module) or any(module.training for module in root.modules()):
        raise ValueError("An eval component with exclusive lifecycle control is required")
    source_storages = {getattr(weight, item.name).untyped_storage().data_ptr()
                       for item in spec.payloads}
    owners = {}
    for path, module in root.named_modules(remove_duplicate=False):
        if torch.nn.utils.parametrize.is_parametrized(module):
            raise ValueError("Parametrized owners are unsupported")
        for key, tensor in module._parameters.items():
            if tensor is weight:
                entry = owners.setdefault((id(module), key), [module, key, []])
                entry[2].append(f"{path}.{key}" if path else key)
            elif tensor is not None and tensor.device.type == "cpu":
                candidates = (tensor.qdata, tensor.scale) if type(tensor) is type(weight) else (tensor,)
                for candidate in candidates:
                    if type(candidate) in (torch.Tensor, torch.nn.Parameter):
                        if candidate.untyped_storage().data_ptr() in source_storages:
                            raise ValueError("Distinct parameter objects share composite payload storage")
                    else:
                        raise ValueError("Other tensor subclasses require explicit ownership support")
        for tensor in module._buffers.values():
            if tensor is None or tensor.device.type != "cpu":
                continue
            if type(tensor) is not torch.Tensor:
                raise ValueError("Subclass buffer owners are unsupported")
            if tensor.untyped_storage().data_ptr() in source_storages:
                raise ValueError("A buffer aliases composite payload storage")
    return Float8WeightOwnerGroup(weight, owners.values())
