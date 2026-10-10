"""Alias-preserving publication/lifetime contracts for native tensor views.

This is ownership bookkeeping, not placement/residency policy. Native execution
must supply views and its final-consumer completion event before retirement.
It is not yet wired into the validated dense-Linear attachment path.
"""

class TensorOwnerGroup:
    """One identity-unique tensor and every direct module owner of that tensor.

    Loader storage is retained only while preparing publication or recovering
    a partial publication failure. Successful publication drops that reference.
    Retirement invalidates this unloaded component with meta placeholders; it
    never reconstructs CPU weights or pretends the component can run afterward.
    """

    def __init__(self, tensor, kind, owners):
        if kind not in ("parameter", "buffer"):
            raise ValueError("kind must be parameter or buffer")
        self.kind = kind
        self.owners = tuple((module, key, tuple(names)) for module, key, names in owners)
        if not self.owners:
            raise ValueError("At least one tensor owner is required")
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.nbytes = tensor.numel() * tensor.element_size()
        self._original = tensor
        self._published = None
        self._placeholder = None
        self._completion = None
        self.state = "prepared"

    def _table(self, module):
        return module._parameters if self.kind == "parameter" else module._buffers

    def snapshot(self):
        return {"schema_version": 1, "kind": self.kind, "names": [name for _, _, names in self.owners for name in names],
                "owner_count": len(self.owners), "shape": list(self.shape),
                "dtype": str(self.dtype), "bytes": self.nbytes, "state": self.state}

    @property
    def loader_tensor(self):
        if self.state != "prepared":
            raise RuntimeError("Loader tensor is available only before publication")
        return self._original

    def publish(self, view):
        """Publish an externally owned view without copies or breaking aliases."""
        import torch

        if self.state != "prepared":
            raise RuntimeError("Tensor view has already been published or needs rollback")
        self._validate_publication_view(view)
        for module, key, _ in self.owners:
            if self._table(module).get(key) is not self._original:
                raise RuntimeError("A tensor owner changed before publication")
        self._published = torch.nn.Parameter(view, requires_grad=False) if self.kind == "parameter" else view
        self.state = "publication_failed"  # Own both sides until all aliases commit.
        try:
            for module, key, _ in self.owners:
                self._table(module)[key] = self._published
        except BaseException:
            # No consumer can launch from a successful publication yet. Keep
            # both references and expose explicit rollback, preserving error.
            raise
        self._original = None
        self.state = "published"
        return self._published

    def _validate_publication_view(self, view):
        import torch

        if not isinstance(view, torch.Tensor) or view.device.type == "meta":
            raise ValueError("Publication requires a materialized tensor view")
        if tuple(view.shape) != self.shape or view.dtype != self.dtype or not view.is_contiguous():
            raise ValueError("Native view shape/dtype/contiguity differs from the captured tensor")

    def _make_retirement_placeholder(self):
        import torch

        placeholder = torch.empty(self.shape, dtype=self.dtype, device="meta")
        return (torch.nn.Parameter(placeholder, requires_grad=False)
                if self.kind == "parameter" else placeholder)

    def rollback_failed_publication(self):
        """Retry-safe rollback, only before publication has ever succeeded."""
        if self.state != "publication_failed":
            raise RuntimeError("Only failed publication can be rolled back without a completion event")
        for module, key, _ in self.owners:
            value = self._table(module).get(key)
            if value is not self._original and value is not self._published:
                raise RuntimeError("A tensor owner changed during publication recovery")
        for module, key, _ in self.owners:
            self._table(module)[key] = self._original
        self._published = None
        self.state = "prepared"

    def retire(self, completion):
        """Invalidate aliases only after the final native consumer is complete.

        False means pending, not permission to close native backing. Query or
        assignment failure retains views/event ownership and allows retry with
        the same event. Callers still own the native arena/backing close gate.
        """
        import torch

        if self.state == "retired":
            return True
        if self.state not in ("published", "retiring"):
            raise RuntimeError("Only published tensor views can be retired")
        if completion is None or not callable(getattr(completion, "query", None)):
            raise ValueError("Final-consumer completion event is required")
        if self._completion is not None and self._completion is not completion:
            raise RuntimeError("Cannot replace a retained completion event")
        self._completion = completion
        if completion.query() is not True:
            return False
        for module, key, _ in self.owners:
            value = self._table(module).get(key)
            if value is not self._published and (self._placeholder is None or value is not self._placeholder):
                raise RuntimeError("A published tensor owner changed before retirement")
        if self._placeholder is None:
            self._placeholder = self._make_retirement_placeholder()
        self.state = "retiring"
        for module, key, _ in self.owners:
            self._table(module)[key] = self._placeholder
        self._published = None
        self._completion = None
        self.state = "retired"
        return True


def capture_model_tensor_owners(root):
    """Capture direct parameter/buffer aliases, including shared module paths.

    No storage pointer access, CPU copies, CUDA/native calls or module mutation.
    Different tensor objects sharing storage are not merged. Parametrized and
    tensor/Parameter subclass semantics require a separate adapter, not guessing.
    """
    import torch

    if not isinstance(root, torch.nn.Module):
        raise TypeError("root must be a torch.nn.Module")
    groups = {}
    modules = tuple(root.named_modules(remove_duplicate=False))
    for _, module in modules:
        if torch.nn.utils.parametrize.is_parametrized(module):
            raise TypeError("Parametrized modules require a dedicated ownership adapter")
    # Deterministic parameter groups first, followed by buffer groups.
    for kind in ("parameter", "buffer"):
        for path, module in modules:
            table = module._parameters if kind == "parameter" else module._buffers
            for key, tensor in table.items():
                if tensor is None:
                    continue
                expected_type = torch.nn.Parameter if kind == "parameter" else torch.Tensor
                if type(tensor) is not expected_type:
                    raise TypeError("Tensor subclasses require a dedicated ownership adapter")
                group = groups.setdefault((kind, id(tensor)), {"tensor": tensor, "owners": {}})
                owner = group["owners"].setdefault((id(module), key), [module, key, []])
                owner[2].append(f"{path}.{key}" if path else key)
    return tuple(TensorOwnerGroup(group["tensor"], kind, group["owners"].values())
                 for (kind, _), group in groups.items())
