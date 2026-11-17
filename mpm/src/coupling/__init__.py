"""MPM side of the MPM <-> XPBD window exchange (P2): contracts (mirror of XPBD's
xpbd-exchange-v1), the MPM endpoint with source registration and stop rules, an MPM-side
prescribed peer with XPBD KinematicSurfaceBridge semantics, the batch archive / replay (P2-U2);
the wire format, the XPBD service process, the process link and the two-process driver (P2-U3).
Not part of the solver core (mpm/); scenes and tests import it with Work_place on sys.path.

Exports are lazy (PEP 562, P2-U3): importing coupling, coupling.wire or coupling.xpbd_service
loads neither Taichi nor mpm, so the spawned XPBD service process stays free of both; the
names are the same as before (endpoint, prescribed and archive load on first use)."""
import importlib

from .contracts import SCHEMA, ImpulseWindow, LoadBatch, SourceIdentity, WindowOffer, substeps_of

_LAZY = dict(MPMEndpoint="endpoint", EndpointStop="endpoint", StopRules="endpoint", STOP_DEFAULTS="endpoint",
             code_hashes="endpoint", PrescribedPeer="prescribed", receipt_digest="prescribed",
             XPBDLink="link", PeerError="link", CoupledDriver="driver")
_SUBMODULES = ("archive", "endpoint", "prescribed", "wire", "xpbd_service", "link", "driver", "contracts")
__all__ = ["SCHEMA", "ImpulseWindow", "LoadBatch", "SourceIdentity", "WindowOffer", "substeps_of", *_LAZY,
           *_SUBMODULES]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
