# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Domain containers for decomposed relaxation-rate evaluations.

This module defines reusable domain-level containers for relaxation results
that are computed under a specific set of evaluation conditions. The
containers store independent physical channels, such as dipolar, contact, and
Curie contributions, while exposing the total rate as a derived quantity.
The top-level evaluation container may also store the relaxation-model label
used to generate the decomposition.
"""


class RelaxationChannels:
    """Per-channel relaxation rates keyed by nucleus label.

    This container stores independent relaxation channels for one relaxation
    observable (for example ``R1`` or ``R2``). The canonical source of truth is
    the set of physical contributions. The total rate is always derived from the
    available contributions and is therefore exposed as a computed property.

    Args:
        dipolar: Optional per-nucleus dipolar contribution mapping.
        contact: Optional per-nucleus contact contribution mapping.
        curie: Optional per-nucleus Curie contribution mapping.
    """

    def __init__(
        self,
        dipolar: dict[str, float] | None = None,
        contact: dict[str, float] | None = None,
        curie: dict[str, float] | None = None,
    ) -> None:
        self.dipolar = self._coerce_rate_map(dipolar)
        self.contact = self._coerce_rate_map(contact)
        self.curie = self._coerce_rate_map(curie)

    @staticmethod
    def _coerce_rate_map(
        value: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """Normalise a relaxation-rate mapping to ``dict[str, float]``.

        Args:
            value: Mapping keyed by nucleus label.

        Returns:
            A copied ``dict[str, float]`` or ``None``.

        Raises:
            TypeError: If `value` is not a mapping or contains invalid key/value
                types.
        """
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Relaxation rate map must be a dict[str, float] or None")

        out: dict[str, float] = {}
        for key, rate in value.items():
            if not isinstance(key, str):
                raise TypeError("Relaxation rate-map keys must be strings")
            out[str(key)] = float(rate)
        return out

    @property
    def has_any_component(self) -> bool:
        """Return ``True`` when at least one physical channel is present."""
        return any(
            component is not None
            for component in (self.dipolar, self.contact, self.curie)
        )

    @property
    def total(self) -> dict[str, float] | None:
        """Return the total relaxation rate derived from available channels.

        Returns:
            Per-nucleus total relaxation rates, or ``None`` if no components are
            present.
        """
        components = [
            component
            for component in (self.dipolar, self.contact, self.curie)
            if component is not None
        ]
        if not components:
            return None

        keys = set().union(*(component.keys() for component in components))
        return {
            key: sum(component.get(key, 0.0) for component in components)
            for key in sorted(keys)
        }


class RelaxationEvaluation:
    """Relaxation-rate decomposition for one model evaluation.

    Args:
        relaxation_model: Optional relaxation-model label associated with this
            evaluation.
        r1: Optional decomposition for ``R1``.
        r2: Optional decomposition for ``R2``.
        tau_R: Optional rotational correlation time used (s).
        tau_e1: Optional electronic T1e correlation time used (s).
        tau_e2: Optional electronic T2e correlation time used (s).
    """

    def __init__(
        self,
        relaxation_model: str | None = None,
        r1: RelaxationChannels | None = None,
        r2: RelaxationChannels | None = None,
        tau_R: float | None = None,
        tau_e1: float | None = None,
        tau_e2: float | None = None,
    ) -> None:
        self.relaxation_model = relaxation_model
        self.r1 = r1 if r1 is not None else RelaxationChannels()
        self.r2 = r2 if r2 is not None else RelaxationChannels()
        self.tau_R = tau_R
        self.tau_e1 = tau_e1
        self.tau_e2 = tau_e2
