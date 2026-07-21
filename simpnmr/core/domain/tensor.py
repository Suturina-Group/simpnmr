"""Define tensor-based domain containers for magnetic properties.

Provides Hyperfine, Susceptibility, and Shift classes used across the library.
"""

import copy

import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike, NDArray

from simpnmr.core.const.physics import GE


class Hyperfine:
    """Hyperfine coupling container for a single nucleus.

    Args:
        fc: Isotropic Fermi-contact hyperfine tensor (``ppm Å^-3``).
        sd: Traceless spin-dipolar contribution to the hyperfine tensor
            (``ppm Å^-3``).
        orb: Traceless orbital contribution to the hyperfine tensor
            (``ppm Å^-3``).
        tensor_full: Physical full hyperfine tensor (3x3) if available.

    Attributes:
        fc: Isotropic Fermi-contact hyperfine tensor (``ppm Å^-3``).
        sd: Traceless spin-dipolar contribution to the hyperfine tensor
            (``ppm Å^-3``).
        orb: Traceless orbital contribution to the hyperfine tensor
            (``ppm Å^-3``).
        tensor_full: Physical full hyperfine tensor (3x3) if available.
        eigvals: Eigenvalues of ``tensor_full``.
        eigvecs: Eigenvectors corresponding to ``eigvals``.
    """

    def __init__(
        self,
        *,
        fc: NDArray | None = None,
        sd: NDArray | None = None,
        orb: NDArray | None = None,
        tensor_full: NDArray | None = None,
    ) -> None:
        self._eigvals = None
        self._eigvecs = None
        if fc is None:
            self.fc = np.zeros((3, 3), dtype=float)
        else:
            arr = np.asarray(fc, dtype=float)
            if arr.shape != (3, 3):
                raise ValueError("fc must be a (3x3) matrix")
            self.fc = arr

        if sd is None:
            self.sd = np.zeros((3, 3), dtype=float)
        else:
            arr = np.asarray(sd, dtype=float)
            if arr.shape != (3, 3):
                raise ValueError("sd must be a (3x3) matrix")
            self.sd = arr

        if orb is None:
            self.orb = np.zeros((3, 3), dtype=float)
        else:
            arr = np.asarray(orb, dtype=float)
            if arr.shape != (3, 3):
                raise ValueError("orb must be a (3x3) matrix")
            self.orb = arr

        if tensor_full is None:
            self.tensor_full = None
        else:
            self.tensor_full = tensor_full

        # Conformer-averaged ⟨r⁻⁶⟩ (Å⁻⁶). Set only when HFC data originate
        # from a conformer-averaging workflow; None otherwise.
        self.r_inv6: float | None = None

    @property
    def fc(self) -> NDArray:
        """Fermi-contact hyperfine tensor (``ppm Å^-3``)."""
        return self._fc

    @fc.setter
    def fc(self, tensor: NDArray):
        arr = np.asarray(tensor, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("fc must be a (3x3) matrix")
        self._fc = arr
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def sd(self) -> NDArray:
        """Traceless spin-dipolar contribution to the hyperfine tensor."""
        return self._sd

    @sd.setter
    def sd(self, tensor: NDArray):
        arr = np.asarray(tensor, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("sd must be a (3x3) matrix")
        self._sd = arr
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def orb(self) -> NDArray:
        """Traceless orbital contribution to the hyperfine tensor."""
        return self._orb

    @orb.setter
    def orb(self, tensor: NDArray):
        arr = np.asarray(tensor, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("orb must be a (3x3) matrix")
        self._orb = arr
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def tensor_full(self) -> NDArray | None:
        """Physical full hyperfine tensor (3x3) if available.

        This tensor should represent the sum of physical contributions (e.g.,
        FC + SD (+ ORB if applicable)). It is used for diagnostics (e.g.,
        eigendecomposition) and must not be reconstructed from effective
        quantities.
        """
        return self._tensor_full

    @tensor_full.setter
    def tensor_full(self, tensor: NDArray | None):
        if tensor is None:
            self._tensor_full = None
            self._eigvals = None
            self._eigvecs = None
            return

        arr = np.asarray(tensor, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("tensor_full must be a (3x3) matrix")
        self._tensor_full = arr
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def eigvals(self) -> NDArray:
        """Eigenvalues of the physical full hyperfine tensor."""
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError("Values must be 3 element arraylike")
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        """Eigenvectors of the physical full hyperfine tensor (dimensionless)."""
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        self._eigvecs = intensor
        return

    def calc_eig(self):
        """Computes and stores eigenvalues/eigenvectors of ``tensor_full``.

        Returns:
            A tuple ``(vals, vecs)`` as returned by ``numpy.linalg.eigh``.
        """
        if self.tensor_full is None:
            raise RuntimeError(
                "Hyperfine eigendecomposition requires tensor_full, "
                "but it is not available."
            )
        vals, vecs = la.eigh(self.tensor_full)
        self._eigvals = vals[np.argsort(np.abs(vals))]
        self._eigvecs = vecs[:, np.argsort(np.abs(vals))]
        return vals, vecs

    @staticmethod
    def calc_pdip(r_nuc: ArrayLike, r_elec: ArrayLike = np.zeros(3)):
        """Computes the point-dipole approximation to the dipolar hyperfine tensor.

        Args:
            r_nuc: Nucleus coordinates.
            r_elec: Electron coordinates.

        Returns:
            Dipolar hyperfine tensor as a ``(3, 3)`` array.
        """
        r_nuc = np.asarray(r_nuc)
        r_elec = np.asarray(r_elec)

        r = r_nuc - r_elec
        rnorm = la.norm(r)

        pdip = 3 * np.outer(r, r) / rnorm**5 - np.eye(3) / rnorm**3
        pdip /= 4 * np.pi

        return pdip


class Susceptibility:
    """Magnetic susceptibility tensor.

    Args:
        tensor: Susceptibility tensor as a ``(3, 3)`` NumPy array (Å³).
        temperature: Temperature that this tensor corresponds to (K).

    Attributes:
        tensor: Susceptibility tensor as a ``(3, 3)`` NumPy array (Å³).
        iso: Isotropic susceptibility (Å³).
        dtensor: Deviatoric susceptibility tensor ``tensor - iso * I`` (Å³).
        eigvals: Eigenvalues of `tensor`.
        eigvecs: Eigenvectors corresponding to ``eigvals``.
        alpha: ZYZ Euler alpha angle between the input frame and the eigenframe
            (degrees).
        beta: ZYZ Euler beta angle between the input frame and the eigenframe
            (degrees).
        gamma: ZYZ Euler gamma angle between the input frame and the eigenframe
            (degrees).
        axiality: Axiality of the tensor (Å³).
        rhombicity: Rhombicity of the tensor (Å³).
        irred: Irreducible spherical components (length-5 complex array) ordered
            ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        temperature: Temperature that this tensor corresponds to (K).
    """

    def __init__(
        self, tensor: NDArray = np.zeros([3, 3]), temperature: float = 0.0
    ) -> None:
        self._dtensor = None
        self._iso = None
        self._iso_spin_only = None
        self._iso_g_corr = None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None

        self.temperature = temperature
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        """Susceptibility tensor as a ``(3, 3)`` array (Å³)."""
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Chi must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Chi must be np.array (3x3) of floats")
        self._tensor = intensor

        # and delta susceptibility
        self.calc_dtensor()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        self._axiality = None
        self._rhombicity = None
        self._irred = None
        self._alpha = None
        self._beta = None
        self._gamma = None
        return

    @property
    def iso(self) -> float:
        """Isotropic susceptibility (Å³)."""
        if self._iso is None:
            self.calc_iso()
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    @property
    def iso_spin_only(self) -> float | None:
        """Spin-only isotropic susceptibility (Å³), if available."""
        return self._iso_spin_only

    @iso_spin_only.setter
    def iso_spin_only(self, val: float | None):
        if val is not None and not isinstance(val, (float, np.floating)):
            raise TypeError(
                "Spin-only isotropic susceptibility must be a float or None"
            )
        self._iso_spin_only = None if val is None else float(val)
        return

    @property
    def iso_g_corr(self) -> float | None:
        """g-corrected isotropic susceptibility (Å³), if available."""
        return self._iso_g_corr

    @iso_g_corr.setter
    def iso_g_corr(self, val: float | None):
        if val is not None and not isinstance(val, (float, np.floating)):
            raise TypeError(
                "g-corrected isotropic susceptibility must be a float or None"
            )
        self._iso_g_corr = None if val is None else float(val)
        return

    def calc_iso(self):
        """Computes and stores the isotropic component from `self.tensor`."""
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        """Computes the isotropic component of a susceptibility tensor.

        Args:
            tensor: Susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            The isotropic value (trace/3).
        """
        return np.trace(tensor) / 3.0

    @property
    def dtensor(self) -> NDArray:
        """Deviatoric susceptibility tensor as a ``(3, 3)`` array (Å³)."""
        if self._dtensor is None:
            self.calc_dtensor()
        return self._dtensor

    @dtensor.setter
    def dtensor(self, tensor: NDArray):
        self._dtensor = tensor
        return

    def calc_dtensor(self):
        """Computes and stores the delta susceptibility tensor from `self.tensor`."""
        self.dtensor = self._calc_dtensor(self.tensor)
        return

    @staticmethod
    def _calc_dtensor(tensor: NDArray) -> NDArray:
        """Computes the delta susceptibility tensor.

        Args:
            tensor: Susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            The delta susceptibility tensor ``tensor - I * trace(tensor)/3``.
        """
        return tensor - (np.eye(3) * Susceptibility._calc_iso(tensor))

    @property
    def eigvals(self) -> NDArray:
        """Eigenvalues of the susceptibility tensor."""
        # Recalculate if not populated
        if self._eigvals is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvals

    @eigvals.setter
    def eigvals(self, value: ArrayLike):
        value = np.asarray(value)
        if np.size(value) != 3 or np.shape(value) != (3,):
            raise TypeError("Values must be 3 element arraylike")
        self._eigvals = value
        return

    @property
    def eigvecs(self) -> NDArray:
        """Eigenvectors of the susceptibility tensor (dimensionless)."""
        # Recalculate if not populated
        if self._eigvecs is None:
            self.eigvals, self.eigvecs = self.calc_eig()
        return self._eigvecs

    @eigvecs.setter
    def eigvecs(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("Vectors must be np.array (3x3) of floats")
        self._eigvecs = intensor
        return

    def calc_eig(self):
        """Computes and stores eigenvalues/eigenvectors of `self.tensor`.

        Returns:
            A tuple ``(vals, vecs)`` as returned by ``numpy.linalg.eigh``.
        """
        vals, vecs = la.eigh(self.tensor)

        self._eigvals = vals[np.argsort(np.abs(vals))]
        self._eigvecs = vecs[:, np.argsort(np.abs(vals))]

        return vals, vecs

    @property
    def axiality(self) -> float:
        if self._axiality is None:
            self.calc_axiality()
        return self._axiality

    @axiality.setter
    def axiality(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError("Axiality must be a float")
        else:
            self._axiality = value
        return

    def calc_axiality(self):
        devals = la.eigvalsh(self.dtensor)
        self.axiality = 1.5 * devals[np.argmax(np.abs(devals))]
        return

    @property
    def rhombicity(self) -> float:
        if self._rhombicity is None:
            self.calc_rhombicity()
        return self._rhombicity

    @rhombicity.setter
    def rhombicity(self, value: float):
        if not isinstance(value, np.floating):
            raise ValueError("Rhombicity must be a float")
        else:
            self._rhombicity = value
        return

    def calc_rhombicity(self):
        devals = la.eigvalsh(self.dtensor)
        order = np.argsort(np.abs(devals))
        devals = devals[order]
        self.rhombicity = 0.5 * (devals[0] - devals[1])
        return

    @property
    def alpha(self) -> float:
        """ZYZ Euler alpha angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._alpha is None:
            self.calc_euler()
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Alpha must be a float")
        else:
            self._alpha = value
        return

    @property
    def beta(self) -> float:
        """ZYZ Euler beta angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._beta is None:
            self.calc_euler()
        return self._beta

    @beta.setter
    def beta(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Beta must be a float")
        else:
            self._beta = value
        return

    @property
    def gamma(self) -> float:
        """ZYZ Euler gamma angle between the input frame and the eigenframe.

        The value is stored in degrees.
        """
        # Calculate if unpopulated
        if self._gamma is None:
            self.calc_euler()
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not isinstance(value, np.floating):
            raise ValueError("Gamma must be a float")
        else:
            self._gamma = value
        return

    def calc_euler(self):
        """Computes ZYZ Euler angles mapping input frame to eigenframe.

        Eigenvectors are sorted by deviation from iso (|λ − iso|),
        consistent with the axiality/rhombicity convention:
          col 0 → x (least anisotropic)
          col 1 → y (intermediate)
          col 2 → z (axial, largest deviation)

        Sign convention: the largest-magnitude component of each eigenvector
        is forced positive, then the determinant is checked and col 0 is
        flipped if needed to ensure a proper rotation (det = +1).  This makes
        the eigenvectors — and therefore the angles — unique for
        non-degenerate tensors.

        Angle ranges:
          α ∈ [0°, 360°),  β ∈ [0°, 180°],  γ ∈ [0°, 360°)

        The ZYZ decomposition is derived from the axial column (col 2):
          β  = arccos(V[2, 2])
          α  = arctan2(V[1, 2], V[0, 2])  mod 360
          γ  = arctan2(V[2, 1], −V[2, 0]) mod 360

        Angles are stored in degrees.
        """
        _ev = np.abs(self.eigvals - self.iso)
        order = np.argsort(_ev)
        _ev_sorted = _ev[order]
        _vecs = self.eigvecs[:, order].copy()

        # Fix sign of each eigenvector: largest-magnitude component → positive
        for j in range(3):
            idx = np.argmax(np.abs(_vecs[:, j]))
            if _vecs[idx, j] < 0:
                _vecs[:, j] = -_vecs[:, j]

        # Ensure proper rotation matrix (det = +1)
        if np.linalg.det(_vecs) < 0:
            _vecs[:, 0] = -_vecs[:, 0]

        self.alpha = np.float64(
            np.rad2deg(np.arctan2(_vecs[1, 2], _vecs[0, 2])) % 360
        )
        self.beta = np.float64(
            np.rad2deg(np.arccos(np.clip(_vecs[2, 2], -1.0, 1.0)))
        )
        # γ is undefined when the two minor axes are degenerate (axial tensor);
        # fix to 0 in that case.
        if np.isclose(_ev_sorted[0], _ev_sorted[1], rtol=1e-8):
            self.gamma = np.float64(0.0)
        else:
            self.gamma = np.float64(
                np.rad2deg(np.arctan2(_vecs[2, 1], -_vecs[2, 0])) % 360
            )
        return

    @property
    def irred(self) -> NDArray:
        """Irreducible spherical components of the susceptibility tensor.

        Returns a length-5 complex array ordered
        ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        """
        # Calculate if unpopulated
        if self._irred is None:
            self.calc_irred()
        return self._irred

    @irred.setter
    def irred(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        elif value.shape != (5,):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        elif not np.iscomplexobj(value):
            raise TypeError(
                "Irreducible Spherical Components must be 5 element "
                "arraylike of complex numbers"
            )
        self._irred = value
        return

    def calc_irred(self):
        """Computes and stores irreducible spherical components from `self.tensor`."""
        self.irred = self._calc_irred(self.tensor)
        return

    @staticmethod
    def _calc_irred(tensor: NDArray) -> NDArray:
        """Computes irreducible spherical components of a susceptibility tensor.

        Note:
            This does not include any isotropic contribution.

        Args:
            tensor: Real susceptibility tensor as a ``(3, 3)`` array.

        Returns:
            A length-5 complex128 array ordered ``chi_-2, chi_-1, chi_0, chi_1, chi_2``.
        """
        irred = np.zeros(5, dtype=np.complex128)
        # chi_-2
        irred[0] = +np.sqrt(2 * np.pi / 15) * (
            tensor[0, 0] - tensor[1, 1] + 1j * (tensor[0, 1] + tensor[1, 0])
        )
        # chi_-1
        irred[1] = -np.sqrt(2 * np.pi / 15) * (
            tensor[0, 2] - tensor[2, 0] + 1j * (tensor[1, 2] + tensor[2, 1])
        )
        # chi_0
        irred[2] = +np.sqrt(4 * np.pi / 45) * (
            2 * tensor[2, 2] - tensor[0, 0] - tensor[1, 1]
        )
        # chi_+1
        irred[3] = -np.sqrt(2 * np.pi / 15) * (
            tensor[0, 2] - tensor[2, 0] - 1j * (tensor[1, 2] + tensor[2, 1])
        )
        # chi_2
        irred[4] = +np.sqrt(2 * np.pi / 15) * (
            tensor[0, 0] - tensor[1, 1] - 1j * (tensor[0, 1] + tensor[1, 0])
        )

        return irred


class Shift:
    """Chemical shift components for a nucleus.

    This container tracks diamagnetic and hyperfine contributions to the total
    paramagnetic chemical shift.

    Attributes:
        dia: Diamagnetic chemical shift (ppm).
        fc: Fermi contact chemical shift (ppm).
        fc_spin_only: Spin-only Fermi contact chemical shift reference (ppm).
        fc_delta_g_corr: Difference between canonical and spin-only Fermi
            contact shifts (ppm).
        orb_iso: Orbital isotropic shift contribution (ppm).
        orb_aniso: Orbital anisotropic shift contribution (ppm).
        orb: Total orbital shift contribution (ppm), equal to ``orb_iso + orb_aniso``.
        pc: Pseudocontact chemical shift (ppm).
        hf: Hyperfine chemical shift (ppm), equal to ``fc + pc``.
        paramag: Total paramagnetic chemical shift (ppm), equal to
            ``fc + pc + orb``.
        total: Total chemical shift (ppm), equal to ``dia + paramag``.
        avg: Averaged total shift (ppm). Defaults to ``total`` and is reset to
            ``total`` whenever any component is modified.
        lw: Linewidth of the signal, or None when no linewidth model has
            assigned a value.
    """

    def __init__(
        self,
        dia: float = 0.0,
        pc: float = 0.0,
        fc: float = 0.0,
        fc_spin_only: float | None = None,
        fc_delta_g_corr: float | None = None,
        orb_iso: float = 0.0,
        orb_aniso: float = 0.0,
        lw: float | None = None,
    ) -> None:
        self._pc = pc  # Pseudocontact
        self._fc = fc  # Fermi Contact
        self._fc_spin_only = fc_spin_only  # Spin-only Fermi Contact reference
        self._fc_delta_g_corr = fc_delta_g_corr  # Canonical minus spin-only FC
        self._orb_iso = orb_iso  # Orbital isotropic
        self._orb_aniso = orb_aniso  # Orbital anisotropic
        self._dia = dia  # Diamagnetic
        self._lw = lw
        # Full 3x3 shift tensors (raw, non-symmetric); the scalar components
        # above are one third of each tensor's trace.
        self._pc_tensor = np.zeros((3, 3), dtype=float)
        self._fc_tensor = np.zeros((3, 3), dtype=float)
        self._orb_tensor = np.zeros((3, 3), dtype=float)
        self._avg = copy.copy(self.total)
        pass

    @property
    def total(self) -> float:
        return self.dia + self.paramag

    @property
    def hf(self) -> float:
        return self.pc + self.fc

    @property
    def paramag(self) -> float:
        return self.hf + self.orb

    # --- Full 3x3 shift tensors (raw, non-symmetric) -----------------------
    @property
    def pc_tensor(self) -> NDArray:
        return self._pc_tensor

    @pc_tensor.setter
    def pc_tensor(self, val: NDArray):
        self._pc_tensor = np.asarray(val, dtype=float)

    @property
    def fc_tensor(self) -> NDArray:
        return self._fc_tensor

    @fc_tensor.setter
    def fc_tensor(self, val: NDArray):
        self._fc_tensor = np.asarray(val, dtype=float)

    @property
    def orb_tensor(self) -> NDArray:
        return self._orb_tensor

    @orb_tensor.setter
    def orb_tensor(self, val: NDArray):
        self._orb_tensor = np.asarray(val, dtype=float)

    @property
    def paramag_tensor(self) -> NDArray:
        """Total paramagnetic shift tensor (ppm): fc + pc + orb (raw 3x3)."""
        return self._fc_tensor + self._pc_tensor + self._orb_tensor

    @property
    def avg(self) -> float:
        return self._avg

    @avg.setter
    def avg(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._avg = float(val)
        return

    @property
    def pc(self) -> float:
        return self._pc

    @pc.setter
    def pc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._pc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def fc(self) -> float:
        return self._fc

    @fc.setter
    def fc(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._fc = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def fc_spin_only(self) -> float | None:
        """Spin-only Fermi contact chemical shift reference (ppm)."""
        return self._fc_spin_only

    @fc_spin_only.setter
    def fc_spin_only(self, val: float | None):
        if val is not None and not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float or None")
        self._fc_spin_only = None if val is None else float(val)
        return

    @property
    def fc_delta_g_corr(self) -> float | None:
        """Canonical minus spin-only Fermi contact shift difference (ppm)."""
        return self._fc_delta_g_corr

    @fc_delta_g_corr.setter
    def fc_delta_g_corr(self, val: float | None):
        if val is not None and not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float or None")
        self._fc_delta_g_corr = None if val is None else float(val)
        return

    @property
    def orb_iso(self) -> float:
        return self._orb_iso

    @orb_iso.setter
    def orb_iso(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._orb_iso = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def orb_aniso(self) -> float:
        return self._orb_aniso

    @orb_aniso.setter
    def orb_aniso(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Chemical shift must be a float")
        self._orb_aniso = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def orb(self) -> float:
        return self.orb_iso + self.orb_aniso

    @property
    def dia(self) -> float:
        return self._dia

    @dia.setter
    def dia(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Diamagnetic chemical shift must be a float")
        self._dia = float(val)
        self.avg = copy.copy(self.total)
        return

    @property
    def lw(self) -> float | None:
        return self._lw

    @lw.setter
    def lw(self, val: float | None):
        if val is None:
            self._lw = None
            return
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Linewidth must be a float")
        self._lw = float(val)
        return

    @staticmethod
    def calc_orb_iso(
        A: Hyperfine, chi: "Susceptibility", g_tensor_dft: NDArray
    ) -> float:
        """
        Compute isotropic part of the orbital shift contribution.

        Args:
            A: Hyperfine coupling container providing spin-dipolar and orbital tensors.
            chi: Magnetic susceptibility tensor.
            g_tensor_dft: Electronic g-tensor derived from DFT (3x3).

        Returns:
            Isotropic part of the orbital shift contribution.
        """
        g_tensor_dft = np.asarray(g_tensor_dft, dtype=float)
        if g_tensor_dft.shape != (3, 3):
            raise ValueError("g_tensor_dft must be a (3, 3) matrix")

        try:
            g_inv_t = la.inv(g_tensor_dft).T
        except la.LinAlgError as exc:
            raise ValueError("g_tensor_dft must be invertible") from exc

        a_orb_eff = GE * (g_inv_t @ (A.sd + A.orb).T)
        shift = chi.iso * (1.0 / 3.0) * np.trace(a_orb_eff)
        return shift

    @staticmethod
    def calc_orb_aniso(
        A: Hyperfine, chi: "Susceptibility", g_tensor_dft: NDArray
    ) -> float:
        """
        Compute anisotropic part of the orbital shift contribution.

        Args:
            A: Hyperfine coupling container providing spin-dipolar and orbital tensors.
            chi: Magnetic susceptibility tensor.
            g_tensor_dft: Electronic g-tensor derived from DFT (3x3).

        Returns:
            Anisotropic part of the orbital shift contribution.
        """
        g_tensor_dft = np.asarray(g_tensor_dft, dtype=float)
        if g_tensor_dft.shape != (3, 3):
            raise ValueError("g_tensor_dft must be a (3, 3) matrix")

        try:
            g_inv_t = la.inv(g_tensor_dft).T
        except la.LinAlgError as exc:
            raise ValueError("g_tensor_dft must be invertible") from exc

        a_orb_eff = GE * (g_inv_t @ (A.sd + A.orb).T)
        shift = (1.0 / 3.0) * np.trace(chi.dtensor @ a_orb_eff - chi.dtensor @ A.sd)
        return shift

    @staticmethod
    def calc_orb(A: Hyperfine, chi: "Susceptibility", g_tensor_dft: NDArray) -> float:
        """
        Compute the orbital shift contribution.

        Args:
            A: Hyperfine coupling container providing spin-dipolar and orbital tensors.
            chi: Magnetic susceptibility tensor.
            g_tensor_dft: Electronic g-tensor derived from DFT (3x3).

        Returns:
            The orbital shift contribution.
        """
        shift = Shift.calc_orb_iso(A, chi, g_tensor_dft) + Shift.calc_orb_aniso(
            A, chi, g_tensor_dft
        )
        return shift

    @staticmethod
    def calc_pcs(
        A: Hyperfine,
        chi: "Susceptibility",
    ) -> float:
        """Compute the pseudocontact shift contribution.

        Args:
            A: Hyperfine coupling container providing the spin-dipolar tensor.
            chi: Magnetic susceptibility tensor.

        Returns:
            The pseudocontact shift (PCS).
        """
        shift = 1.0 / 3.0 * np.trace(chi.dtensor @ A.sd)
        return shift

    @staticmethod
    def calc_fcs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Computes the Fermi contact contribution to the chemical shift."""
        shift = chi.iso * (1.0 / 3.0 * np.trace(A.fc))
        return shift

    # --- Full 3x3 shift tensors (raw, non-symmetric) -----------------------
    # Each is the shift tensor before the isotropic average; taking one third
    # of its trace recovers the corresponding scalar above.
    @staticmethod
    def calc_pcs_tensor(A: Hyperfine, chi: "Susceptibility") -> NDArray:
        """Pseudocontact shift tensor (ppm). ``trace/3`` == :meth:`calc_pcs`."""
        return chi.dtensor @ A.sd

    @staticmethod
    def calc_fcs_tensor(A: Hyperfine, chi: "Susceptibility") -> NDArray:
        """Fermi-contact shift tensor (ppm). ``trace/3`` == :meth:`calc_fcs`."""
        return chi.iso * A.fc

    @staticmethod
    def calc_orb_tensor(
        A: Hyperfine, chi: "Susceptibility", g_tensor_dft: NDArray
    ) -> NDArray:
        """Orbital shift tensor (ppm). ``trace/3`` == :meth:`calc_orb`."""
        g_tensor_dft = np.asarray(g_tensor_dft, dtype=float)
        if g_tensor_dft.shape != (3, 3):
            raise ValueError("g_tensor_dft must be a (3, 3) matrix")
        try:
            g_inv_t = la.inv(g_tensor_dft).T
        except la.LinAlgError as exc:
            raise ValueError("g_tensor_dft must be invertible") from exc
        a_orb_eff = GE * (g_inv_t @ (A.sd + A.orb).T)
        return chi.iso * a_orb_eff + chi.dtensor @ a_orb_eff - chi.dtensor @ A.sd

    @staticmethod
    def calc_paramag(
        A: Hyperfine, chi: "Susceptibility", g_tensor_dft: NDArray
    ) -> float:
        """Compute the total paramagnetic shift (FC + PCS + orbital)."""
        return (
            Shift.calc_fcs(A, chi)
            + Shift.calc_pcs(A, chi)
            + Shift.calc_orb(A, chi, g_tensor_dft)
        )
