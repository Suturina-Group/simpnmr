"""Define tensor-based domain containers for magnetic properties.

Provides Hyperfine, Susceptibility, and Shift classes used across the library.
"""

import copy

import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike, NDArray


class Hyperfine:
    """Hyperfine coupling tensor for a single nucleus.

    Args:
        tensor: Hyperfine tensor as a ``(3, 3)`` NumPy array. Units are
            ``ppm Å^-3``.

    Attributes:
        tensor: Hyperfine tensor as a ``(3, 3)`` NumPy array (``ppm Å^-3``).
        iso: Isotropic hyperfine coupling (``ppm Å^-3``).
        dip: Dipolar hyperfine tensor (``ppm Å^-3``).
        eigvals: Eigenvalues of the total hyperfine tensor.
        eigvecs: Eigenvectors corresponding to ``eigvals``.
    """

    def __init__(self, tensor: NDArray = np.zeros([3, 3])) -> None:
        self._iso = None
        self._dip = None
        self._eigvals = None
        self._eigvecs = None
        self.tensor = copy.deepcopy(tensor)
        pass

    @property
    def tensor(self) -> NDArray:
        """Hyperfine coupling tensor as a ``(3, 3)`` array (``ppm Å^-3``)."""
        return self._tensor

    @tensor.setter
    def tensor(self, intensor: NDArray):
        if not isinstance(intensor, np.ndarray):
            raise TypeError("A must be np.array (3x3) of floats")
        elif intensor.shape != (3, 3):
            raise TypeError("A must be np.array (3x3) of floats")
        self._tensor = intensor

        # Recalculate isotropic hyperfine
        self.calc_iso()
        # and dipolar hyperfine
        self.calc_dip()

        # and reset eigenvalues and eigenvectors to None
        self._eigvals = None
        self._eigvecs = None
        return

    @property
    def iso(self) -> float:
        """Isotropic hyperfine coupling constant (``ppm Å^-3``)."""
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
        return

    def calc_iso(self):
        """Computes and stores the isotropic component from `self.tensor`."""
        self.iso = self._calc_iso(self.tensor)
        return

    @staticmethod
    def _calc_iso(tensor: NDArray) -> float:
        """Computes the isotropic component of a hyperfine tensor.

        Args:
            tensor: Hyperfine tensor as a ``(3, 3)`` array.

        Returns:
            The isotropic value (trace/3).
        """
        return np.trace(tensor) / 3.0

    @property
    def dip(self) -> NDArray:
        """Dipolar hyperfine tensor as a ``(3, 3)`` array."""
        return self._dip

    @dip.setter
    def dip(self, tensor: NDArray):
        self._dip = tensor
        return

    def calc_dip(self):
        """Computes and stores the dipolar component from `self.tensor`."""
        self.dip = self._calc_dip(self.tensor)
        return

    @staticmethod
    def _calc_dip(tensor: NDArray) -> NDArray:
        """Computes the dipolar part of a hyperfine tensor.

        Args:
            tensor: Hyperfine tensor as a ``(3, 3)`` array.

        Returns:
            The dipolar tensor ``tensor - I * trace(tensor)/3``.
        """
        return tensor - np.eye(3) * Hyperfine._calc_iso(tensor)

    @property
    def eigvals(self) -> NDArray:
        """Eigenvalues of the hyperfine tensor."""
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
        """Eigenvectors of the hyperfine tensor (dimensionless)."""
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

        # Recalculate isotropic susceptibility
        self.calc_iso()
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
        return self._iso

    @iso.setter
    def iso(self, val: float):
        self._iso = val
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
        """Computes and stores ZYZ Euler angles mapping input frame to eigenframe.

        Angles are stored in degrees.
        """
        _ev = np.abs(self.eigvals - self.iso)
        order = np.argsort(_ev)
        _vecs = self.eigvecs[:, order]

        self.alpha = np.rad2deg(np.arctan2(_vecs[2, 1], -_vecs[0, 1]))
        self.beta = np.rad2deg(np.arccos(_vecs[1, 1]))
        self.gamma = np.rad2deg(np.arctan2(-_vecs[1, 2], _vecs[1, 0]))
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
        pc: Pseudocontact chemical shift (ppm).
        hf: Hyperfine chemical shift (ppm), equal to ``fc + pc``.
        total: Total chemical shift (ppm), equal to ``dia + hf``.
        avg: Averaged total shift (ppm). Defaults to ``total`` and is reset to
            ``total`` whenever any component is modified.
        lw: Linewidth of the signal.
    """

    def __init__(
        self, dia: float = 0.0, pc: float = 0.0, fc: float = 0.0, lw: float = 1.0
    ) -> None:
        self._pc = pc  # Pseudocontact
        self._fc = fc  # Fermi Contact
        self._dia = dia  # Diamagnetic
        self._lw = lw
        self._avg = copy.copy(self.total)
        pass

    @property
    def total(self) -> float:
        return self.dia + self.hf

    @property
    def hf(self) -> float:
        return self.pc + self.fc

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
    def lw(self) -> float:
        return self._lw

    @lw.setter
    def lw(self, val: float):
        if not isinstance(val, (float, np.floating)):
            raise TypeError("Linewidth must be a float")
        self._lw = float(val)
        return

    @staticmethod
    def calc_pcs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Compute the pseudocontact shift contribution.

        Args:
            A: Hyperfine coupling tensor.
            chi: Magnetic susceptibility tensor.

        Returns:
            The pseudocontact shift (PCS).
        """
        shift = 1.0 / 3.0 * np.trace(chi.dtensor @ A.dip)
        return shift

    @staticmethod
    def calc_fcs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Computes the Fermi contact contribution to the chemical shift."""
        shift = chi.iso * A.iso
        return shift

    @staticmethod
    def calc_hfs(A: Hyperfine, chi: "Susceptibility") -> float:
        """Computes the total hyperfine shift (Fermi contact + PCS)."""
        return Shift.calc_fcs(A, chi) + Shift.calc_pcs(A, chi)
