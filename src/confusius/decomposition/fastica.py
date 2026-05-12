"""FastICA decomposition for `(time, ...)` fUSI DataArrays."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA as _SklearnFastICA
from sklearn.utils.validation import check_is_fitted

from confusius.validation import validate_time_series


class FastICA(BaseEstimator, TransformerMixin):
    """Fast independent component analysis for fUSI data.

    This estimator wraps [`sklearn.decomposition.FastICA`][sklearn.decomposition.FastICA]
    while preserving xarray metadata through `transform` and `inverse_transform`.

    Parameters
    ----------
    n_components : int, default: None
        Number of independent components to estimate. If `None`, all components are
        kept, subject to the underlying scikit-learn behavior.
    algorithm : {"parallel", "deflation"}, default: "parallel"
        FastICA iteration algorithm.
    whiten : {"unit-variance", "arbitrary-variance"} or bool, default: "unit-variance"
        Whitening strategy passed to scikit-learn.
    fun : {"logcosh", "exp", "cube"} or callable, default: "logcosh"
        Functional form of the G function used in the approximation to neg-entropy.
    fun_args : dict[str, Any] or None, default: None
        Arguments to send to the functional form specified by `fun`.
    max_iter : int, default: 200
        Maximum number of iterations during fit.
    tol : float, default: 1e-4
        Tolerance at which the un-mixing matrix is considered to have converged.
    w_init : (n_components, n_components) numpy.ndarray or None, default: None
        Initial un-mixing array.
    whiten_solver : {"svd", "eigh"}, default: "svd"
        Whitening solver to use.
    random_state : int or None, default: None
        Random state for reproducible initialization.

    Attributes
    ----------
    components_ : (n_components, ...) xarray.DataArray
        Estimated un-mixing matrix reshaped to the original spatial geometry.
    mixing_ : (..., n_components) xarray.DataArray
        Estimated mixing matrix reshaped to the fitted spatial geometry.
    mean_ : (...) xarray.DataArray
        Per-feature empirical mean from the training set. Defined only when the
        fitted scikit-learn estimator exposes `mean_`.
    whitening_ : (n_components, ...) xarray.DataArray
        Whitening matrix reshaped to the original spatial geometry. Defined only when
        the fitted scikit-learn estimator exposes `whitening_`.
    n_components_ : int
        Number of components estimated during fit.
    n_iter_ : int
        Number of iterations run by the FastICA algorithm.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : (n_features_in_,) numpy.ndarray
        Feature names seen during `fit`. Defined only when flattened feature labels are
        all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.decomposition import FastICA
    >>>
    >>> rng = np.random.default_rng(0)
    >>> data = xr.DataArray(
    ...     rng.standard_normal((200, 10, 20)),
    ...     dims=["time", "y", "x"],
    ... )
    >>>
    >>> ica = FastICA(n_components=5, random_state=0)
    >>> signals = ica.fit_transform(data)
    >>> signals.dims
    ('time', 'component')
    >>> reconstructed = ica.inverse_transform(signals)
    >>> reconstructed.dims
    ('time', 'y', 'x')
    """

    def __init__(
        self,
        *,
        n_components: int | None = None,
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: Literal["arbitrary-variance", "unit-variance"] | bool = "unit-variance",
        fun: Literal["logcosh", "exp", "cube"] | Callable[..., Any] = "logcosh",
        fun_args: dict[str, Any] | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        w_init: npt.NDArray[np.floating] | None = None,
        whiten_solver: Literal["svd", "eigh"] = "svd",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def fit(self, X: xr.DataArray, y: None = None) -> "FastICA":
        """Fit FastICA on `(time, ...)` fUSI data.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        FastICA
            Fitted estimator.
        """
        del y

        X_proc, X_stacked, spatial_dims = self._prepare_data(
            X,
            check_layout=False,
            operation_name="FastICA.fit",
        )

        fastica = _SklearnFastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            max_iter=self.max_iter,
            tol=self.tol,
            w_init=self.w_init,
            whiten_solver=self.whiten_solver,
            random_state=self.random_state,
        )
        fastica.fit(X_proc)

        self.spatial_dims_ = spatial_dims
        self._spatial_sizes_ = {dim: int(X.sizes[dim]) for dim in self.spatial_dims_}
        self._feature_coord_ = X_stacked.coords["feature"]
        self._fit_attrs_ = dict(X.attrs)
        self._fit_name_ = X.name

        component_coord = np.arange(fastica.components_.shape[0], dtype=np.intp)
        self.components_ = self._reshape_component_matrix(
            fastica.components_,
            component_coord,
            long_name="Independent components",
        )
        self.mixing_ = self._reshape_feature_component_matrix(
            fastica.mixing_,
            component_coord,
        )

        if hasattr(fastica, "mean_"):
            mean_stacked = xr.DataArray(
                fastica.mean_,
                dims=["feature"],
                coords={"feature": self._feature_coord_},
            )
            self.mean_ = mean_stacked.unstack("feature")

        if hasattr(fastica, "whitening_"):
            self.whitening_ = self._reshape_component_matrix(
                fastica.whitening_,
                component_coord,
                long_name="Whitening matrix",
            )

        self.n_components_ = int(fastica.components_.shape[0])
        self.n_iter_ = int(fastica.n_iter_)
        self.n_features_in_ = int(X_proc.shape[1])

        if len(self.spatial_dims_) == 1 and self.spatial_dims_[0] in X.coords:
            feature_labels = np.asarray(X.coords[self.spatial_dims_[0]].values)
            if all(isinstance(value, (str, np.str_)) for value in feature_labels):
                self.feature_names_in_ = feature_labels.astype(str, copy=False)

        self._fastica = fastica

        return self

    def fit_transform(
        self, X: xr.DataArray, y: None = None, **fit_params: object
    ) -> xr.DataArray:
        """Fit FastICA on `X` and return transformed source signals.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data.
        y : None, optional
            Ignored. Present for scikit-learn API compatibility.
        **fit_params : object
            Additional fit parameters. Unsupported for this estimator.

        Returns
        -------
        (time, component) xarray.DataArray
            FastICA source signals in component space.
        """
        del y

        if fit_params:
            keys = ", ".join(sorted(fit_params))
            raise TypeError(
                f"Unexpected fit parameters for FastICA.fit_transform: {keys}."
            )

        return self.fit(X).transform(X)

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """Project data into FastICA component space.

        Parameters
        ----------
        X : (time, ...) xarray.DataArray
            Input fUSI data with the same spatial dimensions and sizes as the data used
            during fit.

        Returns
        -------
        (time, component) xarray.DataArray
            FastICA source signals in component space.
        """
        check_is_fitted(self)
        X_proc, _, _ = self._prepare_data(
            X,
            check_layout=True,
            operation_name="FastICA.transform",
        )
        signals = self._fastica.transform(X_proc)

        if "time" in X.coords:
            time_coord: npt.NDArray[np.generic] | xr.DataArray = X.coords["time"]
        else:
            time_coord = np.arange(X.sizes["time"], dtype=np.intp)

        transformed = xr.DataArray(
            signals,
            dims=["time", "component"],
            coords={
                "time": time_coord,
                "component": self.components_.coords["component"],
            },
        )
        transformed.attrs.update({"long_name": "FastICA signals"})
        return transformed

    def inverse_transform(
        self, X: xr.DataArray | npt.NDArray[np.floating]
    ) -> xr.DataArray:
        """Reconstruct data from FastICA component signals.

        Parameters
        ----------
        X : (time, component) xarray.DataArray or (time, component) numpy.ndarray
            Signals in FastICA component space.

        Returns
        -------
        (time, ...) xarray.DataArray
            Reconstructed data in the fitted spatial geometry.
        """
        check_is_fitted(self)

        if isinstance(X, xr.DataArray):
            if set(X.dims) != {"time", "component"}:
                raise ValueError(
                    "X must have exactly the dimensions {'time', 'component'} for "
                    "inverse_transform."
                )
            X_ordered = X.transpose("time", "component")
            signals = np.asarray(X_ordered.values, dtype=np.float64)
            time_coord = (
                X_ordered.coords["time"]
                if "time" in X_ordered.coords
                else np.arange(X_ordered.sizes["time"], dtype=np.intp)
            )
        elif isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    f"X must be 2D with shape (time, component), got {X.shape}."
                )
            signals = np.asarray(X, dtype=np.float64)
            time_coord = np.arange(signals.shape[0], dtype=np.intp)
        else:
            raise TypeError(f"X must be DataArray or ndarray, got {type(X)}")

        if signals.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {signals.shape[1]} components, but FastICA was fitted with "
                f"{self.n_components_}."
            )

        reconstructed = self._fastica.inverse_transform(signals)
        reconstructed_stacked = xr.DataArray(
            reconstructed,
            dims=["time", "feature"],
            coords={"time": time_coord, "feature": self._feature_coord_},
        )
        reconstructed_da = reconstructed_stacked.unstack("feature")
        reconstructed_da.attrs.update(self._fit_attrs_)
        reconstructed_da.name = self._fit_name_
        return reconstructed_da

    def _prepare_data(
        self,
        X: xr.DataArray,
        check_layout: bool,
        operation_name: str,
    ) -> tuple[npt.NDArray[np.float64], xr.DataArray, tuple[str, ...]]:
        """Validate and stack `(time, ...)` data into `(time, feature)` matrix."""
        validate_time_series(X, operation_name=operation_name)

        input_spatial_dims = tuple(str(dim) for dim in X.dims if dim != "time")
        if len(input_spatial_dims) == 0:
            raise ValueError(
                "X must have at least one spatial dimension besides 'time'."
            )

        if check_layout:
            spatial_dims = self.spatial_dims_
            if set(input_spatial_dims) != set(spatial_dims):
                raise ValueError(
                    "X spatial dimensions do not match fitted dimensions. "
                    f"Expected {spatial_dims}, got {input_spatial_dims}."
                )
            for dim in spatial_dims:
                if X.sizes[dim] != self._spatial_sizes_[dim]:
                    raise ValueError(
                        f"Spatial dimension '{dim}' has size {X.sizes[dim]} in X, "
                        f"but expected {self._spatial_sizes_[dim]} from fit."
                    )
        else:
            spatial_dims = input_spatial_dims

        X_ordered = X.transpose("time", *spatial_dims)
        X_stacked = X_ordered.stack(feature=list(spatial_dims))
        X_proc = np.asarray(X_stacked.values, dtype=np.float64)
        return X_proc, X_stacked, spatial_dims

    def _reshape_component_matrix(
        self,
        matrix: npt.NDArray[np.floating],
        component_coord: npt.NDArray[np.intp],
        *,
        long_name: str,
    ) -> xr.DataArray:
        """Reshape a `(component, feature)` matrix back to spatial geometry."""
        matrix_stacked = xr.DataArray(
            matrix,
            dims=["component", "feature"],
            coords={"component": component_coord, "feature": self._feature_coord_},
        )
        reshaped = matrix_stacked.unstack("feature")
        reshaped.attrs.update({"long_name": long_name, "cmap": "coolwarm"})
        return reshaped

    def _reshape_feature_component_matrix(
        self,
        matrix: npt.NDArray[np.floating],
        component_coord: npt.NDArray[np.intp],
    ) -> xr.DataArray:
        """Reshape a `(feature, component)` matrix back to spatial geometry."""
        matrix_stacked = xr.DataArray(
            matrix,
            dims=["feature", "component"],
            coords={"feature": self._feature_coord_, "component": component_coord},
        )
        reshaped = matrix_stacked.unstack("feature").transpose(
            *self.spatial_dims_, "component"
        )
        reshaped.attrs.update({"long_name": "Mixing matrix", "cmap": "coolwarm"})
        return reshaped

    def __sklearn_is_fitted__(self) -> bool:
        """Check whether the estimator has been fitted."""
        return hasattr(self, "components_") and hasattr(self, "n_components_")
