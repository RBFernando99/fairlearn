# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
from fairlearn.preprocessing import CorrelationRemover
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
import pandas as pd

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"


class Moment:
    """Generic moment.

    Our implementations of the reductions approach to fairness
    :footcite:p:`agarwal2018reductions` make use
    of :class:`Moment` objects to describe both the optimization objective
    and the fairness constraints
    imposed on the solution. This is an abstract class for all such objects.

    Read more in the :ref:`User Guide <reductions>`.
    """

    def __init__(self):
        self.data_loaded = False

    def load_data(self, X, y: pd.Series, *, sensitive_features: pd.Series = None):
        """Load a set of data for use by this object.

        Parameters
        ----------
        X : array
            The feature array
        y : :class:`pandas.Series`
            The label vector
        sensitive_features : :class:`pandas.Series`
            The sensitive feature vector (default None)
        """
        assert self.data_loaded is False, "data can be loaded only once"
        if sensitive_features is not None:
            assert isinstance(sensitive_features, pd.Series)
        self.X = X
        self._y = y
        self.tags = pd.DataFrame({_LABEL: y})
        if sensitive_features is not None:
            self.tags[_GROUP_ID] = sensitive_features
        self.data_loaded = True
        self._gamma_descr = None

    @property
    def total_samples(self):
        """Return the number of samples in the data."""
        return self.X.shape[0]

    @property
    def _y_as_series(self):
        """Return the y array as a :class:`~pandas.Series`."""
        return self._y

    def gamma(self, predictor):  # noqa: D102
        """Calculate the degree to which constraints are currently violated by the predictor."""
        raise NotImplementedError()

    def bound(self):  # noqa: D102
        """Return vector of fairness bound constraint the length of gamma."""
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):  # noqa: D102
        """Return the projected lambda values."""
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):  # noqa: D102
        """Return the signed weights."""
        raise NotImplementedError()

    def _moment_type(self):
        """Return the moment type, e.g., ClassificationMoment vs LossMoment."""
        return NotImplementedError()


# Ensure that Moment shows up in correct place in documentation
# when it is used as a base class
Moment.__module__ = "fairlearn.reductions"


class ClassificationMoment(Moment):
    """Moment that can be expressed as weighted classification error."""

    def _moment_type(self):
        return ClassificationMoment


# Ensure that ClassificationMoment shows up in correct place in documentation
# when it is used as a base class
ClassificationMoment.__module__ = "fairlearn.reductions"


class LossMoment(Moment):
    """Moment that can be expressed as weighted loss."""

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss

    def _moment_type(self):
        return LossMoment


# Ensure that LossMoment shows up in correct place in documentation
# when it is used as a base class
LossMoment.__module__ = "fairlearn.reductions"


class BiasReduction(Moment):
    """Bias Reduction moment class"""

    def __init__(self, X_train, y_train, X_test, y_test, gender_bin_test, grid_search=False):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.gender_bin_test = gender_bin_test
        self.grid_search = grid_search
        self.best_params_ = None

    def transform(self):
        remover = CorrelationRemover(sensitive_feature_ids=[9], alpha=1.0)
        remover.fit(self.X_train)
        X_train_transformed = remover.transform(self.X_train)
        X_test_transformed = remover.transform(self.X_test)

        scaler = preprocessing.StandardScaler().fit(X_train_transformed)
        X_train_transformed_scaled = scaler.transform(X_train_transformed)
        X_test_transformed_scaled = scaler.transform(X_test_transformed)

        return X_train_transformed_scaled, X_test_transformed_scaled

    def fit(self, X_train_scaled, y_train):
        if self.grid_search:
            lr_params = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], "solver": ["liblinear"]}
            lr = LogisticRegression()
            clf = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy')
            clf.fit(X_train_scaled, y_train)

            print("Best parameters set found on training set:")
            print(clf.best_params_)
            self.best_params_ = clf.best_params_

            lr = LogisticRegression(C=self.best_params_['C'], penalty=self.best_params_['penalty']
            , solver=self.best_params_['solver']
            )
        else:
            lr = LogisticRegression(solver='liblinear')

        lr.fit(X_train_scaled, y_train)

        return lr

    def predict(self, lr, X_test_scaled):
        y_test_pred = lr.predict(X_test_scaled)
        y_test_scores = lr.decision_function(X_test_scaled)

        y_test_men = self.y_test[self.gender_bin_test==0]
        y_test_scores_men = y_test_scores[self.gender_bin_test==0]

        y_test_women = self.y_test[self.gender_bin_test==1]
        y_test_scores_women = y_test_scores[self.gender_bin_test==1]

        composite_men = pd.DataFrame({'y_test_men': y_test_men, 'y_test_scores_men': y_test_scores_men})
        composite_women = pd.DataFrame({'y_test_women': y_test_women, 'y_test_scores_women': y_test_scores_women})

        return y_test_pred, composite_men, composite_women

    def _moment_type(self):
        return BiasReduction


# Ensure that LossMoment shows up in correct place in documentation
# when it is used as a base class
BiasReduction.__module__ = "fairlearn.reductions"
