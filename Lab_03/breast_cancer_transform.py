"""preprocessing_fn for the breast cancer feature engineering pipeline.

Transform runs this once over the full training split to compute whatever
statistics the tft.* analyzers need (means, variances, min/max, quantile
boundaries, vocabularies), bakes those constants into a TensorFlow graph, and
then replays that graph over every split. That is what makes the training and
serving paths agree: the eval split is scaled with the *training* statistics,
never its own.
"""

import tensorflow as tf
import tensorflow_transform as tft

import breast_cancer_constants

_MEAN_FEATURE_KEYS = breast_cancer_constants.MEAN_FEATURE_KEYS
_SE_FEATURE_KEYS = breast_cancer_constants.SE_FEATURE_KEYS
_WORST_FEATURE_KEYS = breast_cancer_constants.WORST_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = breast_cancer_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = breast_cancer_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = breast_cancer_constants.LABEL_KEY
_transformed_name = breast_cancer_constants.transformed_name
_bucketized_name = breast_cancer_constants.bucketized_name


def _fill(tensor):
    """CsvExampleGen yields sparse tensors; densify before any analyzer runs."""
    if isinstance(tensor, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(tensor, default_value=0)
    return tensor


def preprocessing_fn(inputs):
    """Map raw columns to transformed features.

    'id' is deliberately absent from the output: it is in the inferred schema
    because it is in the CSV, but feeding a record identifier to a model lets
    it memorise rows instead of learning the measurements.
    """
    outputs = {}

    # The three families are scaled differently because they are differently
    # shaped. _mean is the best-behaved -- roughly symmetric, no long tail --
    # so min-max gives a bounded [0, 1] feature with nothing important
    # compressed. _se is right-skewed, as standard errors generally are, and
    # _worst is by construction the extreme-value tail of each measurement.
    # For those two a single large value would pull the min-max range out and
    # squash the bulk of the distribution into a narrow band, so they get
    # z-score, which is unbounded but far less sensitive to the tail.
    for key in _MEAN_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(_fill(inputs[key]))

    for key in _SE_FEATURE_KEYS + _WORST_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(_fill(inputs[key]))

    for key in _BUCKET_FEATURE_KEYS:
        outputs[_bucketized_name(key)] = tft.bucketize(
            _fill(inputs[key]), _FEATURE_BUCKET_COUNT[key]
        )

    # M/B -> integer ids. Doing it here rather than in pandas means the same
    # mapping ships with the model and applies at serving time.
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        _fill(inputs[_LABEL_KEY])
    )

    return outputs
