"""Feature groups for the breast cancer feature engineering pipeline.

The WDBC dataset describes each nucleus with 10 base measurements, recorded
three times over: the mean across nuclei, the standard error, and the "worst"
(mean of the three largest) value. Grouping the keys by that structure rather
than listing 30 flat names keeps the transform readable and lets each family
be scaled on its own terms.
"""

BASE_MEASUREMENTS = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal_dimension",
]

MEAN_FEATURE_KEYS = [f"{name}_mean" for name in BASE_MEASUREMENTS]
SE_FEATURE_KEYS = [f"{name}_se" for name in BASE_MEASUREMENTS]
WORST_FEATURE_KEYS = [f"{name}_worst" for name in BASE_MEASUREMENTS]

NUMERIC_FEATURE_KEYS = MEAN_FEATURE_KEYS + SE_FEATURE_KEYS + WORST_FEATURE_KEYS

# Bucketizing radius_mean gives the pipeline one categorical-from-continuous
# feature, mirroring what the census lab does with age. Unlike the census lab
# the bucketed key is also a numeric key, so it gets its own output suffix and
# the scaled and bucketed versions both survive into the transformed dataset.
BUCKET_FEATURE_KEYS = ["radius_mean"]
FEATURE_BUCKET_COUNT = {"radius_mean": 4}

# 'diagnosis' is M/B, so it needs a vocabulary the way the census label did.
LABEL_KEY = "diagnosis"

# Present in the CSV and therefore in the inferred schema, but it is a record
# identifier. SchemaGen cannot know that -- the transform has to exclude it.
ID_FEATURE_KEYS = ["id"]


def transformed_name(key):
    return key + "_xf"


def bucketized_name(key):
    return key + "_bucketized_xf"
