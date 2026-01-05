import enum
import typing

# Types


type ActivationFunction = typing.Callable[[float], float]
type ActivationFunctionDerivative = typing.Callable[[float], float]


class ActivationFunctionIdentifier(enum.Enum):
    HYPERBOLIC_TANGENT = "hyperbolic_tangent"
    IDENTITY = "identity"
    RECTIFIED_LINEAR_UNIT = "rectified_linear_unit"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    STEP = "step"


ActivationFunctionPair = typing.TypedDict(
    "ActivationFunctionPair",
    {
        "derivative": ActivationFunctionDerivative,
        "function": ActivationFunction,
        "identifier": ActivationFunctionIdentifier,
    },
)


class Dataset(enum.Enum):
    BALANCED = "balanced"
    BY_CLASS = "byclass"
    BY_MERGE = "bymerge"
    DIGITS = "digits"
    LETTERS = "letters"
    MNIST = "mnist"


class DatasetSplit(enum.Enum):
    TEST = "test"
    TRAIN = "train"


type Label = int
type LabelCharacter = str
type LabelMappings = dict[Label, LabelCharacter]
type ListImage = list[list[float]]
type TupleImage = tuple[tuple[float, ...], ...]
