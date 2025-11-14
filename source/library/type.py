import enum

# Types


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
