from top.bbh.dataset import get_dataset as bbh
from top.bbh.utils import BBH_TASKS
from top.algebraic.dataset import get_dataset as algebraic
from top.coin.dataset import get_dataset as coin
from top.concatenation.dataset import get_dataset as concatenation
from top.got.dataset import get_dataset as got


def get_dataset(description: str, dataset_name_or_path: str):
    if description in BBH_TASKS:
        return bbh(description)
    elif description == "concatenation":
        return concatenation(dataset_name_or_path)
    elif description == "coin":
        return coin(dataset_name_or_path)
    elif description == "algebraic":
        return algebraic(dataset_name_or_path)
    elif description in ["keyword_counting", "set_intersection", "sorting"]:
        return got(description)
    else:
        raise ValueError(f"Unsupported dataset description {description}")
