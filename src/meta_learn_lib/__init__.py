import warnings

from .backend import pytree as _pytree  # noqa: F401

warnings.filterwarnings("ignore", message="Could not resolve the type hint", category=UserWarning)
warnings.filterwarnings("ignore", message="Could not determine whether", category=UserWarning)
