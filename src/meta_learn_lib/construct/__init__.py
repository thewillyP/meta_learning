import warnings

warnings.filterwarnings("ignore", message="Could not resolve the type hint", category=UserWarning, module=r"plum\..*")
warnings.filterwarnings("ignore", message="Could not determine whether", category=UserWarning, module=r"plum\..*")
