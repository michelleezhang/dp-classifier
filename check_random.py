# !pip install diffprivlib
from sklearn.utils import check_random_state as skl_check_random_state
import secrets

def check_random_state(seed, secure=False):
    """Turn seed into a np.random.RandomState or secrets.SystemRandom instance.
    If secure=True, and seed is None (or was generated from a previous None seed), then secrets is used.  Otherwise a
    np.random.RandomState is used.
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None and secure is False, return the RandomState singleton used by np.random.
        If seed is None and secure is True, return a SystemRandom instance from secrets.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState or SystemRandom instance, return it.
        Otherwise raise ValueError.
    secure : bool, default: False
        Specifies if a secure random number generator from secrets can be used.
    """
    if secure:
        if isinstance(seed, secrets.SystemRandom):
            return seed

        if seed is None or seed is np.random.mtrand._rand:  # pylint: disable=protected-access
            return secrets.SystemRandom()
    elif isinstance(seed, secrets.SystemRandom):
        raise ValueError("secrets.SystemRandom instance cannot be passed when secure is False.")

    return skl_check_random_state(seed)