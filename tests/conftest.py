import pytest

from cirrhosis_ml.config import PBC_CACHE_DIR
from cirrhosis_ml.pbc.load import CACHE_FILENAME, load_pbc


@pytest.fixture(scope="session")
def pbc():
    """Prepared PBC data; downloads UCI 878 once if the local cache is absent."""
    try:
        return load_pbc()
    except ConnectionError:
        if not (PBC_CACHE_DIR / CACHE_FILENAME).is_file():
            pytest.skip("PBC cache absent and UCI unreachable")
        raise
