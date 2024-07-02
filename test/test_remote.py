import unittest
import logging

from zh5.file import PagedFile


class PageAndRemote(unittest.TestCase):
    def test_ifca(self):
        f = PagedFile("https://api.cloud.ifca.es:8080/swift/v1/tests/cv625a_2_1hr__198810-198810_page.nc")

        import pandas as pd
        pd.DataFrame.from_records(list(f["m01s05i206"].inspect_chunks())).to_csv("chunks.csv", index=False)

        print(f.cache_hits, f.cache_misses)


if __name__ == "__main__":
    unittest.main()
