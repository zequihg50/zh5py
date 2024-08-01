import unittest
import logging

from zh5.file import File, PagedFile


class PageAndRemote(unittest.TestCase):
    def test_bln(self):
        f = File("https://uor-aces-o.s3-ext.jc.rl.ac.uk/bnl/19881001T0000Z/cv625a_2_1hr__198810-198810.nc")

        # 362 seconds (6 min)
        # print(f["m01s05i216"][:10].max())
        f.close()

    def test_ifca_page(self):
        f = PagedFile("https://api.cloud.ifca.es:8080/swift/v1/tests/cv625a_2_1hr__198810-198810_page.nc")
        print(f["m01s05i216"][:].max())
        # print(list(f["m01s05i216"].inspect_chunks()))

        # import pandas as pd
        # pd.DataFrame.from_records(list(f["m01s05i206"].inspect_chunks())).to_csv("chunks.csv", index=False)

        print(f.cache_hits, f.cache_misses)
        f.close()


if __name__ == "__main__":
    unittest.main()
