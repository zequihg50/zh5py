import unittest

from zh5.file import PagedFile


class PageAndRemote(unittest.TestCase):
    def test_ifca(self):
        f = PagedFile("https://api.cloud.ifca.es:8080/swift/v1/tests/cv625a_2_1hr__198810-198810_page.nc")
        f["m01s05i206"].chunk_locations()
        print(f.cache_hits, f.cache_misses)


if __name__ == "__main__":
    unittest.main()
