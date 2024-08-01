import logging
import math


def find_nearest_multiple_8bytes(n):
    return math.ceil(n / 8) * 8


class AttributeMessage:
    def __init__(self, fh, offset, size):
        self._fh = fh
        self._offset = offset
        self._size = size

        fh.seek(offset)
        byts = fh.read(8)
        self._version = byts[0]
        self._name_size = find_nearest_multiple_8bytes(int.from_bytes(byts[2:4], "little"))
        self._datatype_size = find_nearest_multiple_8bytes(int.from_bytes(byts[4:6], "little"))
        self._dataspace_size = find_nearest_multiple_8bytes(int.from_bytes(byts[6:8], "little"))

    @property
    def version(self):
        return self._version

    @property
    def name(self):
        self._fh.seek(self._offset + 8)
        byts = self._fh.read(self._name_size)
        logging.debug(f"Read attribute name bytes {byts}")

        return byts.decode("utf-8").replace("\x00", "")

    @property
    def value(self):
        offset = 8 + self._name_size + self._dataspace_size + self._datatype_size
        self._fh.seek(self._offset + offset)
        byts = self._fh.read(self._size - offset)
        logging.debug(f"Read attribute value bytes {byts}")

        return byts.decode("utf-8").replace("\x00", "")
