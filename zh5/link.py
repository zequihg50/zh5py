from zh5.heap import FractalHeap
from zh5.tree import BtreeV2


class Link:
    def solve(self):
        raise NotImplementedError


class SimpleLink(Link):
    def __init__(self, name, offset):
        self._name = name
        self._offset = offset

    def solve(self):
        return self._offset

    @property
    def name(self):
        return self._name


class LinkInfoMessage(Link):
    def __init__(self, file, offset):
        self._f = file
        self._o = offset
        self._undefined_address = b"".join([b"\xff"] * self._f.size_of_offsets)

        self._f.seek(offset)
        byts = self._f.read(2)
        self._version = byts[0]
        self._flags = byts[1]

        self._maximum_creation_index = None
        self._fractal_heap_address = None
        self._address_of_v2_btree_for_name_index = None
        self._address_of_v2_btree_for_creation_order_index = None

        s = self._f.size_of_offsets * 2
        if self._flags == 0:
            byts = self._f.read(s)
            self._fractal_heap_address = int.from_bytes(
                byts[:self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_name_index = int.from_bytes(
                byts[self._f.size_of_offsets:2 * self._f.size_of_offsets], "little")
        elif self._flags == 1:
            byts = self._f.read(s + 8)
            self._maximum_creation_index = byts[:8]
            self._fractal_heap_address = int.from_bytes(
                byts[8:self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_name_index = int.from_bytes(
                byts[8 + self._f.size_of_offsets:8 + 2 * self._f.size_of_offsets], "little")
        elif self._flags == 2:
            byts = self._f.read(s)
            self._fractal_heap_address = int.from_bytes(
                byts[:self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_name_index = int.from_bytes(
                byts[self._f.size_of_offsets:2 * self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_creation_order_index = int.from_bytes(
                byts[2 * self._f.size_of_offsets:3 * self._f.size_of_offsets], "little")
        elif self._flags == 3:
            byts = self._f.read(24 + 8)
            self._maximum_creation_index = byts[:8]
            self._fractal_heap_address = int.from_bytes(byts[8:8 + self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_name_index = int.from_bytes(
                byts[8 + self._f.size_of_offsets:8 + 2 * self._f.size_of_offsets], "little")
            self._address_of_v2_btree_for_creation_order_index = int.from_bytes(
                byts[8 + 2 * self._f.size_of_offsets:8 + 3 * self._f.size_of_offsets], "little")
        else:
            raise ValueError("What?")

        self._heap = FractalHeap(self._f, self._fractal_heap_address)
        self._btree_name = BtreeV2(self._f, self._address_of_v2_btree_for_name_index)
        self._btree_order = BtreeV2(self._f, self._address_of_v2_btree_for_creation_order_index)

    def solve(self):
        for record in self._btree_order.records():
            data = self._heap.get_data(record["heap_id"])
            l = LinkMessage(self._f, data)
            yield l


class LinkMessage(Link):
    def __init__(self, fh, offset):
        self._fh = fh
        self._offset = offset

        fh.seek(offset)
        byts = fh.read(2)
        self._version = byts[0]
        self._flags = byts[1]

        if (self._flags >> 3) & 0b1:
            self._link_type = int.from_bytes(fh.read(1), "little")
        else:
            self._link_type = 0  # hard link is cero, no stored in the file

        if (self._flags >> 2) & 0b1:
            self._creation_order = int.from_bytes(fh.read(8), "little")

        if (self._flags >> 4) & 0b1:
            self._link_name_character_set = fh.read(1).decode("ascii")
        else:
            self._link_name_character_set = "ascii"

        self._length_of_link_name = int.from_bytes(fh.read(2 ** (self._flags & 0b11)), "little")
        self._link_name = fh.read(self._length_of_link_name)

        # ToDo: hard link support only at the moment
        self._link_information = int.from_bytes(fh.read(self._fh.size_of_offsets), "little")  # object header address

        # logging.debug(
        #    f"Initialised Link {self._link_name.decode(self._link_name_character_set)}, encoded as {self._link_name_character_set}, points to {self._link_information}.")

    @property
    def cs(self):
        return self._link_name_character_set

    @property
    def name(self):
        return self._link_name.decode(self._link_name_character_set)

    def solve(self):
        if self._link_type == 0:
            return self._link_information

    def __str__(self):
        return f"LinkMessage(type={self._link_type}, name={self.name})"

    def __repr__(self):
        return self.__str__()
