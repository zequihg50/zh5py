import math


class LocalHeap:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(8 + 2 * self._f.size_of_lengths + self._f.size_of_offsets)

        assert byts[:4] == b"HEAP"
        self._version = byts[4]

        frm, to = 8, 8 + self._f.size_of_lengths
        self._data_segment_size = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._offset_head_free_list = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_offsets
        self._address_data_segment = int.from_bytes(byts[frm:to], "little")

    @property
    def address_data_segment(self):
        return self._address_data_segment


class GlobalHeapObject:
    def __init__(self, file):
        self._f = file

        byts = self._f.read(8 + self._f.size_of_lengths)

        self._heap_object_index = int.from_bytes(byts[:2], "little")
        self._reference_count = int.from_bytes(byts[2:4], "little")
        # 4 empty bytes
        size = int.from_bytes(byts[-self._f.size_of_lengths:], "little")
        self._object_size = math.ceil(size / 8) * 8
        self._object_data = self._f.read(self._object_size)

    @property
    def index(self):
        return self._heap_object_index

    @property
    def data(self):
        return self._object_data


class GlobalHeapCollection:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset
        self._offset_data = self._o + 8 + self._f.size_of_lengths

        self._f.seek(self._o)
        byts = self._f.read(8 + self._f.size_of_lengths)

        assert byts[:4] == b"GCOL"
        assert byts[4] == 1

        self._size = int.from_bytes(byts[8:8 + self._f.size_of_lengths], "little")  # size in bytes

        # load all the values into memory
        # questionable but I/O efficient assuming the heap is not too big (important for remote data access)
        # 127 16-byte heap objects plus their overhead (the collection header of 16 bytes and the 16 bytes
        # of information about each heap object)
        self._values = []
        heap_object = GlobalHeapObject(self._f)
        while heap_object.index != 0 and heap_object.index < 128:
            self._values.append(heap_object)
            heap_object = GlobalHeapObject(self._f)

    def __getitem__(self, item):  # item is the integer id of the global heap object (1 to N)
        if item > self._size:
            raise ValueError(
                f"Asking for global heap {self._o} object id {item} greater than collection size {self._size}.")

        return self._values[item]


# The Global Heap is the set of collections, which are independent of each other and
# can be localized only when reference by some other object
class GlobalHeap:
    def __init__(self, file):
        self._f = file
        self._collections = {}  # the key is the offset of the collection

    def __getitem__(self, item):  # item is the offset of the collection
        if item not in self._collections:
            self._collections[item] = GlobalHeapCollection(self._f, item)
        return self._collections[item]


class FractalHeapIndirectBlock:
    def __init__(self, file, offset, heap, nrows):
        self._f = file
        self._o = offset
        self._heap = heap
        self._nrows = nrows

        self._f.seek(self._o)
        byts = self._f.read(5 + self._f.size_of_offsets)
        assert byts[:4] == b"FHIB"
        self._version = byts[4]
        self._heap_header_address = int.from_bytes(byts[5:5 + self._f.size_of_offsets], "little")  # for integrity
        self._block_offset = int.from_bytes(self._f.read(self._heap.maximum_heap_size), "little")

        self._offset_data = self._f.tell()

    def read(self):
        # py5 _indirect_info (is this only for root indirect block?)
        nobjects = self._nrows * self._heap.table_width
        ndirect_max = self._heap.max_dblock_rows * self._heap.table_width
        if self._nrows <= ndirect_max:
            ndirect = nobjects
            nindirect = 0
        else:
            ndirect = ndirect_max
            nindirect = nobjects - ndirect_max

        self._f.seek(self._offset_data)
        direct, indirect = [], []
        for i in range(ndirect):
            address = int.from_bytes(self._f.read(8), "little")
            if address == self._f.undefined_address:
                break
            row = i // self._heap.table_width
            block_size = 2 ** max(row - 1, 0) * self._heap.starting_block_size
            direct.append((address, block_size))
        for i in range(nindirect):
            address = int.from_bytes(self._f.read(8), "little")
            if address == self._f.undefined_address:
                break
            row = i // self._heap.table_width
            block_size = 2 ** max(row - 1, 0) * self._heap.starting_block_size
            indirect.append((address, block_size))

        for address, block_size in direct:
            block = FractalHeapDirectBlock(self._f, address, self._heap, block_size)
            obj = block
            yield obj

        for address, nrows in indirect:
            block = FractalHeapIndirectBlock(self._f, address, self._heap, nrows)
            for obj in block.read():
                yield obj


class FractalHeapDirectBlock:
    def __init__(self, file, offset, heap, size):
        self._f = file
        self._o = offset
        self._heap = heap
        self._size = size

        self._f.seek(self._o)
        byts = self._f.read(5 + self._f.size_of_offsets)
        assert byts[:4] == b"FHDB"
        assert byts[4] == 0

        self._heap_header_address = int.from_bytes(byts[5:5 + self._f.size_of_offsets], "little")

        nbyts = math.ceil(self._heap.maximum_heap_size / 8)
        self._block_offset = int.from_bytes(self._f.read(nbyts), "little")

    def read(self):
        self._f.seek(self._o)
        byts = self._f.read(self._size)
        return byts

    @property
    def offset(self):
        return self._o

    @property
    def size(self):
        return self._size


class FractalHeap:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        size = (22 + 12 * self._f.size_of_lengths + 3 * self._f.size_of_offsets)
        byts = self._f.read(size)

        assert byts[:4] == b"FRHP"
        self._version = byts[4]
        self._head_id_length = int.from_bytes(byts[5:7], "little")
        self._io_filters_encoded_length = int.from_bytes(byts[7:9], "little")
        self._flags = byts[9]
        self._maximum_size_managed_objects = int.from_bytes(byts[10:14], "little")
        frm, to = 14, 14 + self._f.size_of_lengths
        self._next_huge_object_id = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_offsets
        self._v2_btree_address_huge_objects = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._amount_free_space_managed_blocks = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_offsets
        self._address_managed_block_fsm = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._amount_managed_space_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._amount_allocated_managed_space_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._offset_direct_block_allocation_iterator_managed_space = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._number_managed_objects_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._size_huge_objects_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._number_huge_objects_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._size_tiny_objects_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._number_tiny_objects_in_heap = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + 2
        self._table_width = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._starting_block_size = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_lengths
        self._maximum_direct_block_size = int.from_bytes(byts[frm:to], "little")

        frm, to = to, to + 2
        #  The value stored is the log2 of the actual value,
        #  that is: the number of bits of the address space.
        #  'Huge' and 'tiny' objects are not counted in this value,
        #  since they do not store objects in the linear address space of the heap.
        self._maximum_heap_size_bits = int.from_bytes(byts[frm:to], "little")
        self._maximum_heap_size = self.nbits // 8 + min(self.nbits % 8, 1)  # from pyfive

        frm, to = to, to + 2
        self._starting_n_of_rows_indirect_block = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + self._f.size_of_offsets
        self._address_root_block = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + 2
        self._current_n_of_rows_in_root_indirect_block = int.from_bytes(byts[frm:to], "little")

        self._size_of_filtered_root_direct_block = None
        self._io_filter_mask = None
        self._io_filter_information = None
        if self._io_filters_encoded_length > 0:
            self._size_of_filtered_root_direct_block = int.from_bytes(self._f.read(self._f.size_of_lengths), "little")
            self._io_filter_mask = self._f.read(4)
            self._io_filter_information = self._f.read(self._io_filters_encoded_length)

        self._checksum = self._f.read(4)

        # things I don't understand
        # this is the fractal heap ID for managed objects (tiny, huge, managed)
        value = min(self._maximum_direct_block_size, self._maximum_size_managed_objects)
        value = value.bit_length()
        self._managed_object_length_size = value // 8 + min(value % 8, 1)

        self._managed = []
        if self._address_root_block != self._f.undefined_address:
            nrows = self._current_n_of_rows_in_root_indirect_block
            if nrows > 0:
                block = FractalHeapIndirectBlock(self._f, self._address_root_block, self, nrows)
                for b in block.read():
                    self._managed.append(b)
            else:
                block = FractalHeapDirectBlock(self._f, self._address_root_block, self, self._starting_block_size)
                self._managed.append(block)

    @property
    def nbits(self):
        return self._maximum_heap_size_bits

    @property
    def maximum_heap_size(self):
        return self._maximum_heap_size

    @property
    def table_width(self):
        return self._table_width

    @property
    def starting_block_size(self):
        return self._starting_block_size

    @property
    def max_dblock_rows(self):
        log2_maximum_direct_block_size = int(math.log2(self._maximum_direct_block_size))
        log2_starting_block_size = int(math.log2(self._starting_block_size))

        assert 2 ** log2_maximum_direct_block_size == self._maximum_direct_block_size
        assert 2 ** log2_starting_block_size == self._starting_block_size

        return log2_maximum_direct_block_size - log2_starting_block_size + 2

    def get_data(self, heap_id):
        firstbyte = heap_id[0]
        reserved = firstbyte & 15  # bit 0-3
        idtype = (firstbyte >> 4) & 3  # bit 4-5
        version = firstbyte >> 6  # bit 6-7
        data_offset = 1
        if idtype == 0:  # managed
            assert version == 0
            nbytes = self._maximum_heap_size
            offset = int.from_bytes(heap_id[data_offset:data_offset + nbytes], "little")
            data_offset += nbytes

            nbytes = self._managed_object_length_size
            size = int.from_bytes(heap_id[data_offset:data_offset + nbytes], "little")

            # calculate direct block offset
            # ToDo this involves having all the fractal heap read into memory, fix (need to implement proper reading
            # of the doubling table)
            acc = 0
            block_id = 0
            for i, block in enumerate(self._managed):
                block_length = block.size
                acc += block_length
                if offset < acc:
                    block_id = i
                    acc -= block_length
                    break

            # return self._managed[offset:offset + size] # Do not return data, return the offset in the file
            a = self._managed[block_id].offset + (offset - acc)
            return a
        elif idtype == 1:  # tiny
            raise NotImplementedError
        elif idtype == 2:  # huge
            raise NotImplementedError
        else:
            raise NotImplementedError
