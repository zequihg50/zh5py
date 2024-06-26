class FractalHeap:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        size = (19 + 12 * self._f.size_of_lengths + 3 * self._f.size_of_offsets)
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

        # estoy leyendo los 512 bytes a pelo porq se q es pequeño
        # pero tengo que entender como sacar info del fractal heap
        self._f.seek(self._address_root_block)
        self._data = self._f.read(512)  # block size
        # ahora tengo los 512 bytes del direct block

    @property
    def nbits(self):
        return self._maximum_heap_size_bits

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

            # return self._data[offset:offset+size]  # esto es la ñapa del __init__ al final
            return self._address_root_block + offset
        elif idtype == 1:  # tiny
            raise NotImplementedError
        elif idtype == 2:  # huge
            raise NotImplementedError
        else:
            raise NotImplementedError
