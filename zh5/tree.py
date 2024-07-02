class BtreeV1:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        file.seek(offset)
        byts = file.read(8 + self._f.size_of_offsets * 2)
        assert byts[:4] == b"TREE"
        self._node_type = byts[4]  # 0 group, 1 dataset
        self._node_level = byts[5]  # 0 is root of the tree
        self._entries_used = int.from_bytes(byts[6:8], "little")
        self._entries_offset = file.tell()
        self._address_left_sibling = int.from_bytes(byts[8:8 + self._f.size_of_offsets], "little")
        self._address_right_sibling = int.from_bytes(
            byts[8 + self._f.size_of_offsets:8 + 2 * self._f.size_of_offsets], "little")

        # N entries = B-tree contains N child pointers and N+1 keys.
        # Each tree has 2K + 1 keys with 2K child pointers interleaved between
        # the keys.The number of keys and child pointers actually containing
        # valid values is determined by the node’s Entries Used field.If that
        # field is N, then the B-tree contains N child pointers and N+1 keys.
        n = self._entries_used + self._entries_used + 1
        bytsl = (
                self._entries_used * self._f.size_of_offsets +  # child entries
                self._entries_used * 24  # 1-4, 4-8, 1dim+64bit
        )
        byts = file.read(bytsl)
        # esto es donde está el chunk en bytes en el fichero
        # el tamanio se saca de la key
        # 24 es el tamanio de la primera key, de donde sale? 8+8*2, for type1 nodes
        # print(int.from_bytes(byts[24:24 + self._fh.size_of_offsets], "little"))

    @property
    def keysize(self):
        raise NotImplementedError

    @property
    def level(self):
        return self._node_level

    @property
    def type(self):
        return self._node_type

    @property
    def sibling_left(self):
        return self._address_left_sibling if self._address_left_sibling != self._f.undefined_address else None

    @property
    def sibling_right(self):
        return self._address_right_sibling if self._address_right_sibling != self._f.undefined_address else None

    def children(self):
        self._f.seek(self._entries_offset)
        for i in range(self._entries_used):
            self._f.read(self.keysize)  # read the key
            byts = self._f.read(self._f.size_of_offsets)  # read the child pointer


class BtreeV1Chunk(BtreeV1):
    def __init__(self, file, offset, dataset):
        super(BtreeV1Chunk, self).__init__(file, offset)
        self._dataset = dataset

    @property
    def keysize(self):
        return 8 + 8 * (self._dataset.ndim + 1)
        # return 24

    def chunk_locations(self, counter=0):
        keysize = self.keysize
        for i in range(self._entries_used):
            self._f.seek(self._entries_offset + ((keysize + self._f.size_of_offsets) * i))
            kbyts = self._f.read(keysize)  # read the key
            byts = self._f.read(self._f.size_of_offsets)  # read the child pointer

            if self.level == 0:
                # This is a good place to introduce zarr/kerchunk like indexing
                # print(f'Chunk {counter} {int.from_bytes(byts, "little")} (size: {int.from_bytes(kbyts[:4], "little")})')
                counter += 1
            else:
                back_to = self._f.tell()
                child = BtreeV1Chunk(self._f, int.from_bytes(byts, "little"), self._dataset)
                counter = child.chunk_locations(counter)
                self._f.seek(back_to)

        return counter

    def inspect_chunks(self):
        keysize = self.keysize
        for i in range(self._entries_used):
            offset = self._entries_offset + ((keysize + self._f.size_of_offsets) * i)
            self._f.seek(offset)
            kbyts = self._f.read(keysize)  # read the key
            byts = self._f.read(self._f.size_of_offsets)  # read the child pointer

            chunk_offset_list = []
            for i in range(self._dataset.ndim):
                frm = 8 + 8 * i
                to = frm + 8
                chunk_offset_list.append(int.from_bytes(kbyts[frm:to], "little"))

            if self.level != 0:
                child = BtreeV1Chunk(self._f, int.from_bytes(byts, "little"), self._dataset)
                yield from child.inspect_chunks()
            else:
                d = {"offset": int.from_bytes(byts, "little"),
                     "length": int.from_bytes(kbyts[:4], "little"),
                     "filter_mask": kbyts[4:8],
                     "chunk_offset": tuple(chunk_offset_list),
                     "type": "chunk",
                     "object": self._dataset.name}
                yield d


class BtreeV1Group(BtreeV1):
    def __init__(self, file, offset, group):
        super().__init__(file, offset)
        self._group = group

    @property
    def keysize(self):
        return self._f.size_of_lengths

    def symbol_table_entries(self):
        keysize = self.keysize
        for i in range(self._entries_used):
            offset = self._entries_offset + self._f.size_of_offsets + ((keysize + self._f.size_of_offsets) * i)
            self._f.seek(offset)
            byts = self._f.read(self._f.size_of_offsets)  # read the child pointer
            kbyts = self._f.read(keysize)  # read the key

            if self.level != 0:
                child = BtreeV1Group(self._f, int.from_bytes(byts, "little"), self._group)
                yield from child.symbol_table_entries()
            else:
                entry = {}
                entry["offset"] = int.from_bytes(kbyts, "little")
                entry["snod"] = int.from_bytes(byts, "little")
                yield entry


class BtreeV2:
    def __init__(self, file, offset):
        self._f = file
        self._f.seek(offset)

        byts = self._f.read(22 + self._f.size_of_offsets + self._f.size_of_lengths)
        assert byts[:4] == b"BTHD"

        self._version = byts[4]
        self._type = byts[5]
        self._node_size = int.from_bytes(byts[6:10], "little")
        self._record_size = int.from_bytes(byts[10:12], "little")
        self._depth = int.from_bytes(byts[12:14], "little")
        self._split_percent = byts[14]
        self._merge_percent = byts[15]
        self._root_node_address = int.from_bytes(byts[16:16 + self._f.size_of_offsets], "little")
        self._number_of_records_in_root_node = int.from_bytes(
            byts[16 + self._f.size_of_offsets:16 + 2 + self._f.size_of_offsets], "little")
        pos = 16 + 2 + self._f.size_of_offsets
        self._total_number_of_records_in_btree = byts[pos:pos + self._f.size_of_lengths]
        pos += self._f.size_of_lengths
        self._checksum = byts[-4:]

        if self._depth == 0:
            self._root_node = BtreeV2LeafNode(self._f, self._root_node_address, self)
        else:
            self._root_node = BtreeV2InternalNode(self._f, self._root_node_address, self)

    @property
    def type(self):
        return self._type

    @property
    def record_size(self):
        return self._record_size

    @property
    def nrecords(self):
        return self._number_of_records_in_root_node

    def records(self):
        yield from self._root_node.records()

    def parse_record(self):  # de momento retorno dict, ya veré como hacer esto
        d = {}
        if self._type == 6:
            byts = self._f.read(15)
            d["creation_order"] = int.from_bytes(byts[:8], "little")
            d["heap_id"] = byts[8:]

        return d


class BtreeV2LeafNode:
    def __init__(self, file, offset, tree):
        self._f = file
        self._o = offset
        self._tree = tree

        self._f.seek(self._o)
        byts = self._f.read(6)
        assert byts[:4] == b"BTLF"
        self._version = byts[4]
        self._type = byts[5]
        assert self._type == self._tree.type

        # maybe need to review this, in this example the root node is a leaf node
        # might be different if internal nodes are present
        # From HDF5 spec: size of this field is determined by the number of records
        # for this node and the record size (from the header). The format of records
        # depends on the type of B-tree.
        self._record_offset = self._f.tell()
        self._record_size = self._tree.record_size * self._tree.nrecords

        self._f.seek(self._record_offset + self._record_size)
        self._checksum = self._f.read(4)

    def records(self):
        self._f.seek(self._record_offset)
        for i in range(self._tree.nrecords):
            record = self._tree.parse_record()
            pos = self._f.tell()
            yield record
            self._f.seek(pos)


class BtreeV2InternalNode:
    def __init__(self, file, offset, tree):
        pass

    def records(self):
        yield 1
