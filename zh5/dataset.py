from zh5.tree import BtreeV1Chunk


class DataLayoutMessageV1V2:
    def __init__(self, fh, offset):
        pass


class DataLayoutMessageV3:
    def __init__(self, fh, offset):
        self._fh = fh
        self._offset = offset

        fh.seek(offset)
        byts = fh.read(2)
        self._properties_offset = fh.tell()
        assert byts[0] == self.version
        self._layout_class = byts[1]

    @property
    def version(self):
        return 3

    @property
    def properties_offset(self):
        return self._properties_offset


class ChunkedDataset:
    def __init__(self, fh, offset):
        self._fh = fh
        self._offset = offset

        fh.seek(offset)
        byts = fh.read(1 + fh.size_of_offsets)
        self._dimensionality = byts[0]
        self._address = int.from_bytes(byts[1:], "little")
        byts = fh.read(
            4 * (self._dimensionality + 1 - 1))  # -1 because chunked dimensionality is +1 for whatever reason

    @property
    def address(self):
        return self._address


class FilterPipelineMessageV1:
    def __init__(self, fh, offset, sb):
        self._fh = fh
        self._offset = offset
        self._sb = sb

        fh.seek(offset)
        byts = fh.read(8)

        self._number_of_filters = fh[1]  # 32 max
        self._filter_description_list_offset = offset + 8

    @property
    def version(self):
        return 1

    @property
    def nfilters(self):
        return self._number_of_filters

    def filters(self):
        self._fh.seek(self._filter_description_list_offset)
        for _ in range(self.nfilters):
            byts = self._fh.read(8)

            yield


class FilterPipelineMessageV2:
    def __init__(self, fh, offset, sb):
        self._fh = fh
        self._offset = offset
        self._sb = sb


class DatatypeMessage:
    def __init__(self, byts):
        self._clazz = byts[0] >> 4
        self._version = (byts[0] << 4) >> 4

    @property
    def clazz(self):
        return self._clazz

    @property
    def version(self):
        return self._version


class DataspaceMessage:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(5)
        self._version = byts[0]
        self._dimensionality = byts[1]

    @property
    def dimensionality(self):
        return self._dimensionality


class Dataset:
    def __init__(self, file, do, name=None, dataspace=None):
        self._f = file
        self._do = do
        self._name = name
        self._dataspace = dataspace

    def chunk_locations(self):
        layout, dataspace = None, None
        for m in self._do.msgs():
            if m["type"] == 8:
                layout = DataLayoutMessageV3(self._f, m["offset"])
                c = ChunkedDataset(self._f, layout.properties_offset)
                b = BtreeV1Chunk(self._f, c.address, self)
                b.chunk_locations(0)

    @property
    def name(self):
        if self._name is None:
            raise NotImplementedError
        return self._name

    @property
    def ndims(self):
        return self.dataspace.dimensionality

    @property
    def dataspace(self):
        if self._dataspace is None:
            for m in self._do.msgs():
                if m["type"] == 0x00001:
                    self._dataspace = DataspaceMessage(self._f, m["offset"])
        return self._dataspace

    def msgs(self):
        for m in self._do:
            yield m

    def inspect_metadata(self):
        yield from self._do.inspect_metadata(self.name)

    def inspect_chunks(self):
        layout, dataspace = None, None
        for m in self._do.msgs():
            if m["type"] == 8:
                layout = DataLayoutMessageV3(self._f, m["offset"])
                c = ChunkedDataset(self._f, layout.properties_offset)
                b = BtreeV1Chunk(self._f, c.address, self)
                counter = 0
                # this assumes btree yields chunks in order
                for chunk in b.inspect_chunks():
                    c = chunk.copy()
                    c["id"] = counter
                    yield c
                    counter += 1
