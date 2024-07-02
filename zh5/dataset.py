import numpy as np

from zh5.codecs import FilterPipelineMessageV1, FilterPipelineMessageV2
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


class DatatypeMessage:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(8)

        self._clazz = byts[0] >> 4
        self._version = (byts[0] << 4) >> 4
        # the three class bit fields
        self._b0 = byts[1]
        self._b8 = byts[2]
        self._b16 = byts[3]

        self._size = int.from_bytes(byts[4:8], "little")

    @property
    def clazz(self):
        return self._clazz

    @property
    def version(self):
        return self._version

    @property
    def class_bit_fields(self):
        return self._b0, self._b8, self._b16

    @property
    def size(self):
        return self._size


class FloatDatatype:
    def __init__(self, message):
        self._m = message

    @property
    def dtype(self):
        byte_order = self._m.class_bit_fields[0] & 0x01
        dtype_char = "f"
        if byte_order == 0:
            byte_order_char = '<'  # little-endian
        else:
            byte_order_char = '>'  # big-endian
        dtype_string = f"{byte_order_char}f{self._m.size}"
        return dtype_string


class DataspaceMessage:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(5)
        self._version = byts[0]
        self._dimensionality = byts[1]

        self._shape = None

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def shape(self):
        if self._shape is None:
            self._f.seek(self._o + 8)
            shape = []
            byts = self._f.read(self._f.size_of_lengths * self.dimensionality)
            for i in range(self.dimensionality):
                frm = self._f.size_of_lengths * i
                to = frm + self._f.size_of_lengths
                shape.append(int.from_bytes(byts[frm:to], "little"))
            self._shape = tuple(shape)

        return self._shape


class Dataset:
    def __init__(self, file, do, name=None, dataspace=None, dtype=None):
        self._f = file
        self._do = do
        self._name = name
        self._dataspace = dataspace
        self._dtype = dtype

    @property
    def name(self):
        if self._name is None:
            raise NotImplementedError
        return self._name

    @property
    def ndim(self):
        return self.dataspace.dimensionality

    @property
    def shape(self):
        return self.dataspace.shape

    @property
    def dataspace(self):
        if self._dataspace is None:
            for m in self._do.msgs():
                if m["type"] == 0x0001:
                    self._dataspace = DataspaceMessage(self._f, m["offset"])
                    break
        return self._dataspace

    @property
    def dtype(self):
        if self._dtype is None:
            for m in self._do.msgs():
                if m["type"] == 0x0003:
                    dtype = DatatypeMessage(self._f, m["offset"])

                    if dtype.clazz == 1:
                        self._dtype = FloatDatatype(dtype)
                    else:
                        raise ValueError("Not implemented datatype.")

                    break
        return self._dtype.dtype

    def msgs(self):
        for m in self._do:
            yield m

    def inspect_chunks(self):
        raise NotImplementedError

    # lo suyo esq cada tipo de dataset implemente esto
    def __getitem__(self, item):
        raise NotImplementedError


class ChunkedDataset(Dataset):
    def __init__(self, file, do, name=None, dataspace=None, layout=None):
        super().__init__(file, do, name, dataspace)
        self._layout = layout

        self._f.seek(self._layout.properties_offset)
        byts = self._f.read(1 + self._f.size_of_offsets)
        self._dimensionality = byts[0]  # this is the dimensionality of the chunk, not the dataset
        self._address = int.from_bytes(byts[1:], "little")

        byts = self._f.read(
            4 * (self._dimensionality + 1 - 1))  # -1 because chunked dimensionality is +1 for whatever reason
        chunkshape = list()
        for i in range(0, len(byts) - 4, 4):
            size = int.from_bytes(byts[i:i + 4], "little")
            chunkshape.append(size)
        self._chunkshape = tuple(chunkshape)
        self._itemsize = int.from_bytes(byts[-4:], "little")

        self._filter_pipeline = None

    @property
    def address(self):
        if self._address == self._f.undefined_address:
            return None

        return self._address

    @property
    def chunkshape(self):
        return self._chunkshape

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def filter_pipeline(self):
        if self._filter_pipeline is None:
            for m in self._do.msgs():
                if m["type"] == 0x000B:
                    self._f.seek(m["offset"])
                    version = int.from_bytes(self._f.read(1), "little")
                    if version == 1:
                        self._filter_pipeline = FilterPipelineMessageV1(self._f, m["offset"])
                    elif version == 2:
                        self._filter_pipeline = FilterPipelineMessageV2(self._f, m["offset"])
                    else:
                        raise ValueError("Invalid version for filter pipeline.")

        return self._filter_pipeline

    def inspect_chunks(self):
        layout, dataspace = None, None
        for m in self._do.msgs():
            if m["type"] == 8:
                # layout = DataLayoutMessageV3(self._f, m["offset"])
                # c = ChunkedDataset(self._f, layout.properties_offset)
                b = BtreeV1Chunk(self._f, self.address, self)
                counter = 0
                # this assumes btree yields chunks in order
                for chunk in b.inspect_chunks():
                    c = chunk.copy()
                    c["id"] = counter
                    yield c
                    counter += 1

    def get_chunk_coords(self, hyperslab):  # this returns linear projection
        ndim = self.ndim
        chunk_queue = [[None] * ndim]

        for dim in range(ndim):
            start = hyperslab[dim].start or 0
            stop = hyperslab[dim].stop or self.shape[dim]
            step = hyperslab[dim].step or 1

            for i in range(len(chunk_queue)):
                chunk = chunk_queue.pop(0)
                prev = None
                for j in range(start, stop, step):
                    c = chunk.copy()
                    c[dim] = j // self.chunkshape[dim]
                    if dim == ndim - 1:
                        if prev is None or prev != c:
                            prev = c
                            yield tuple(c)
                    elif not chunk_queue or c != chunk_queue[-1]:
                        chunk_queue.append(c)

    def get_chunk_coords_dataset_projection(self, hyperslab):
        ndim = self.ndim
        chunk_queue = [[None] * ndim]

        for dim in range(ndim):
            start = hyperslab[dim].start or 0
            stop = hyperslab[dim].stop or self.shape[dim]
            step = hyperslab[dim].step or 1

            for i in range(len(chunk_queue)):
                chunk = chunk_queue.pop(0)
                prev = None
                for j in range(start, stop, step):
                    c = chunk.copy()
                    c[dim] = (j // self.chunkshape[dim]) * self.chunkshape[dim]
                    if dim == ndim - 1:
                        if prev is None or prev != c:
                            prev = c
                            yield tuple(c)
                    elif not chunk_queue or c != chunk_queue[-1]:
                        chunk_queue.append(c)

    def normalize_slice(self, s, dim):
        return slice(s.start or 0, s.stop or self.shape[dim], s.step or 1)

    def __getitem__(self, item):
        ndim = len(self.shape)

        normalized_hyperslab = []  # list of slices
        if not isinstance(item, tuple):
            if isinstance(item, slice):
                normalized_hyperslab.append(self.normalize_slice(item, 0))
            elif isinstance(item, int):
                normalized_hyperslab.append(slice(item, item + 1, 1))

            for dim in range(ndim - 1):
                normalized_hyperslab.append(slice(0, self.shape[dim + 1], 1))
        elif len(item) < ndim:
            for dim in range(len(item)):
                if isinstance(item[dim], int):
                    normalized_hyperslab.append(slice(item[dim], item[dim] + 1, 1))
            for dim in range(len(item), ndim):
                normalized_hyperslab.append(slice(0, self.shape[dim], 1))
        else:
            for dim in range(ndim):
                if isinstance(item[dim], int):
                    normalized_hyperslab.append(slice(item[dim], item[dim] + 1, 1))
                else:
                    normalized_hyperslab.append(self.normalize_slice(item[dim], dim))
        normalized_hyperslab = tuple(normalized_hyperslab)

        # get the chunks and fill the numpy array
        chunks = np.array(list(self.get_chunk_coords_dataset_projection(tuple(normalized_hyperslab))))

        # create array to store the data
        padded_shape = tuple(chunks.max(axis=0, initial=0) -
                             chunks.min(axis=0, initial=max(self.shape)) +
                             np.array(self.chunkshape))
        data = np.empty(padded_shape, dtype="f4")

        chunk_origin = chunks.min(axis=0, initial=max(self.shape))
        available_chunks = list(self.inspect_chunks())
        for requested_chunk in chunks:
            for available_chunk in available_chunks:
                if tuple(requested_chunk) == available_chunk["chunk_offset"]:
                    # read chunk data
                    chunk_buffer = self._f.read_chunk(
                        available_chunk["offset"], available_chunk["length"])

                    # filter pipeline
                    if self.filter_pipeline:
                        filters = list(self.filter_pipeline.filters())
                        for f in filters[::-1]:
                            chunk_buffer = f.decode(chunk_buffer)

                    # chunk array
                    chunk_arr = np.frombuffer(chunk_buffer, self.dtype).reshape(self.chunkshape)

                    chunk_vector = np.array(available_chunk["chunk_offset"])
                    region = tuple([slice(i, i + j) for i, j in zip((chunk_vector - chunk_origin), self.chunkshape)])
                    data[region] = chunk_arr

        # restrict the selection to the area requested by the user
        region = tuple([slice(s.start - chunk_origin[i], s.stop - chunk_origin[i], s.step)
                        for i, s in enumerate(normalized_hyperslab)])
        return data[region]
