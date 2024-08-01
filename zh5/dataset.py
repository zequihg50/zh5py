import asyncio

import aiohttp
import numpy as np

from zh5.codecs import FilterPipelineMessageV1, FilterPipelineMessageV2
from zh5.dtypes import DatatypeMessage, FloatDatatype, VLStringDatatype, FixedPointDatatype
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
    def layout_class(self):
        return self._layout_class

    @property
    def properties_offset(self):
        return self._properties_offset


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

        if self._dtype is None:
            self.dtype  # Just to initialize the dtype

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
                    dtype_message = DatatypeMessage(self._f, m["offset"])

                    if dtype_message.clazz == 0:
                        self._dtype = FixedPointDatatype(self._f, dtype_message)
                    elif dtype_message.clazz == 1:
                        self._dtype = FloatDatatype(self._f, dtype_message)
                    elif dtype_message.clazz == 9:  # variable-length
                        b0, b8, b16 = dtype_message.class_bit_fields
                        if b0 == 0:  # sequence: variable-length sequence of any datatype
                            pass
                        elif b0 == 1:  # string
                            self._dtype = VLStringDatatype(self._f, dtype_message)
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

    def _normalize_slice(self, s, dim):
        return slice(s.start or 0, s.stop or self.shape[dim], s.step or 1)

    def _normalize_hyperslab(self, item):
        ndim = self.ndim
        normalized_hyperslab = []  # list of slices

        if not isinstance(item, tuple):
            if isinstance(item, slice):
                normalized_hyperslab.append(self._normalize_slice(item, 0))
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
                    normalized_hyperslab.append(self._normalize_slice(item[dim], dim))

        return tuple(normalized_hyperslab)


class ContiguousDataset(Dataset):
    def __init__(self, file, do, name=None, dataspace=None, layout=None):
        super().__init__(file, do, name, dataspace)
        self._layout = layout

        self._f.seek(self._layout.properties_offset)
        byts = self._f.read(self._f.size_of_offsets + self._f.size_of_lengths)
        self._address = int.from_bytes(byts[:self._f.size_of_offsets], "little")
        self._size = int.from_bytes(byts[self._f.size_of_offsets:], "little")

    def inspect_chunks(self):
        pass

    def __getitem__(self, item):
        if self.address is None:
            raise ValueError(f"Uninitialized array: {self.name}.")  # ToDo return numpy array with fill value

        normalized_slice = self._normalize_hyperslab(item)
        if self._dtype.is_memmap:
            arr = np.memmap(
                filename=self._f.name,
                dtype=self.dtype,
                shape=self.shape,
                offset=self.address,
                order="C")
            return arr[tuple(normalized_slice)]
        else:
            # assume it is vlen, each cell is a global_heap_id
            heap_arr_dtype = np.dtype([
                ('unknown', np.void, 4),
                ('global_heap_collection_id', np.int64),
                ('object_id', np.int32)])
            heap_arr = np.memmap(
                filename=self._f.name,
                dtype=heap_arr_dtype,
                shape=self.shape,
                offset=self.address,
                order="C")[tuple(normalized_slice)]
            arr = np.vectorize(self._dtype.parse)(heap_arr)
            return arr

    @property
    def address(self):
        if self._address == self._f.undefined_address:
            return None  # storage not yet allocated for this array

        return self._address


class LocalChunkReader:
    def __init__(self, fname, dataset):
        self._fname = fname
        self._dataset = dataset

    def fetch_chunks(self, chunks):
        results = []
        with open(self._fname, "rb") as f:
            for chunk in chunks:
                f.seek(chunk["byte_offset"])
                byts = f.read(chunk["byte_length"])
                if self._dataset.filter_pipeline:
                    filters = list(self._dataset.filter_pipeline.filters())
                    for filt in filters[::-1]:
                        byts = filt.decode(byts)
                results.append((chunk["chunk_offset"], byts))

        return results


class HTTPChunkReader:
    def __init__(self, fname, dataset):
        self._url = fname
        self._dataset = dataset

    async def fetch_chunk(self, session, chunk_id, frm, length):
        headers = {'Range': f'bytes={frm}-{frm + length}'}
        async with session.get(self._url, headers=headers) as response:
            byts = await response.read()
        if self._dataset.filter_pipeline:
            filters = list(self._dataset.filter_pipeline.filters())
            for f in filters[::-1]:
                byts = f.decode(byts)

        return chunk_id, byts

    async def fetch_chunks_async(self, chunks):
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10)) as session:
            tasks = [
                self.fetch_chunk(session, chunk["chunk_offset"], chunk["byte_offset"], chunk["byte_length"])
                for chunk in chunks]
            results = await asyncio.gather(*tasks)

        return results

    def fetch_chunks(self, chunks):
        return asyncio.run(self.fetch_chunks_async(chunks))


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
        self._btree = None

        # init the btree chunk cache
        self._btree_idx = {}
        for chunk in self.btree.inspect_chunks():
            chunk_offset = chunk["chunk_offset"]
            self._btree_idx[chunk_offset] = (chunk["offset"], chunk["length"])

        # chunk reader
        if self._f.name.startswith("http://") or self._f.name.startswith("https://"):
            self._cr = HTTPChunkReader(self._f.name, self)
        else:
            self._cr = LocalChunkReader(self._f.name, self)

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
    def btree(self):
        if self._btree is None:
            for m in self._do.msgs():
                if m["type"] == 8:
                    self._btree = BtreeV1Chunk(self._f, self.address, self)

        return self._btree

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

    def inspect_btree(self):
        yield from self.btree.inspect_nodes()

    def inspect_chunks(self):
        layout, dataspace = None, None
        for m in self._do.msgs():
            if m["type"] == 8:
                counter = 0
                # this assumes btree yields chunks in order
                for chunk in self.btree.inspect_chunks():
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

    def __getitem__(self, item):
        normalized_hyperslab = self._normalize_hyperslab(item)

        # get the chunks and fill the numpy array
        chunks = np.array(list(self.get_chunk_coords_dataset_projection(tuple(normalized_hyperslab))))

        # create array to store the data
        padded_shape = tuple(chunks.max(axis=0, initial=0) -
                             chunks.min(axis=0, initial=max(self.shape)) +
                             np.array(self.chunkshape))
        data = np.empty(padded_shape, dtype="f4")
        chunk_origin = chunks.min(axis=0, initial=max(self.shape))

        matched_chunks = []
        for requested_chunk in chunks:
            requested_chunk_tuple = tuple(requested_chunk)
            if requested_chunk_tuple in self._btree_idx:
                matched_chunks.append({
                    "chunk_offset": requested_chunk_tuple,
                    "byte_offset": self._btree_idx[requested_chunk_tuple][0],
                    "byte_length": self._btree_idx[requested_chunk_tuple][1]})

        for chunk_offset, chunk_buffer in self._cr.fetch_chunks(matched_chunks):
            chunk_arr = np.frombuffer(chunk_buffer, self.dtype).reshape(self.chunkshape)
            chunk_vector = np.array(chunk_offset)
            region = tuple([slice(i, i + j) for i, j in zip((chunk_vector - chunk_origin), self.chunkshape)])
            data[region] = chunk_arr

        # restrict the selection to the area requested by the user
        region = tuple([slice(s.start - chunk_origin[i], s.stop - chunk_origin[i], s.step)
                        for i, s in enumerate(normalized_hyperslab)])

        return data[region]
