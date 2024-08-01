import numcodecs


class FilterDescription:
    @property
    def id(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def client_data(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError


class FilterDescriptionV1(FilterDescription):
    def __init__(self, file, offset):
        self._f = file
        self._o = offset
        self._offset_data = offset + 8

        self._f.seek(offset)
        byts = self._f.read(8)

        self._id = int.from_bytes(byts[:2], "little")
        self._name_length = int.from_bytes(byts[2:4], "little")
        self._flags = int.from_bytes(byts[4:6], "little")
        self._number_client_data_values = int.from_bytes(byts[6:8], "little")

        assert self._id != 0

        # attributes that can be cached
        self._name = None
        self._size = None
        self._client_data = None

    @property
    def id(self):
        return self._id

    @property
    def size(self):
        if self._size is None:
            padding = 4 if self._number_client_data_values % 2 else 0
            self._size = 8 + self._name_length + 4 * self._number_client_data_values + padding

        return self._size

    @property
    def name(self):
        if self._name is None:
            self._f.seek(self._offset_data)
            name = self._f.read(self._name_length)
            if int.from_bytes(name, "little") != 0:
                self._name = name.replace(b"\x00", b"").decode("ascii")
            else:
                self._name = ""

        return self._name

    @property
    def client_data(self):
        if self._client_data is None:
            self._f.seek(self._offset_data)
            client_data = self._f.read(self._name_length + 4 * self._number_client_data_values)[self._name_length:]
            self._client_data = client_data

        return self._client_data


class FilterDescriptionV2(FilterDescription):
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        self._id = int.from_bytes(self._f.read(2), "little")

        if self._id < 256:
            self._name_length = 0
            self._offset_data = self._o + 6
        else:
            self._name_length = int.from_bytes(self._f.read(2), "little")
            self._offset_data = self._o + 8

        byts = self._f.read(4)
        self._flags = byts[0:2]
        self._number_client_data_values = int.from_bytes(byts[2:4], "little")

        # attributes that can be cached
        self._name = None
        self._size = None
        self._client_data = None

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        if self._name is None:
            if self._name_length == 0:
                self._name = ""
            else:
                self._f.seek(self._offset_data)
                byts = self._f.read(self._name_length)
                self._name = byts.replace(b"\x00", b"").decode("ascii")

        return self._name

    @property
    def size(self):
        if self._size is None:
            self._size = 8 + len(self.name) + 4 * self._number_client_data_values

        return self._size

    @property
    def client_data(self):
        if self._client_data is None:
            self._f.seek(self._offset_data)
            self._client_data = self._f.read(
                self._name_length + 4 * self._number_client_data_values)[self._name_length:]

        return self._client_data


class FilterPipeline:
    def filter_descriptions(self):
        raise NotImplementedError

    def filters(self):
        for fd in self.filter_descriptions():
            if fd.id == 1:
                yield numcodecs.Zlib(int.from_bytes(fd.client_data, "little"))
            elif fd.id == 2:
                yield numcodecs.Shuffle(int.from_bytes(fd.client_data, "little"))
            elif fd.id == 3:
                yield numcodecs.Fletcher32()
            else:
                raise ValueError(f"Not supporter filter with id {fd.id}.")


class FilterPipelineMessageV1(FilterPipeline):
    def __init__(self, file, offset):
        self._f = file
        self._o = offset
        self._filters_offset = offset + 8

        self._f.seek(self._o)
        byts = self._f.read(2)
        self._version = byts[0]
        self._number_of_filters = byts[1]  # max 32

        assert self._version == 1
        assert self._number_of_filters <= 32

    def filter_descriptions(self):
        filters = []
        offset = self._filters_offset
        for i in range(self._number_of_filters):
            f = FilterDescriptionV1(self._f, offset)
            filters.append(f)
            offset += f.size

        return filters


class FilterPipelineMessageV2(FilterPipeline):
    def __init__(self, file, offset):
        self._f = file
        self._o = offset
        self._filters_offset = offset + 2

        self._f.seek(self._o)
        byts = self._f.read(2)
        self._version = byts[0]
        self._number_of_filters = byts[1]

    def filter_descriptions(self):
        filters = []
        offset = self._filters_offset
        for i in range(self._number_of_filters):
            f = FilterDescriptionV2(self._f, offset)
            filters.append(f)
            offset += f.size

        return filters
