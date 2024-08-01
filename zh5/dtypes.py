class DatatypeMessage:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(8)

        self._clazz = byts[0] & 0b1111
        self._version = byts[0] >> 4
        # the three class bit fields
        self._b0 = byts[1]
        self._b8 = byts[2]
        self._b16 = byts[3]
        self._size = int.from_bytes(byts[4:8], "little")
        self._properties_offset = self._o + 8

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

    @property
    def properties_offset(self):
        return self._properties_offset


class Datatype:
    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def is_memmap(self):
        '''Can be backed by a numpy memmap when stored contiguously.'''
        return NotImplementedError


class FixedPointDatatype(Datatype):
    def __init__(self, file, message):
        self._f = file
        self._m = message

        b0, b8, b16 = self._m.class_bit_fields
        self._byte_order = ">" if b0 & 1 else "<"
        self._lo_pad = (b0 >> 1) & 1
        self._hi_pad = (b0 >> 2) & 1
        self._signed = (b0 >> 3) & 1

        self._f.seek(self._m.properties_offset)
        byts = self._f.read(4)
        self._bit_offset = int.from_bytes(byts[:2], "little")
        self._bit_precision = int.from_bytes(byts[2:4], "little")

    @property
    def dtype(self):
        return f"{self._byte_order}i{self._m.size}"

    @property
    def is_memmap(self):
        return True


class FloatDatatype(Datatype):
    def __init__(self, file, message):
        self._f = file
        self._m = message

    @property
    def dtype(self):
        byte_order = self._m.class_bit_fields[0] & 0x01
        if byte_order == 0:
            byte_order_char = '<'  # little-endian
        else:
            byte_order_char = '>'  # big-endian

        dtype_string = f"{byte_order_char}f{self._m.size}"
        return dtype_string

    @property
    def is_memmap(self):
        return True


class VLStringDatatype(Datatype):
    def __init__(self, file, message):
        self._f = file
        self._m = message
        b0, b8, b16 = self._m.class_bit_fields
        self._padding = (b0 >> 4)
        self._character_set = b8 & 3

    @property
    def dtype(self):
        return "|O"

    @property
    def is_memmap(self):
        return False

    def parse(self, global_heap):
        unknown_bytes = global_heap[0]  # ToDo what is this? Undefined in the spec
        gcol_id = global_heap[1]
        object_index = global_heap[2]

        global_heap_collection = self._f.get_global_heap(gcol_id)
        data = global_heap_collection[object_index - 1].data
        data = data.replace(b"\x00", b"")
        data = data.decode("utf-8") if self._character_set == 1 else data

        return data
