import logging
import struct
import urllib.request
from collections import OrderedDict

from zh5.attr import AttributeMessage
from zh5.dataset import DataspaceMessage, DataLayoutMessageV3, ChunkedDataset, ContiguousDataset
from zh5.heap import LocalHeap, GlobalHeap
from zh5.link import LinkMessage, LinkInfoMessage, SimpleLink
from zh5.tree import BtreeV1Group

SIGNATURE = b"\x89HDF\r\n\x1a\n"


class HTTPRangeReader:
    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.length = self._get_content_length()

    def _get_content_length(self):
        req = urllib.request.Request(self.url, method='HEAD')
        with urllib.request.urlopen(req) as response:
            return int(response.headers['Content-Length'])

    def read(self, size=-1):
        if size == -1:
            size = self.length - self.pos
        start = self.pos
        end = start + size - 1
        headers = {'Range': f'bytes={start}-{end}'}
        logging.debug(f"HTTP range header request: {headers}.")
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = response.read()
        self.pos += len(data)
        return data

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.length + offset
        else:
            raise ValueError("Invalid value for 'whence'.")
        self.pos = max(0, min(self.pos, self.length))

    def tell(self):
        return self.pos

    def close(self):
        pass


class DriverInformationBlock:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(16)
        self._version = byts[0]
        self._driver_information_size = int.from_bytes(byts[4:8], "little")
        self._driver_identification = byts[8:16]
        self._driver_information = self._f.read(self._driver_information_size)

    @property
    def driver_identification(self):
        return self._driver_identification

    @property
    def driver_information(self):
        return self._driver_information


class Superblock:
    @property
    def entrypoint(self):
        raise NotImplementedError

    @property
    def version(self):
        raise NotImplementedError

    @property
    def size_of_offsets(self):
        raise NotImplementedError

    @property
    def size_of_lengths(self):
        raise NotImplementedError

    @property
    def superblock_extension_address(self):
        raise ValueError(
            f"This version of the superblock (version={self.version}) does not support the superblock extension address.")

    @property
    def undefined_address(self):
        return 2 ** (self.size_of_offsets * 8) - 1

    @property
    def group_leaf_node_k(self):
        raise ValueError(f"This version of the superblock (version={self.version}) does not support Group Leaf Node K.")

    @property
    def group_internal_node_k(self):
        raise ValueError(
            f"This version of the superblock (version={self.version}) does not support Group Internal Node K.")

    @property
    def driver(self):
        raise NotImplementedError


class SuperblockV01(Superblock):
    def __init__(self, fh, offset):
        fh.seek(offset)
        byts = fh.read(24)
        self._f = fh
        self._o = offset

        self._signature = byts[0:8]
        self._version = byts[9]
        self._version_file_free_space_storage = byts[10]
        self._version_root_group_symbol_table_entry = byts[11]
        # 1 byte empty
        self._version_number_shared_header_message_format = byts[13]
        self._size_of_offsets = byts[14]
        self._size_of_lengths = byts[15]
        # 1 byte empty
        self._group_leaf_node_k = int.from_bytes(byts[17:19], "little")
        self._group_internal_node_k = int.from_bytes(byts[19:21], "little")
        self._file_consistency_flags = int.from_bytes(byts[21:], "little")

        if self._version == 1:
            byts = fh.read(4 + self.size_of_offsets * 4 + 4)
            self._indexed_storage_internal_node_k = int.from_bytes(byts[0:2], "little")
            # 2 bytes empty
            byts = byts[4:]
        else:
            byts = fh.read(self.size_of_offsets * 4 + 4)

        frm, to = 0, self.size_of_offsets
        self._base_address = int.from_bytes(byts[frm:to], "little")
        frm, to = self.size_of_offsets, self.size_of_offsets * 2
        self._address_of_file_free_space_info = int.from_bytes(byts[frm:to], "little")
        frm, to = self.size_of_offsets * 2, self.size_of_offsets * 3
        self._end_of_file_address = int.from_bytes(byts[frm:to], "little")

        frm, to = self.size_of_offsets * 3, self.size_of_offsets * 4
        self._driver_information_block_address = int.from_bytes(byts[frm:to], "little")
        self._driver_information_block = None
        if self._driver_information_block_address != self.undefined_address:
            self._driver_information_block = DriverInformationBlock(self._f, self._driver_information_block_address)

        self._root_group_symbol_table_entry = int.from_bytes(byts[-4:], "little")

    @property
    def version(self):
        return self._version

    @property
    def entrypoint(self):
        return self._root_group_symbol_table_entry

    @property
    def size(self):
        return 56

    @property
    def size_of_offsets(self):
        return self._size_of_offsets

    @property
    def size_of_lengths(self):
        return self._size_of_lengths

    @property
    def group_internal_node_k(self):
        return self._group_internal_node_k

    @property
    def group_leaf_node_k(self):
        return self._group_leaf_node_k

    @property
    def driver(self):
        return self._driver_information_block


class SuperblockV23(Superblock):
    def __init__(self, fh, offset):
        self._f = fh
        self._o = offset

        fh.seek(offset)
        header = struct.unpack_from("8sBBBc", fh.read(12))

        self._signature = header[0]
        self._version = header[1]
        self._size_of_offsets = header[2]
        self._size_of_lengths = header[3]
        self._file_consistency_flags = header[4]
        self._base_address = int.from_bytes(fh.read(self.size_of_offsets), "little")
        self._superblock_extension_address = int.from_bytes(fh.read(self.size_of_offsets), "little")
        self._end_of_file_address = int.from_bytes(fh.read(self.size_of_offsets), "little")
        self._root_group_object_header_address = int.from_bytes(fh.read(self.size_of_offsets), "little")
        self._superblock_checksum = fh.read(4)

    @property
    def version(self):
        return self._version

    @property
    def entrypoint(self):
        return self._root_group_object_header_address

    @property
    def size(self):
        return 0

    @property
    def size_of_offsets(self):
        return self._size_of_offsets

    @property
    def size_of_lengths(self):
        return self._size_of_lengths

    @property
    def superblock_extension_address(self):
        return self._superblock_extension_address

    @property
    def driver(self):
        raise NotImplementedError


class FileReadStrategy:
    def read(self, n):
        raise NotImplementedError

    def seek(self, pos):
        raise NotImplementedError

    def tell(self):
        raise NotImplementedError


class SimpleFileReadStrategy(FileReadStrategy):
    def __init__(self, file):
        self._f = file

    def read(self, n):
        return self._f.read(n)

    def seek(self, pos):
        self._f.seek(pos)

    def tell(self):
        return self._f.tell()


class PageFileReadStrategy(FileReadStrategy):
    def __init__(self, file, page_size, pos):
        self._f = file
        self._page_size = page_size

        self._pos = pos
        self._metadata_cache = {}

        self._cache_hits = 0
        self._cache_misses = 0

    def read(self, n):
        page = self._pos // self._page_size  # page id of current byte position in the file
        page_offset = self._page_size * page  # page offset in the file of the page id
        byte_diff = self._pos - page_offset  # length from page offset in the file to current file offset

        buf = bytearray(n)
        pending = n
        frm_page = byte_diff
        to_page = min(self._page_size, frm_page + pending)
        i = 0  # counts the number of pages read
        while pending > 0:
            frm_buf = n - pending
            to_buf = frm_buf + (to_page - frm_page)
            buf[frm_buf:to_buf] = self._get_page_data(page + i, frm_page, to_page)
            frm_page = 0
            pending = pending - (to_buf - frm_buf)
            to_page = min(self._page_size, frm_page + pending)
            i += 1

        self._pos += n  # update the current position

        return buf

    def seek(self, pos):
        self._pos = pos
        self._f.seek(pos)

    def tell(self):
        return self._pos

    def _read_page(self, pageid):
        pos = self._page_size * pageid
        back_to = self._f.tell()
        self._f.seek(pos)
        byts = self._f.read(self._page_size)
        self._f.seek(back_to)
        return byts

    def _get_page_data(self, pageid, frm, to):
        if pageid not in self._metadata_cache:
            self._cache_misses += 1
            self._metadata_cache[pageid] = self._read_page(pageid)
        else:
            self._cache_hits += 1

        return self._metadata_cache[pageid][frm:to]

    @property
    def cache_hits(self):
        return self._cache_hits

    @property
    def cache_misses(self):
        return self._cache_misses

    def reset_cache(self):
        self._metadata_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0


class File:
    def __init__(self, name):
        self._name = name
        if name.startswith("http://") or name.startswith("https://"):
            self._fh = HTTPRangeReader(name)
        else:
            self._fh = open(name, "rb", buffering=0)

        self._read_strategy = SimpleFileReadStrategy(self._fh)
        self._root_group = None
        self._global_heap = GlobalHeap(self)

        # init the superblock, find it at byte 0, 512, 1024, 2048, ...
        superblock_begins = 0
        self._fh.seek(superblock_begins)
        byts = struct.unpack("8sB", self._fh.read(9))
        signature = byts[0]
        if signature != SIGNATURE:
            i = 512
            while signature != SIGNATURE:
                self._fh.seek(i)
                signature = self._fh.read(8)
                superblock_begins = i
                i *= 2
        superblock_version = byts[1]
        if superblock_version == 0 or superblock_version == 1:
            self._sb = SuperblockV01(self._fh, superblock_begins)
        elif superblock_version == 2 or superblock_version == 3:
            self._sb = SuperblockV23(self._fh, superblock_begins)
        else:
            raise ValueError("Unknown superblock version.")

    def __getitem__(self, item):
        return self.root_group[item]

    def close(self):
        self._fh.close()

    @property
    def name(self):
        return self._name

    @property
    def root_group(self):
        if self._root_group is None:
            if self._sb.version < 2:
                ste = SymbolTableEntry(self, self._sb.entrypoint + self._sb.size)
                self._root_group = Group(self, ste.object_header_address)
            elif self._sb.version >= 2 & self._sb.version < 4:
                self._root_group = Group(self, self._sb.entrypoint)
        return self._root_group

    @property
    def attrs(self):
        return self.root_group.attrs

    @property
    def links(self):
        return self.root_group.links

    @property
    def size_of_offsets(self):
        return self._sb.size_of_offsets

    @property
    def size_of_lengths(self):
        return self._sb.size_of_offsets

    @property
    def undefined_address(self):
        return self._sb.undefined_address

    @property
    def group_leaf_node_k(self):
        return self._sb.group_leaf_node_k

    @property
    def group_internal_node_k(self):
        return self._sb.group_internal_node_k

    @property
    def meta_name(self):
        return self.name

    @property
    def raw_name(self):
        return self.name

    @property
    def chunk_offset(self):
        return 0

    def datasets(self):
        yield from self._root_group.datasets()

    def seek(self, pos):
        self._read_strategy.seek(pos)

    def read(self, n):
        return self._read_strategy.read(n)

    def tell(self):
        return self._read_strategy.tell()

    def _read_file_space_info(self):
        address = self._sb.superblock_extension_address
        self.seek(address)
        object_header_version = int.from_bytes(self.read(1), "little")

        if object_header_version == 1:
            oh = ObjectHeaderV1(self, address)
        elif object_header_version == 2:
            oh = ObjectHeaderV2(self, address)
        else:
            raise ValueError("Unknown object header version.")

        for m in oh.msgs():
            if m["type"] == 23:  # FileSpaceInfoMessage
                self._read_strategy.seek(m["offset"])
                byts = self._read_strategy.read(2)
                version = byts[0]

                if version == 0:
                    file_space_info = FileSpaceInfoV0()
                elif version == 1:
                    file_space_info = FileSpaceInfoV1(self, m["offset"], byts[1])
                else:
                    raise ValueError("Unknown file space info message version.")

                return file_space_info

    def inspect_metadata(self):
        yield from self._root_group.inspect_metadata()

    def get_global_heap(self, heap_id):
        return self._global_heap[heap_id]

    @property
    def driver(self):
        return self._sb.driver

    def project_chunk(self, chunk_offset):
        return chunk_offset


class PagedFile(File):
    """This class overrides access methods in order to take advantage of page buffering."""

    def __init__(self, name):
        super().__init__(name)

        if self._sb.superblock_extension_address != self.undefined_address:
            self._file_space_info = self._read_file_space_info()
        else:
            self._file_space_info = None

        self._read_strategy = PageFileReadStrategy(
            self._fh,
            self.page_size,
            self._read_strategy.tell())
        self._simple_read_strategy = SimpleFileReadStrategy(self._fh)

    def seek(self, pos):
        self._read_strategy.seek(pos)

    def read(self, n):
        return self._read_strategy.read(n)

    def tell(self):
        return self._read_strategy.tell()

    @property
    def page_size(self):
        if self._sb.superblock_extension_address == self.undefined_address:
            return 4096

        return self._file_space_info.page_size

    @property
    def cache_hits(self):
        return self._read_strategy.cache_hits

    @property
    def cache_misses(self):
        return self._read_strategy.cache_misses

    def reset_cache(self):
        self._read_strategy.reset_cache()


class SplitFile(File):
    def __init__(self, name, meta_ext=None, raw_ext=None):
        self._name = name
        self._meta_ext = meta_ext
        self._raw_ext = raw_ext
        if self._meta_ext is None:
            self._meta_ext = "-m.h5"
        if self._raw_ext is None:
            self._raw_ext = "-r.h5"

        super().__init__(f"{name}{self._meta_ext}")

        if name.startswith("http://") or name.startswith("https://"):
            with urllib.request.urlopen(self.meta_name) as response:
                self._meta = response.read()
        else:
            with open(self.meta_name, "rb") as fh:
                self._meta = fh.read()
        self._pos = 0

    @property
    def raw_name(self):
        return f"{self._name.rstrip(self._meta_ext)}{self._raw_ext}"

    @property
    def meta_name(self):
        return f"{self._name}"

    @property
    def name(self):
        return f"{self._name}"

    def read(self, n):
        byts = self._meta[self._pos:self._pos + n]
        self._pos += n
        return byts

    def seek(self, pos):
        self._pos = pos

    def tell(self):
        return self._pos

    def close(self):
        self._fh.close()

    # Properties related to the "split" driver
    @property
    def members(self):
        # 1 superblock, 2 btree, 3 raw data, 4 global heap, 5 local heap, 6 object header
        byts = self.driver.driver_information
        members = {
            "superblock": {
                "member": byts[0]
            },
            "btree": {
                "member": byts[1]
            },
            "raw": {
                "member": byts[2]
            },
            "global_heap": {
                "member": byts[3]
            },
            "local_heap": {
                "member": byts[4]
            },
            "object_header": {
                "member": byts[5]
            }
        }

        n_members = len(set([members[x]["member"] for x in members]))
        name_offset = 8 + 16 * len(members)
        for member in members:
            offset = 0 if members[member]["member"] == 1 else 1
            address_offset = 8 + offset * 16  # 16 is two times 8, one for address and one for end of address (address, length)
            members[member]["address"] = int.from_bytes(
                byts[address_offset:address_offset + 8], "little")
            members[member]["length"] = int.from_bytes(
                byts[address_offset + 8:address_offset + 16], "little")

            # ToDo
            members[member]["name"] = ""
            name_offset += 0

        return members

    def project_chunk(self, chunk_offset):
        '''The driver information block for the "split" driver is the information block for the "multi"
         driver (see Layout: Multi Driver Information).
        :return: Projected chunk offset.
        '''
        members = self.members
        offset = chunk_offset - members["raw"]["address"]
        return offset


class SymbolTableEntry:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        byts = self._f.read(self._f.size_of_offsets * 2 + 4 + 4 + 16)

        frm, to = 0, self._f.size_of_offsets
        self.link_name_offset = int.from_bytes(byts[frm:to], "little")
        frm, to = self._f.size_of_offsets, 2 * self._f.size_of_offsets
        self.object_header_address = int.from_bytes(byts[frm:to], "little")
        frm, to = to, to + 4
        self.cache_type = int.from_bytes(byts[frm:to], "little")
        # 4 empty bytes
        frm, to = to + 4, to + 4 + 16
        self.scratch_pad = int.from_bytes(byts[frm:to], "little")


class SymbolTableMessage:
    def __init__(self, file, offset, group):
        self._f = file
        self._o = offset
        self._group = group

        self._f.seek(self._o)
        byts = self._f.read(2 * self._f.size_of_offsets)
        self._btree_address = int.from_bytes(byts[0:self._f.size_of_offsets], "little")
        self._heap_address = int.from_bytes(byts[self._f.size_of_offsets:], "little")

        self._btree = BtreeV1Group(self._f, self._btree_address, self._group)
        self._heap = LocalHeap(self._f, self._heap_address)

    def links(self):
        for snod_offset in self._btree.symbol_table_entries():
            snod = snod_offset["snod"]
            symbol_table_node = SymbolTableNode(self._f, snod)
            for offset, object_header_address in symbol_table_node.links():
                self._f.seek(self._heap.address_data_segment + offset)

                # the name is null terminated
                byts = bytearray(0)
                byt = self._f.read(1)
                while byt != b"\x00":
                    byts.append(int.from_bytes(byt, "little"))
                    byt = self._f.read(1)

                link_name = byts.replace(b"\x00", b"").decode("ascii")
                link = SimpleLink(link_name, object_header_address)
                yield link


class SymbolTableNode:  # A leaf of a b-tree
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        self._f.seek(self._o)
        # byts = self._f.read(12)  # group leaf node k
        self._entry_size = 2 * self._f.size_of_offsets + 8 + 16
        byts = self._f.read(8 + 2 * self._f.group_leaf_node_k * self._entry_size)

        assert byts[:4] == b"SNOD"
        assert byts[4] == 1
        # 1 empty byte
        self._number_of_symbols = int.from_bytes(byts[6:8], "little")
        self._group_entries = byts[8:]

    def links(self):
        for i in range(self._number_of_symbols):
            frm, to = i * self._entry_size, i * self._entry_size + self._f.size_of_offsets
            link_name_offset = int.from_bytes(self._group_entries[frm:to], "little")

            frm, to = to, to + self._f.size_of_offsets
            object_header_address = int.from_bytes(self._group_entries[frm:to], "little")

            yield link_name_offset, object_header_address


class ObjectHeader:
    def msgs(self):
        raise NotImplementedError

    def inspect_metadata(self, object_name):
        raise NotImplementedError


class ObjectHeaderV1(ObjectHeader):
    def __init__(self, fh, offset):
        self._fh = fh
        self._fh.seek(offset)
        self._offset = offset

        byts = self._fh.read(16)
        self.version = byts[0]
        assert self.version == 1
        # 1 empty byte
        self.total_number_of_header_messages = int.from_bytes(byts[2:4], "little")
        self.object_reference_count = int.from_bytes(byts[4:8], "little")
        self.object_header_size = int.from_bytes(byts[8:12], "little")
        # 4 empty bytes

    @property
    def offset_data(self):
        return self._offset + 16

    def msgs(self):
        self._fh.seek(self.offset_data)
        buffer = self._fh.read(self.object_header_size)
        offset = 0
        global_offset = self.offset_data
        continuation_queue = []
        for i in range(self.total_number_of_header_messages):
            if len(buffer) == offset and len(continuation_queue) == 0:
                break  # never reach this?
            elif len(buffer) == offset:
                cm = continuation_queue.pop(0)
                self._fh.seek(cm.offset)
                buffer += self._fh.read(cm.length)
                global_offset = cm.offset

            msg = _unpack_struct_from(OrderedDict((
                ('type', 'H'),
                ('size', 'H'),
                ('flags', 'B'),
                ('reserved', '3s'),
            )), buffer, offset)
            # msg['offset'] = self.offset_data + offset + 8 # los 8 del header_msg_info_v1
            msg['offset'] = global_offset + 8
            if msg["type"] == 0x0010:  # OBJECT_CONTINUATION_MSG_TYPE 0x0010
                fh_off, size = struct.unpack_from('<QQ', buffer, offset + 8)
                cm = ContinuationMessage(fh_off, size)
                continuation_queue.append(cm)
            else:
                msg['data'] = buffer[offset:offset + msg['size']]
            offset += msg['size'] + 8
            global_offset += msg['size'] + 8

            yield msg

    def inspect_metadata(self, object_name):
        yield {"offset": self._offset, "length": 16, "type": "object_header", "object": object_name}
        for m in self.msgs():
            yield {"offset": m["offset"], "length": m["size"], "type": "object_header_message", "object": object_name}


class ObjectHeaderV2(ObjectHeader):
    def __init__(self, fh, offset):
        self._fh = fh
        self._offset = offset

        fh.seek(offset)
        byts = self._fh.read(6)
        self._signature = byts[0:4]
        assert self._signature == b"OHDR"

        self._version = byts[4]
        assert self._version == 2

        self._flags = byts[5]
        self._size_of_chunk_size = 2 ** (self._flags & 0b11)

        if self._flags & 0b100000:
            byts = self._fh.read(16)
            self._access_time = int.from_bytes(byts[0:4], "little")
            self._modification_time = int.from_bytes(byts[4:8], "little")
            self._change_time = int.from_bytes(byts[8:12], "little")
            self._birth_time = int.from_bytes(byts[12:16], "little")

        if self._flags & 0b10000:
            byts = self._fh.read(4)
            self._maximum_n_of_compact_attributes = int.from_bytes(byts[0:2], "little")
            self._minimum_n_of_dense_attributes = int.from_bytes(byts[2:4], "little")

        self._size_of_chunk = int.from_bytes(fh.read(self._size_of_chunk_size), "little")
        self._offset_data = fh.tell()

    @property
    def offset_data(self):
        return self._offset_data

    @property
    def creation_order_size(self):
        return (self._flags & 0b100) // 2

    def msgs(self):
        self._fh.seek(self.offset_data)
        buffer = self._fh.read(self._size_of_chunk)
        global_offset = self.offset_data
        offset = 0
        continuation_queue = []
        pending = len(buffer) - offset
        while pending > 8 or len(continuation_queue) > 0:  # 4 byte checksum
            if pending <= 4 + 4:  # 4 byte checksum + optional gap
                cm = continuation_queue.pop(0)
                self._fh.seek(cm.offset)
                byts = self._fh.read(cm.length)
                assert byts[:4] == b"OCHK"
                buffer = byts[4:]
                offset = 0
                global_offset = cm.offset + 4

            msg = _unpack_struct_from(OrderedDict((
                ('type', 'B'),
                ('size', 'H'),
                ('flags', 'B'),
            )), buffer, offset)
            msg['offset'] = global_offset + 4 + self.creation_order_size

            if msg["type"] == 0x0010:  # OBJECT_CONTINUATION_MSG_TYPE 0x0010
                fh_off, size = struct.unpack_from('<QQ', buffer, offset + 4 + self.creation_order_size)
                cm = ContinuationMessage(fh_off, size)
                continuation_queue.append(cm)

            offset += msg["size"] + 4 + self.creation_order_size
            global_offset += msg["size"] + 4 + self.creation_order_size
            pending = len(buffer) - offset

            yield msg

    def inspect_metadata(self, object_name):
        yield {"offset": self._offset, "length": self._offset_data - self._offset, "type": "object_header",
               "object": object_name}
        for m in self.msgs():
            yield {"offset": m["offset"], "length": m["size"], "type": "object_header_message", "object": object_name}


class ContinuationMessage:
    def __init__(self, offset, length):
        self._offset = offset
        self._length = length

    @property
    def offset(self):
        return self._offset

    @property
    def length(self):
        return self._length


class FileSpaceInfoV0:
    pass


class FileSpaceInfoV1:
    def __init__(self, file, offset, strategy):
        self._f = file
        self._o = offset

        self._strategy = strategy

        self._f.seek(self._o)
        nbyts = (3 + self._f.size_of_lengths + 5 + 13 * self._f.size_of_offsets)
        byts = self._f.read(nbyts)

        self._persisting_free_space = byts[3] != 0
        self._free_space_section_threshold = int.from_bytes(
            byts[4:4 + self._f.size_of_lengths], "little")
        self._page_size = int.from_bytes(
            byts[3 + self._f.size_of_lengths:3 + self._f.size_of_lengths + 4], "little")

    @property
    def page_size(self):
        return self._page_size


class GroupInfoMessage:
    def __init__(self, fh, offset):
        self._fh = fh
        self._offset = offset

        fh.seek(offset)
        byts = fh.read(2)
        self._version = byts[0]
        self._flags = byts[1]

        if self._flags == 1:
            byts = fh.read(4)
            self._link_phase_change_maximum_compact_value = byts[:2]
            self._link_phase_change_minimum_dense_value = byts[2:4]
        elif self._flags == 2:
            byts = fh.read(4)
            self._estimated_number_of_entries = byts[:2]
            self._estimated_link_name_length_of_entries = byts[2:4]
        elif self._flags == 3:
            byts = fh.read(8)
            self._link_phase_change_maximum_compact_value = byts[:2]
            self._link_phase_change_minimum_dense_value = byts[2:4]
            self._estimated_number_of_entries = byts[4:6]
            self._estimated_link_name_length_of_entries = byts[6:8]


class Group:
    def __init__(self, file, offset):
        self._f = file
        self._o = offset

        # guess version of data object
        self._f.seek(offset)
        byts = self._f.read(4)
        if byts == b"OHDR":
            self._do = ObjectHeaderV2(self._f, offset)
        else:
            self._do = ObjectHeaderV1(self._f, offset)

    def __getitem__(self, item):
        if isinstance(item, str):
            link = None
            # links = list(self.links())
            for l in self.links():
                if l.name == item:
                    link = l
                    break
            if link is None:
                raise ValueError

            pos = link.solve()  # byte offset of the object header
            self._f.seek(pos)
            byts = self._f.read(5)
            if byts[0:4] == b"OHDR":
                oh = ObjectHeaderV2(self._f, pos)
            elif byts[0] == 1:
                oh = ObjectHeaderV1(self._f, pos)
            else:
                raise ValueError

            # is this a dataset?
            is_dataset, dataspace, layout = False, None, None
            for m in oh.msgs():
                if m["type"] == 0x0001:  # dataspace message
                    is_dataset = True
                    dataspace = DataspaceMessage(self._f, m["offset"])
                elif m["type"] == 0x0008:  # layout message
                    layout = DataLayoutMessageV3(self._f, m["offset"])

            dataset = None
            if is_dataset:
                if layout.layout_class == 2:
                    dataset = ChunkedDataset(self._f, oh, name=item, dataspace=dataspace, layout=layout)
                elif layout.layout_class == 1:
                    dataset = ContiguousDataset(self._f, oh, name=item, dataspace=dataspace, layout=layout)
                else:
                    raise ValueError(f"Layout class not supported ({layout.layout_class}).")

            return dataset

    @property
    def name(self):
        return "/"

    @property
    def attrs(self):
        d = {}
        for msg in self._do.msgs():
            if msg["type"] == 12:
                attr = AttributeMessage(self._f, msg['offset'], msg['size'])
                d[attr.name] = attr.value
        return d

    def links(self):
        for m in self._do.msgs():
            if m["type"] == 6:  # link message
                yield LinkMessage(self._f, m['offset'])
            elif m["type"] == 2:  # link info message
                lim = LinkInfoMessage(self._f, m['offset'])
                for x in lim.solve():
                    yield x
            elif m["type"] == 17:  # symbol table message type
                symbol_table = SymbolTableMessage(self._f, m['offset'], self)
                yield from symbol_table.links()


def _unpack_struct_from(structure, buf, offset=0):
    """ Unpack a structure into an OrderedDict from a buffer of bytes. """
    fmt = '<' + ''.join(structure.values())
    values = struct.unpack_from(fmt, buf, offset=offset)
    return OrderedDict(zip(structure.keys(), values))
