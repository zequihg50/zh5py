import logging
import urllib


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
