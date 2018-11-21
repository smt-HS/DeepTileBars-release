class TrecWebParser:

    def __init__(self, file_path, encoding):
        self.file = open(file_path, encoding=encoding)

    def __iter__(self):
        doc = self.read_doc()
        while doc is not None:
            yield doc
            doc = self.read_doc()

    def read_doc(self):
        doc_no, dochdr, content = "", "", ""
        line = self.file.readline()
        while not line.startswith("<DOC>"):
            line = self.file.readline()
            if line == "":
                return None

        line = self.file.readline()
        while not line.startswith("<DOCNO>"):
            line = self.file.readline()
        doc_no = line.strip().lstrip("<DOCNO>").rstrip("</DOCNO>")

        line = self.file.readline()
        while not line.startswith("<DOCHDR>"):
            line = self.file.readline()
        dochdr = line
        line = self.file.readline()
        while not line.startswith("</DOCHDR>"):
            dochdr += line
            line = self.file.readline()

        line = self.file.readline()
        while not line.startswith("</DOC>"):
            content += line
            line = self.file.readline()

        return doc_no, dochdr, content



