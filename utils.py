def read_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def read_docs(path: Path) -> List[str]:
    return [read_doc(p) for p in path.iterdir() if p.suffix == ".md"]

