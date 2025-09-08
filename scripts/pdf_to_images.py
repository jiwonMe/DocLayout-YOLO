
#!/usr/bin/env python3

import argparse
import concurrent.futures as futures
import io
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Optional deps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdf2image import convert_from_path as p2i_convert
except Exception:
    p2i_convert = None

from PIL import Image


def _log(msg: str):
    print(msg, flush=True)


def list_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.pdf"))
    raise FileNotFoundError(f"Input not found or not a PDF/dir: {input_path}")


def ensure_outdir(base_out: Path):
    (base_out / "images").mkdir(parents=True, exist_ok=True)


def resize_max_side(im: Image.Image, max_side: Optional[int]) -> Image.Image:
    if not max_side or max_side <= 0:
        return im
    w, h = im.size
    side = max(w, h)
    if side <= max_side:
        return im
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), Image.LANCZOS)


def save_image(im: Image.Image, out_path: Path, fmt: str, quality: int, optimize: bool):
    fmt = fmt.lower()
    if fmt == "jpg" or fmt == "jpeg":
        # JPEG can't save RGBA; convert if needed
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(out_path, format="JPEG", quality=quality, optimize=optimize, progressive=True)
    else:
        # PNG
        if im.mode == "P":
            im = im.convert("RGBA")
        im.save(out_path, format="PNG", optimize=optimize, compress_level=9)


def render_page_with_pymupdf(pdf_path: Path, page_index: int, dpi: int, rotate: int) -> Image.Image:
    assert fitz is not None
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0).prerotate(rotate % 360)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # Build PIL Image from pix.samples to avoid temp files
        mode = "RGB" if pix.n < 4 else "RGBA"
        im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return im


def render_page_with_pdf2image(pdf_path: Path, page_index: int, dpi: int, rotate: int) -> Image.Image:
    assert p2i_convert is not None
    # pdf2image works 1-based page numbers; use first_page=last_page
    imgs = p2i_convert(str(pdf_path), dpi=dpi, first_page=page_index + 1, last_page=page_index + 1)
    im = imgs[0]
    if rotate % 360 != 0:
        im = im.rotate(-rotate, expand=True)  # negative to match preRotate direction
    return im


def process_pdf(
    pdf_path: Path,
    out_images_dir: Path,
    fmt: str,
    dpi: int,
    max_side: Optional[int],
    grayscale: bool,
    start: Optional[int],
    end: Optional[int],
    rotate: int,
    quality: int,
    optimize: bool,
    overwrite: bool,
    prefer: str,
) -> Tuple[Path, int]:
    # Count pages
    try:
        if fitz:
            with fitz.open(pdf_path) as doc:
                n_pages = doc.page_count
        elif p2i_convert:
            # crude: read via pdf2image in two steps would be slow; assume end=None means all pages; we can probe by stepping until failure.
            # Better: fallback to PyPDF2 to count pages, but to keep deps minimal, we just do a quick pass:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            n_pages = int(info.get("Pages", 0))
        else:
            raise RuntimeError("Neither PyMuPDF (fitz) nor pdf2image is installed.")
    except Exception as e:
        _log(f"[ERROR] Failed to read {pdf_path.name}: {e}")
        return (pdf_path, 0)

    s = 1 if start is None else max(1, start)
    e = n_pages if end is None else min(end, n_pages)

    count = 0
    for page_idx_1based in range(s, e + 1):
        page_index = page_idx_1based - 1

        out_name = f"{pdf_path.stem}_p{page_idx_1based:04d}.{fmt.lower()}"
        out_path = out_images_dir / out_name
        if out_path.exists() and not overwrite:
            _log(f"[SKIP] {out_path} exists")
            count += 1
            continue

        try:
            if prefer == "pymupdf" and fitz is not None:
                im = render_page_with_pymupdf(pdf_path, page_index, dpi, rotate)
            elif prefer == "pdf2image" and p2i_convert is not None:
                im = render_page_with_pdf2image(pdf_path, page_index, dpi, rotate)
            elif fitz is not None:
                im = render_page_with_pymupdf(pdf_path, page_index, dpi, rotate)
            elif p2i_convert is not None:
                im = render_page_with_pdf2image(pdf_path, page_index, dpi, rotate)
            else:
                raise RuntimeError("No renderer available. Install 'pymupdf' or 'pdf2image+poppler'.")

            if grayscale:
                im = im.convert("L")
            im = resize_max_side(im, max_side)
            save_image(im, out_path, fmt, quality, optimize)
            count += 1
        except Exception as e:
            _log(f"[ERROR] {pdf_path.name} page {page_idx_1based}: {e}")

    return (pdf_path, count)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to dataset images: dataset/images/*.png|jpg (no labels)."
    )
    parser.add_argument("input", type=str, help="PDF file or directory containing PDFs")
    parser.add_argument("-o", "--outdir", type=str, default="dataset", help="Base output dir (default: dataset)")
    parser.add_argument("--fmt", choices=["png", "jpg", "jpeg"], default="png", help="Output image format")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--max-side", type=int, default=0, help="Resize so that max(w,h)=this (0 to disable)")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")
    parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (1-based, inclusive)")
    parser.add_argument("--rotate", type=int, default=0, help="Rotate pages clockwise by degrees (0/90/180/270)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (if fmt=jpg) (default: 95)")
    parser.add_argument("--optimize", action="store_true", help="Optimize PNG/JPEG output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--workers", type=int, default=0, help="Parallel PDFs (0=auto serial)")
    parser.add_argument("--prefer", choices=["auto", "pymupdf", "pdf2image"], default="auto",
                        help="Renderer preference (default: auto)")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_base = Path(args.outdir).expanduser().resolve()
    ensure_outdir(out_base)
    out_images_dir = out_base / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdfs(input_path)
    if not pdfs:
        _log("No PDFs found.")
        return

    _log(f"Found {len(pdfs)} PDFs. Writing images to: {out_images_dir}")

    # Process PDFs possibly in parallel (per file). Per-page parallelism is usually overkill and memory heavy.
    n_workers = args.workers
    if n_workers <= 0:
        n_workers = min(4, os.cpu_count() or 1)

    if n_workers == 1 or len(pdfs) == 1:
        total = 0
        for p in pdfs:
            _, cnt = process_pdf(
                p, out_images_dir, args.fmt, args.dpi, args.max_side, args.grayscale,
                args.start, args.end, args.rotate, args.quality, args.optimize, args.overwrite, args.prefer
            )
            total += cnt
        _log(f"Done. Saved {total} pages as images.")
    else:
        total = 0
        with futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(
                    process_pdf, p, out_images_dir, args.fmt, args.dpi, args.max_side, args.grayscale,
                    args.start, args.end, args.rotate, args.quality, args.optimize, args.overwrite, args.prefer
                )
                for p in pdfs
            ]
            for fut in futures.as_completed(futs):
                _, cnt = fut.result()
                total += cnt
        _log(f"Done. Saved {total} pages as images.")


if __name__ == "__main__":
    main()
