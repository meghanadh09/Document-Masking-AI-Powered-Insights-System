import os
import glob
import shutil
import tempfile


def _masked_name(original: str) -> str:
    base = os.path.basename(original)
    stem, ext = os.path.splitext(base)
    return f"masked_{stem}{ext}"


def mask_file(input_path: str, output_path: str | None = None) -> str:
    """
    Main dispatcher: chooses the right masking/sanitize function based on extension.
    If output_path is None, writes to the same folder as `masked_<filename>`.

    Returns the path to the produced masked file.
    """
    ext = os.path.splitext(input_path)[1].lower()

    # Decide where to write
    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(input_path))
        output_path = os.path.join(out_dir, _masked_name(input_path))

    if ext == ".pdf":
        from sanitize_pdf import sanitize_pdf
        sanitize_pdf(input_path, output_path)
        return output_path

    if ext == ".pptx":
        from sanitize_pptx import sanitize_pptx
        sanitize_pptx(input_path, output_path)
        return output_path

    if ext in (".xlsx", ".xls"):
        from sanitize_xlsx import sanitize_xlsx
        sanitize_xlsx(input_path, output_path)
        return output_path

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        # image handling (face/ID/text blur)
        from blur_photo_tool import process_image
        with tempfile.TemporaryDirectory() as tmp:
            process_image(input_path, tmp)  # writes <name>_blurred.<ext> into tmp
            base = os.path.splitext(os.path.basename(input_path))[0]
            candidates = glob.glob(os.path.join(tmp, f"{base}_blurred*"))
            if candidates:
                shutil.copy2(candidates[0], output_path)
            else:
                shutil.copy2(input_path, output_path)
        return output_path

    raise ValueError(f"Unsupported file type: {ext}")
