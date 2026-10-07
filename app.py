import os
import io
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

# local imports
from masking_tool import mask_file
import ai_insights  # <-- added: your AI Insights module

# ------------------------------------------------------------------------------------
# Folders
# ------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Optiv Masking & Insights API")

# CORS for CRA dev on 3000/3001
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def safe_filename(name: str) -> str:
    return "".join(c for c in name if c not in "\\/:*?\"<>|")

def masked_name(original: str) -> str:
    base, ext = os.path.splitext(original)
    return f"masked_{base}{ext}"

def write_temp_copy(upload: UploadFile) -> str:
    """Save UploadFile to a temp path and return it."""
    suffix = os.path.splitext(upload.filename or "")[1]
    fd, tmp_path = tempfile.mkstemp(prefix="up_", suffix=suffix, dir=UPLOAD_DIR)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path

# ------------------------------------------------------------------------------------
# MASK: single file -> masked file download
# ------------------------------------------------------------------------------------
@app.post("/api/mask-file")
async def api_mask_file(file: UploadFile = File(...)):
    """
    Accept one file (pdf/xlsx/pptx/jpg/png/...) and return the masked file.
    Frontend expects a real download (Content-Disposition attachment).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    in_tmp = write_temp_copy(file)
    out_name = masked_name(safe_filename(file.filename))
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        mask_file(in_tmp, out_path)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Masking failed: {e}")
    finally:
        try:
            os.remove(in_tmp)
        except Exception:
            pass

    if not os.path.exists(out_path):
        raise HTTPException(status_code=500, detail="Masked file not produced.")

    # Return as download
    return FileResponse(
        path=out_path,
        media_type="application/octet-stream",
        filename=out_name,
    )

# ------------------------------------------------------------------------------------
# MASK: zip/folder -> zip of masked files
# ------------------------------------------------------------------------------------
@app.post("/api/mask-zip")
async def api_mask_zip(file: UploadFile = File(...)):
    """
    Accept a .zip that contains files (optionally in subfolders).
    Mask each supported file and return a ZIP of the masked results
    preserving the relative structure.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    # Save uploaded zip to temp
    tmp_zip_path = write_temp_copy(file)

    # Prepare working dirs
    work_dir = tempfile.mkdtemp(prefix="zip_in_")
    out_dir = tempfile.mkdtemp(prefix="zip_out_")

    try:
        # Extract
        with zipfile.ZipFile(tmp_zip_path, "r") as zf:
            zf.extractall(work_dir)

        # Walk and mask
        masked_count = 0
        for root, _, files in os.walk(work_dir):
            for fname in files:
                rel_dir = os.path.relpath(root, work_dir)
                in_path = os.path.join(root, fname)
                rel_out_dir = os.path.join(out_dir, rel_dir)
                os.makedirs(rel_out_dir, exist_ok=True)

                base, ext = os.path.splitext(fname)
                out_name = masked_name(fname) if ext.lower() in [".pdf", ".pptx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"] else fname
                out_path = os.path.join(rel_out_dir, out_name)

                try:
                    mask_file(in_path, out_path)
                    masked_count += 1
                except ValueError:
                    # Unsupported file: copy through unchanged to keep structure
                    shutil.copy2(in_path, os.path.join(rel_out_dir, fname))
                except Exception:
                    # If masking fails for a file, copy original to still return a zip
                    shutil.copy2(in_path, os.path.join(rel_out_dir, fname))

        if masked_count == 0:
            raise HTTPException(status_code=400, detail="No supported files were found to mask.")

        # Create return ZIP
        result_name = f"masked_folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        result_path = os.path.join(OUTPUT_DIR, result_name)
        with zipfile.ZipFile(result_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    arcname = os.path.relpath(abs_path, out_dir)
                    zf.write(abs_path, arcname)

        return FileResponse(
            result_path,
            media_type="application/zip",
            filename=result_name,
        )
    finally:
        # cleanup temps
        for p in [tmp_zip_path, work_dir, out_dir]:
            try:
                if not p:
                    continue
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ------------------------------------------------------------------------------------
# INSIGHTS: single file -> Excel download
# ------------------------------------------------------------------------------------
@app.post("/api/insights-file")
async def api_insights_file(file: UploadFile = File(...)):
    """
    Accept a single file (pdf/jpg/png/xlsx/pptx). We place it into a temp dir and
    call ai_insights.analyze_directory_to_excel on that directory.
    Returns the resulting Excel directly.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Temp placement
    tmp_in_path = write_temp_copy(file)
    work_dir = tempfile.mkdtemp(prefix="ins_dir_")
    out_xlsx_name = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    final_out_path = os.path.join(OUTPUT_DIR, out_xlsx_name)

    try:
        # Put uploaded file inside a directory so analyzer can walk it
        dst = os.path.join(work_dir, os.path.basename(tmp_in_path))
        shutil.move(tmp_in_path, dst)

        # Run your analyzer
        produced_xlsx = ai_insights.analyze_directory_to_excel(
            input_dir=work_dir,
            output_excel_path=final_out_path
        )

        if not os.path.exists(produced_xlsx):
            raise RuntimeError("AI insights did not produce an output file.")

        return FileResponse(
            path=produced_xlsx,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=os.path.basename(produced_xlsx),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights failed: {e}")
    finally:
        # cleanup
        try:
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
        except Exception:
            pass
        shutil.rmtree(work_dir, ignore_errors=True)

# ------------------------------------------------------------------------------------
# INSIGHTS: zip/folder -> Excel download
# ------------------------------------------------------------------------------------
@app.post("/api/insights-zip")
async def api_insights_zip(file: UploadFile = File(...)):
    """
    Accept a .zip of a folder. Extract it, run ai_insights.analyze_directory_to_excel
    over the extracted directory, and return the resulting Excel.
    """
    if not (file.filename and file.filename.lower().endswith(".zip")):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    tmp_zip_path = write_temp_copy(file)
    out_xlsx_name = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    final_out_path = os.path.join(OUTPUT_DIR, out_xlsx_name)

    extracted_dir = None
    try:
        # Use your helper to extract to a temp dir
        extracted_dir = ai_insights.extract_zip_to_tempdir(tmp_zip_path)

        # Run analyzer on the folder
        produced_xlsx = ai_insights.analyze_directory_to_excel(
            input_dir=extracted_dir,
            output_excel_path=final_out_path
        )

        if not os.path.exists(produced_xlsx):
            raise RuntimeError("AI insights did not produce an output file.")

        return FileResponse(
            path=produced_xlsx,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=os.path.basename(produced_xlsx),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights failed: {e}")
    finally:
        # cleanup temps
        try:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)
        except Exception:
            pass
        try:
            if extracted_dir and os.path.isdir(extracted_dir):
                shutil.rmtree(extracted_dir, ignore_errors=True)
        except Exception:
            pass

# ------------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}
