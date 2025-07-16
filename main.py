from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import os
import shutil
import uuid
import re

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def extract_and_merge_text_chars(pdf_path: str) -> pd.DataFrame:
    text_lines = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = re.sub(r"[():\uFF08\uFF09\uFF1A]", "", span["text"]).strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    text_lines.append({
                        "text": text,
                        "x0": x0,
                        "x1": x1,
                        "top": y0,
                        "bottom": y1,
                        "page_num": page_num,
                    })
    doc.close()
    return pd.DataFrame(text_lines)


def highlight_chars_on_pdf(pdf_path: str, output_path: str, chars_df: pd.DataFrame):
    doc = fitz.open(pdf_path)
    page = doc[0]

    for _, row in chars_df.iterrows():
        x0, x1 = sorted([row["x0"], row["x1"]])
        top, bottom = sorted([row["top"], row["bottom"]])
        rect = fitz.Rect(x0, top, x1, bottom)

        if rect.is_empty or rect.x0 >= rect.x1 or rect.y0 >= rect.y1:
            continue

        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(fill=(1.0, 1.0, 0.0), fill_opacity=0.3)
        shape.commit()

    doc.save(output_path)


def process_pdfs(pdf1_path: str, pdf2_path: str, output_path: str):
    df1 = extract_and_merge_text_chars(pdf1_path)
    df2 = extract_and_merge_text_chars(pdf2_path)

    count_df1 = df1["text"].value_counts().rename_axis("text").reset_index(name="count_df1")
    count_df2 = df2["text"].value_counts().rename_axis("text").reset_index(name="count_df2")

    merged_counts = pd.merge(count_df1, count_df2, on="text", how="outer").fillna(0)
    merged_counts = merged_counts.astype({"count_df1": int, "count_df2": int})

    diff_counts = merged_counts.query("count_df1 != count_df2").reset_index(drop=True)
    diff_counts_base_new = diff_counts[diff_counts["count_df2"] > diff_counts["count_df1"]]

    group1 = diff_counts_base_new.query("count_df1 == 0")["text"].tolist()
    group2 = diff_counts_base_new.query("count_df1 > 0")["text"].tolist()

    results = []
    for txt in sorted(group2):
        t1 = df1[df1["text"] == txt]
        t2 = df2[df2["text"] == txt].copy()
        k = len(t2) - len(t1)
        if k <= 0:
            continue

        coords1 = t1[["x0", "top"]].to_numpy()
        coords2 = t2[["x0", "top"]].to_numpy()
        dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
        _, col_ind = linear_sum_assignment(dist)

        unmatched = set(range(len(t2))) - set(col_ind)
        df_unmatched = t2.iloc[list(unmatched)].copy()
        df_unmatched["combined_score"] = np.nan
        results.append(df_unmatched)

    df_final_group2 = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    df_final_group1 = pd.concat([df2[df2["text"] == txt] for txt in sorted(group1)], ignore_index=True) if group1 else pd.DataFrame()

    df_final_group1["group"] = "group1"
    df_final_group2["group"] = "group2"

    all_char_highlight_df = pd.concat([df_final_group2, df_final_group1], ignore_index=True)

    if not all_char_highlight_df.empty:
        out_pdf = pdf2_path.replace(".pdf", "_highlighted.pdf") if output_path is None else output_path
        highlight_chars_on_pdf(pdf2_path, out_pdf, all_char_highlight_df)

    group_counts = all_char_highlight_df["group"].value_counts().to_dict() if not all_char_highlight_df.empty else {}
    return len(all_char_highlight_df), {}, group_counts


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    uid = str(uuid.uuid4())
    return templates.TemplateResponse("upload_form.html", {"request": request, "uid": uid})


@app.post("/compare")
async def compare_pdfs(request: Request, file1: UploadFile, file2: UploadFile, uid: str = Form(...)):
    user_folder = os.path.join("uploads", uid)
    os.makedirs(user_folder, exist_ok=True)

    file1_path = os.path.join(user_folder, "file1.pdf")
    file2_path = os.path.join(user_folder, "file2.pdf")
    result_path = os.path.join(user_folder, "result_difference.pdf")
    result_reverse_path = os.path.join(user_folder, "result_reverse_difference.pdf")

    with open(file1_path, "wb") as f:
        shutil.copyfileobj(file1.file, f)
    with open(file2_path, "wb") as f:
        shutil.copyfileobj(file2.file, f)

    total_changes, detail, group_counts = await run_in_threadpool(lambda: process_pdfs(file1_path, file2_path, result_path))
    total_changes_rev, detail_rev, group_counts_rev = await run_in_threadpool(lambda: process_pdfs(file2_path, file1_path, result_reverse_path))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "uid": uid,
        "total_changes": total_changes,
        "group_counts": group_counts,
        "total_changes_reverse": total_changes_rev,
        "group_counts_reverse": group_counts_rev,
        "old_pdf": f"/uploads/{uid}/file1.pdf",
        "new_pdf": f"/uploads/{uid}/file2.pdf",
        "original_filename1": file1.filename,
        "original_filename2": file2.filename,
    })


@app.get("/download/{uuid}", response_class=FileResponse)
def download_result(uuid: str):
    result_path = os.path.join("uploads", uuid, "result_difference.pdf")
    if not os.path.exists(result_path):
        return HTMLResponse(content="No changes detected", status_code=404)
    return FileResponse(result_path, filename="result_difference.pdf")


@app.get("/view/{uuid}")
async def view_result_pdf(uuid: str):
    result_path = os.path.join("uploads", uuid, "result_difference.pdf")
    if not os.path.exists(result_path):
        return HTMLResponse(content="No changes detected", status_code=404)
    return StreamingResponse(open(result_path, "rb"), media_type="application/pdf")


@app.get("/download_reverse/{uuid}", response_class=FileResponse)
def download_reverse_result(uuid: str):
    result_path = os.path.join("uploads", uuid, "result_reverse_difference.pdf")
    if not os.path.exists(result_path):
        return HTMLResponse(content="No changes detected", status_code=404)
    return FileResponse(result_path, filename="result_reverse_difference.pdf")


@app.get("/view_reverse/{uuid}")
async def view_reverse_result_pdf(uuid: str):
    result_path = os.path.join("uploads", uuid, "result_reverse_difference.pdf")
    if not os.path.exists(result_path):
        return HTMLResponse(content="No changes detected", status_code=404)
    return StreamingResponse(open(result_path, "rb"), media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5004, reload=True)
