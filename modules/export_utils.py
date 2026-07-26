impo** io
from datetime import datetime**import pandas as pd
from reportla**lib.pagesizes import A4
from repo**lab.pdfgen import canvas

from mo**les.config import HISTORY_FILE, S**NATURE


def excel_bytes(sheets: **ct[str, pd.DataFrame]) -> bytes:
**  output = io.BytesIO()

    with**