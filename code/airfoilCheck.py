import xlwings as xw
import os

def airfoilCheck():
    wb = xw.Book.caller()
    sheet = wb.sheets.active  

    # S5 からファイル名を取得
    file_name = sheet.range("S5").value
    if not file_name:
        sheet.range("Q8").value = "S5にファイル名を入力してください"
        return

    # Airfoilフォルダのパス
    base_dir = os.path.dirname(wb.fullname)
    airfoil_dir = os.path.join(base_dir, "Airfoil")
    dat_path = os.path.join(airfoil_dir, f"{file_name}")

    # ファイル存在チェック
    if not os.path.exists(dat_path):
        sheet.range("Q8").value = f"{file_name} が見つかりません"
        return

    # datファイル読み込み
    x_vals, y_vals = [], []
    with open(dat_path, "r") as f:
        lines = f.readlines()[1:]  # 1行目は翼型名なのでスキップ
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue

    sheet.range("N11").value = ["x", "y"]
    sheet.range("N12").value = list(zip(x_vals, y_vals))