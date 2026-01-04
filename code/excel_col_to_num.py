def excel_col_to_num(col: str) -> int:
    """
    Excel列記号を列番号（1始まり）に変換する。
    例:
        A  -> 1
        Z  -> 26
        AA -> 27
        AB -> 28
        XFD -> 16384
    """
    if not isinstance(col, str) or not col:
        raise ValueError("列指定は空でない文字列である必要があります")

    col = col.strip().upper()
    if not col.isalpha():
        raise ValueError(f"不正な列指定: {col}")

    num = 0
    for c in col:
        num = num * 26 + (ord(c) - ord('A') + 1)
    return num


# 例: excel_col_to_num("AB") -> 28