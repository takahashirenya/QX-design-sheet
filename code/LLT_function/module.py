from dataclasses import dataclass
import numpy as np

@dataclass
class WingType:
    span_div: int  # 分割数
    dy: float  # パネル幅
    hspar: float  # 桁位置
    hac: float  # 空力中心位置
    iteration: int = 0
    dynamic_pressure: float = 0.0
    # input_data_to_type で他の属性が埋まる

@dataclass
class FlightState:
    Vair: float  # 速度
    rho: float  # 密度
    mu: float  # 動粘性係数 or 粘性係数（コードは mu を分母に使うので単位系を合わせて）
    alpha: float  # 迎角[deg]
    beta: float  # 横滑り角[deg]
    p: float  # ロール角速度[rad/s]
    r: float  # ヨー角速度[rad/s]
    dh: float  # cgとacの差分で使う補正
    hE: float = 0.0  # 鏡像高さ（省略可）

@dataclass
class Specifications:
    span_div: int
    dy: float
    hspar: float
    hac: float
    dynamic_pressure: float

    S: float = 0.0
    chord_mac: float = 0.0
    y_: float = 0.0
    b: float = 0.0
    AR: float = 0.0
    Cla: float = 0.0

    Drag_parasite: float = 0.0
    L_roll: float = 0.0
    M_pitch: float = 0.0
    N_yaw: float = 0.0

    Lift: float = 0.0
    Drag_induced: float = 0.0
    Drag: float = 0.0

    CL: float = 0.0
    Cdp: float = 0.0
    CDi: float = 0.0
    CD: float = 0.0
    Cm_ac: float = 0.0
    Cm_cg: float = 0.0
    e: float = 0.0
    aw: float = 0.0

    Cyb: float = 0.0
    Cyp: float = 0.0
    Cyr: float = 0.0
    Clb: float = 0.0
    Clp: float = 0.0
    Clr: float = 0.0
    Cnb: float = 0.0
    Cnp: float = 0.0
    Cnr: float = 0.0

