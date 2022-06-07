import math as m
import numpy as np


def Q(x):
    return np.round(x/10) * 10


""""
def our_method(Y, Cb, Cr, D, k):
    delta_Cr = Cr[k] - Q(Cr[k])

    compen_Cb = (D[4] / D[3]) * delta_Cr
    Q_Cb = Q(Cb[k])
    new_Cb = Q_Cb + compen_Cb

    delta_Cb = Cb[k] - Q_Cb  # 用舊的Cb值來減
    compen_Y = (D[1]*delta_Cb + D[2]*delta_Cr) / D[0]
    new_Y = Q(Y[k]) + compen_Y

    return new_Y, new_Cb
"""


def SSE_Directed_Quantization(Y, Cb, Cr, D, k):
    delta_Cr = Cr[k] - Q(Cr[k])

    compen_Cb = (D[4] / D[3]) * delta_Cr
    new_Cb = Cb[k] + compen_Cb
    Q_Cb = Q(new_Cb)

    delta_Cb = Cb[k] - Q_Cb  # 被減的是舊的Cb
    compen_Y = (D[1]*delta_Cb + D[2]*delta_Cr) / D[0]
    Q_Y = Q(Y[k] + compen_Y)

    return Q_Y, Q_Cb


def SSEDQ_Subsampling(Y, Cb, Cr, D, k):
    delta_Cr = Cr[k] - Q(Cr[k])

    compen_Cb = (D[4] / D[3]) * delta_Cr
    new_Cb = Cb[k] + compen_Cb
    Q_Cb = Q(new_Cb)

    delta_Cb = Cb[k] - Q_Cb  # 被減的是舊的Cb
    compen_Y = (D[1]*delta_Cb + D[2]*delta_Cr) / D[0]
    Q_Y = Q(Y[k] + compen_Y)

    return Q_Y, Q_Cb


def SSE(x, x_bar):
    return ((x-x_bar)**2).sum()


def RGBtoYCbCr_BT709(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = (B - Y) / 1.8556
    Cr = (R - Y) / 1.5748
    return Y, Cb, Cr


def YCbCrtoRGB_BT709(img):
    Y = img[:, :, 0]
    Cb = img[:, :, 2]
    Cr = img[:, :, 1]
    R = Y + 1.5748 * Cr
    G = Y - (0.2126 * 1.5748 / 0.7152) * Cr - (0.0722 * 1.8556 / 0.7152) * Cb
    B = Y + 1.8556 * Cb
    return R, G, B

# ------------------
# ITU Rec. BT. 709
# ------------------


m1 = 0.1593017578125
m2 = 78.84375
c2 = 18.8515625
c3 = 18.6875
c1 = 0.8359375


def clip(input_val, max_val, min_val):
    if input_val > max_val:
        return max_val
    elif input_val < min_val:
        return min_val
    else:
        return input_val


def EOTF(N):
    fraction = pow(N, 1/m2)
    fraction -= c1
    fraction = max(fraction, 0)
    denominator = pow(N, 1/m2)
    denominator = c3 * denominator
    denominator = c2 - denominator
    return_val = fraction/denominator
    return_val = pow(return_val, 1 / m1)
    return return_val


def OETF(L):
    fraction = c2 * pow(L, m1)
    fraction += c1
    denominator = c3 * pow(L, m1)
    denominator += 1
    return_val = fraction / denominator
    return_val = pow(return_val, m2)
    return return_val


def RGB_to_RpGpBp(R, G, B):
    # ----------------
    # RGB -> R'G'B'
    # ----------------
    h, w = np.shape(R)
    Rp = np.zeros(R.shape)
    Gp = np.zeros(G.shape)
    Bp = np.zeros(B.shape)
    for i in range(h):
        for j in range(w):
            Rp[i, j] = OETF(max(0, min(R[i, j]/10000, 1)))
            Gp[i, j] = OETF(max(0, min(G[i, j]/10000, 1)))
            Bp[i, j] = OETF(max(0, min(B[i, j]/10000, 1)))
    return Rp, Gp, Bp


def RpGpBp_to_RGB(Rp, Gp, Bp):
    # ----------------
    # R'G'B' -> RGB
    # ----------------
    h, w = np.shape(Rp)
    R = np.zeros(Rp.shape)
    G = np.zeros(Gp.shape)
    B = np.zeros(Bp.shape)
    for i in range(h):
        for j in range(w):
            R[i, j] = 10000 * EOTF(Rp[i, j])
            G[i, j] = 10000 * EOTF(Gp[i, j])
            B[i, j] = 10000 * EOTF(Bp[i, j])
    return R, G, B


def RpGpBp_to_YCbCr(Rp, Gp, Bp):
    # ----------------
    # R'G'B' -> YCbCr
    # ----------------
    Y = 0.2126 * Rp + 0.7152 * Gp + 0.0722 * Bp
    Cb = -0.1146 * Rp - 0.3854 * Gp + 0.5 * Bp
    Cr = 0.5 * Rp - 0.4542 * Gp - 0.0458 * Bp
    return Y, Cb, Cr


def YCbCr_to_RpGpBp(Y, Cb, Cr):
    # ----------------
    # YCbCr -> R'G'B'
    # ----------------
    h, w = np.shape(Y)
    Rp = np.zeros(Y.shape)
    Gp = np.zeros(Cb.shape)
    Bp = np.zeros(Cr.shape)
    for i in range(h):
        for j in range(w):
            Rp[i, j] = Y[i, j] + 1.57480 * Cr[i, j]
            Gp[i, j] = Y[i, j] - 0.18733 * Cb[i, j] - 0.46812 * Cr[i, j]
            Bp[i, j] = Y[i, j] + 1.85560 * Cb[i, j]
    return Rp, Gp, Bp


def YCbCr_to_DYDCbDCr(Y, Cb, Cr, bit_depth=10):

    # --------------------------------
    # Y'CbCr -> DY'DCbDCr
    # 此function對單一pixel進行動作
    # --------------------------------
    h, w = np.shape(Y)
    DY = np.zeros(Y.shape)
    DCb = np.zeros(Cb.shape)
    DCr = np.zeros(Cr.shape)
    for i in range(h):
        for j in range(w):
            mul_val = pow(2, (bit_depth-8))
            DY[i, j] = clip(np.round(mul_val * (219 * Y[i, j] + 16)),
                            pow(2, bit_depth), 0)
            DCb[i, j] = clip(np.round(mul_val * (224 * Cb[i, j] + 128)),
                             pow(2, bit_depth), 0)
            DCr[i, j] = clip(np.round(mul_val * (224 * Cr[i, j] + 128)),
                             pow(2, bit_depth), 0)

    return DY, DCb, DCr


def DYDCbDCr_to_YCbCr(DY, DCb, DCr, bit_depth=10):

    # --------------------------------
    # DY'DCbDCr -> Y'CbCr
    # 此function對單一pixel進行動作
    # --------------------------------
    h, w = np.shape(DY)
    Y = np.zeros(DY.shape)
    Cb = np.zeros(DCb.shape)
    Cr = np.zeros(DCr.shape)
    for i in range(h):
        for j in range(w):
            div_val = pow(2, (bit_depth-8))
            Y[i, j] = clip((DY[i, j]/div_val - 16)/219, 1, 0)
            Cb[i, j] = clip((DCb[i, j]/div_val - 128)/224, 0.5, -0.5)
            Cr[i, j] = clip((DCr[i, j]/div_val - 128)/224, 0.5, -0.5)

    return Y, Cb, Cr
