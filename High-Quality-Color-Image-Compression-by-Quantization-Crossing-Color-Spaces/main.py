import math as m
import numpy as np
import tool
import cv2
import xlwt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
for a in range(1, 2):
    img_name = str(a) + ".png"
    img = cv2.imread('dataset/' + img_name)
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]

    # R = np.array([[227, 226, 221, 235], [228, 223, 223, 22],
    #               [227, 221, 221, 205], [225, 222, 218, 203]])
    # G = np.array([[118, 111, 109, 146], [118, 114, 102, 109],
    #               [119, 113, 101, 95], [127, 119, 108, 94]])
    # B = np.array([[109, 106, 97, 101], [100, 98, 96, 88],
    #               [91, 97, 93, 93], [101, 117, 101, 104]])
    # img = cv2.merge([B, G, R])

    img = np.uint8(img)
    h, w = np.shape(R)
    N = h*w

    Y, Cb, Cr = tool.RGBtoYCbCr_BT709(img)

    # DY = np.array([157, 151, 148, 178, 156, 152, 143, 148,
    #                156, 151, 142, 134, 162, 157, 147, 134])
    # DY = np.reshape(DY, (4, 4))

    # DCb = [111, 112, 109, 95, 106, 108, 111, 104,
    #        102, 108, 110, 114, 104, 115, 112, 121]
    # DCb = np.reshape(DCb, (4, 4))

    # DCr = [183, 186, 185, 175, 184, 183, 189, 187,
    #        183, 183, 188, 183, 178, 180, 183, 182]
    # DCr = np.reshape(DCr, (4, 4))

    # -------------------
    # 計算W = (AH)T(AH) 找 W = (D)T(D)
    # -------------------
    # C = np.zeros(shape=(N, N))
    # for i in range(N):
    #     C[0][i] = 1 / m.sqrt(N)
    # for i in range(1, N):
    #     for j in range(N):
    #         C[i][j] = m.sqrt(2/N) * m.cos(((2*j + 1) * i * m.pi) / (2*N))

    # C_inv = np.linalg.inv(C)
    # C_inv_trans = C_inv.T
    # C_sqr = np.dot(C_inv_trans, C_inv)  # (C_inv)T(C_inv) 結果為一單位矩陣

    # W00 = C_sqr * 3
    # W01 = C_sqr * 1.6681685
    # W02 = C_sqr * 1.10678574
    # W11 = C_sqr * 3.40817753865
    # W12 = C_sqr * 0.08762714254
    # W22 = C_sqr * 2.69913249249
    # W0 = np.hstack((W00, W01, W02))
    # W1 = np.hstack((W01, W11, W12))
    # W2 = np.hstack((W02, W12, W22))
    # W = np.vstack((W0, W1, W2))
    # D = np.linalg.cholesky(W).T  # cholesky gets D, (D)T upper triangular matrix

    # ----------------
    # 簡易版D矩陣
    # ----------------
    D = [1.73205081, 0.96311753, 0.63900304,
         1.5749864, -0.33511902, 1.47597522]

    # -----------------
    # YCbCr -> XYCbCr : 做DCT
    # -----------------
    XY = np.reshape(cv2.dct(Y), (N, 1))
    XCb = np.reshape(cv2.dct(Cb), (N, 1))
    XCr = np.reshape(cv2.dct(Cr), (N, 1))

    # XY = [604.00, 16.28, 8.50, -5.12, 13.27, -24.62, 16.00, -
    #       6.70, 13.00, -8.19, 5.50, -1.48, -0.63, -3.20, 2.03, 0.12]
    # XY = np.reshape(XY, (16, 1))
    # XCb = [435.50, -3.46, -7.00, -1.81, -8.84, 16.49, -5.19,
    #        7.04, 4.00, 4.43, -1.50, -2.38, -1.75, 0.04, 0.53, -0.49]
    # XCb = np.reshape(XCb, (16, 1))
    # XCr = [733.00, -1.43, -5.50, 4.38, 2.77, 5.22, -2.73,
    #        0.37, -7.00, 3.50, -2.50, -1.99, -1.15, 3.87, -2.66, 0.28]
    # XCr = np.reshape(XCr, (16, 1))

    # ------------------------------------------
    # YCbCr DCT coefficient after Quantization
    # XYCbCr -> hat_XYCbCr
    # ------------------------------------------
    f = open("output.txt", "a")
    """
    純粹量化
    """
    hat_XY = np.zeros(shape=(N, 1))
    hat_XCb = np.zeros(shape=(N, 1))
    hat_XCr = np.zeros(shape=(N, 1))
    for i in range(N):
        hat_XY[i, 0] = np.round(XY[i]/10) * 10
        hat_XCb[i, 0] = np.round(XCb[i]/10) * 10
        hat_XCr[i, 0] = np.round(XCr[i]/10) * 10

    # -----------------
    # hat_XYCbCr -> hat_YCbCr : 做IDCT
    # -----------------
    hat_Y = cv2.idct(np.reshape(hat_XY, (h, w)))
    hat_Cb = cv2.idct(np.reshape(hat_XCb, (h, w)))
    hat_Cr = cv2.idct(np.reshape(hat_XCr, (h, w)))

    new_img = cv2.merge([hat_Y, hat_Cr, hat_Cb])
    # new_img = new_img.astype(np.float32)
    new_R, new_G, new_B = tool.YCbCrtoRGB_BT709(new_img)
    for i in range(h):
        for j in range(w):
            new_R[i][j] = tool.clip(new_R[i][j], 255, 0)
            new_G[i][j] = tool.clip(new_G[i][j], 255, 0)
            new_B[i][j] = tool.clip(new_B[i][j], 255, 0)
    new_R = np.uint8(new_R)
    new_G = np.uint8(new_G)
    new_B = np.uint8(new_B)
    new_img = cv2.merge([new_B, new_G, new_R])

    RGB_SSE = tool.SSE(R, new_R) + tool.SSE(G, new_G) + tool.SSE(B, new_B)
    psnr = cv2.PSNR(img, new_img)
    f.write(img_name + " :\n")
    f.write("old_psnr = " + str(psnr) + "\n")
    f.write("old_SSE = " + str(RGB_SSE) + "\n")
    print("psnr = ", psnr)
    cv2.imwrite('output_image/' + str(a) + '_old_output_test.jpg', new_img)
    """
    SSEDQ
    """
    hat_XY = np.zeros(shape=(N, 1))
    hat_XCb = np.zeros(shape=(N, 1))
    hat_XCr = np.zeros(shape=(N, 1))
    for i in range(N):
        hat_XY[i, 0], hat_XCb[i, 0] = tool.SSE_Directed_Quantization(
            XY, XCb, XCr, D, i)
        hat_XCr[i, 0] = np.round(XCr[i]/10) * 10

    # -----------------
    # hat_XYCbCr -> hat_YCbCr : 做IDCT
    # -----------------
    hat_Y = cv2.idct(np.reshape(hat_XY, (h, w)))
    hat_Cb = cv2.idct(np.reshape(hat_XCb, (h, w)))
    hat_Cr = cv2.idct(np.reshape(hat_XCr, (h, w)))

    new_img = cv2.merge([hat_Y, hat_Cr, hat_Cb])
    # new_img = new_img.astype(np.float32)
    new_R, new_G, new_B = tool.YCbCrtoRGB_BT709(new_img)
    for i in range(h):
        for j in range(w):
            new_R[i][j] = tool.clip(new_R[i][j], 255, 0)
            new_G[i][j] = tool.clip(new_G[i][j], 255, 0)
            new_B[i][j] = tool.clip(new_B[i][j], 255, 0)
    new_R = np.uint8(new_R)
    new_G = np.uint8(new_G)
    new_B = np.uint8(new_B)
    new_img = cv2.merge([new_B, new_G, new_R])

    RGB_SSE = tool.SSE(R, new_R) + tool.SSE(G, new_G) + tool.SSE(B, new_B)
    psnr = cv2.PSNR(img, new_img)
    f.write("SSEDQ_psnr = " + str(psnr) + "\n")
    f.write("SSEDQ_SSE = " + str(RGB_SSE) + "\n")
    print("psnr = ", psnr)
    cv2.imwrite('output_image/' + str(a) + '_SSEDQ_output_test.jpg', new_img)

    # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    # for i in range(h):
    #     for j in range(w):
    #         sheet.write(i*w + j, 0, Y[i][j])
    #         sheet.write(i*w + j, 3, Cb[i][j])
    #         sheet.write(i*w + j, 6, Cr[i][j])
    #         sheet.write(i*w + j, 1, hat_Y[i][j])
    #         sheet.write(i*w + j, 4, hat_Cb[i][j])
    #         sheet.write(i*w + j, 7, hat_Cr[i][j])
    # book.save('test.xls')

f.close()
print()
