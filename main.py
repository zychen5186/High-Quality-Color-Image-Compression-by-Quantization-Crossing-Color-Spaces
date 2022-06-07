import math as m
import numpy as np
import tool
import cv2

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
for a in range(1, 28):
    img_name = str(a) + ".png"
    img = cv2.imread('dataset/' + img_name)
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]

    
    img = cv2.merge([B, G, R])

    img = np.uint8(img)
    h, w = np.shape(R)
    N = h*w

    Y, Cb, Cr = tool.RGBtoYCbCr_BT709(img)

# =============================================================================
# ### Dataset used as example in the paper
# ### However the YCbCr value doesn't seems to match the value after performing color conversion on the RGB value

#     h,w = 4,4
#     N = 16
# 
#     R = np.array([[227, 226, 221, 235], [228, 223, 223, 22],
#                   [227, 221, 221, 205], [225, 222, 218, 203]])
#     G = np.array([[118, 111, 109, 146], [118, 114, 102, 109],
#                   [119, 113, 101, 95], [127, 119, 108, 94]])
#     B = np.array([[109, 106, 97, 101], [100, 98, 96, 88],
#                   [91, 97, 93, 93], [101, 117, 101, 104]])
#     img = cv2.merge([B, G, R])
#     img = np.uint8(img)
#     
#     Y, Cb, Cr = tool.RGBtoYCbCr_BT709(img)
# =============================================================================

# =============================================================================
# ### Calculate W = (AH)T(AH) in order to get W = (D)T(D)
#     
#     C = np.zeros(shape=(N, N))
#     for i in range(N):
#         C[0][i] = 1 / m.sqrt(N)
#     for i in range(1, N):
#         for j in range(N):
#             C[i][j] = m.sqrt(2/N) * m.cos(((2*j + 1) * i * m.pi) / (2*N))
# 
#     C_inv = np.linalg.inv(C)
#     C_inv_trans = C_inv.T
#     C_sqr = np.dot(C_inv_trans, C_inv)  # (C_inv)T(C_inv) 結果為一單位矩陣
# 
#     W00 = C_sqr * 3
#     W01 = C_sqr * 1.6681685
#     W02 = C_sqr * 1.10678574
#     W11 = C_sqr * 3.40817753865
#     W12 = C_sqr * 0.08762714254
#     W22 = C_sqr * 2.69913249249
#     W0 = np.hstack((W00, W01, W02))
#     W1 = np.hstack((W01, W11, W12))
#     W2 = np.hstack((W02, W12, W22))
#     W = np.vstack((W0, W1, W2))
#     D = np.linalg.cholesky(W).T  # cholesky gets D, (D)T upper triangular matrix
# 
# =============================================================================
    # ----------------
    # Matrix D after applying Cholesky decomposition on Matrix W
    # ----------------
    D = [1.73205081, 0.96311753, 0.63900304,
         1.5749864, -0.33511902, 1.47597522]

    # -----------------
    # YCbCr -> XYCbCr : DCT
    # -----------------
    XY = np.reshape(cv2.dct(Y), (N, 1))
    XCb = np.reshape(cv2.dct(Cb), (N, 1))
    XCr = np.reshape(cv2.dct(Cr), (N, 1))

    # ------------------------------------------
    # YCbCr DCT coefficient after Quantization
    # XYCbCr -> hat_XYCbCr
    # ------------------------------------------
    f = open("output.txt", "a")
    
    #%%
    """
    noraml Quantization
    """
    hat_XY = np.zeros(shape=(N, 1))
    hat_XCb = np.zeros(shape=(N, 1))
    hat_XCr = np.zeros(shape=(N, 1))
    for i in range(N):
        hat_XY[i, 0] = np.round(XY[i]/10) * 10
        hat_XCb[i, 0] = np.round(XCb[i]/10) * 10
        hat_XCr[i, 0] = np.round(XCr[i]/10) * 10

    # -----------------
    # hat_XYCbCr -> hat_YCbCr : IDCT
    # -----------------
    hat_Y = cv2.idct(np.reshape(hat_XY, (h, w)))
    hat_Cb = cv2.idct(np.reshape(hat_XCb, (h, w)))
    hat_Cr = cv2.idct(np.reshape(hat_XCr, (h, w)))

    new_img = cv2.merge([hat_Y, hat_Cr, hat_Cb])
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
    
    #%%
    """
    SSEDQ
    """
    SSEDQ_hat_XY = np.zeros(shape=(N, 1))
    SSEDQ_hat_XCb = np.zeros(shape=(N, 1))
    SSEDQ_hat_XCr = np.zeros(shape=(N, 1))
    for i in range(N):
        SSEDQ_hat_XY[i, 0], SSEDQ_hat_XCb[i, 0] = tool.SSE_Directed_Quantization(
            XY, XCb, XCr, D, i)
        SSEDQ_hat_XCr[i, 0] = np.round(XCr[i]/10) * 10

    
    # -----------------
    # hat_XYCbCr -> hat_YCbCr : IDCT
    # -----------------
    SSEDQ_hat_Y = cv2.idct(np.reshape(SSEDQ_hat_XY, (h, w)))
    SSEDQ_hat_Cb = cv2.idct(np.reshape(SSEDQ_hat_XCb, (h, w)))
    SSEDQ_hat_Cr = cv2.idct(np.reshape(SSEDQ_hat_XCr, (h, w)))

    SSEDQ_new_img = cv2.merge([SSEDQ_hat_Y, SSEDQ_hat_Cr, SSEDQ_hat_Cb])
    SSEDQ_new_R, SSEDQ_new_G, SSEDQ_new_B = tool.YCbCrtoRGB_BT709(SSEDQ_new_img)

    for i in range(h):
        for j in range(w):
            SSEDQ_new_R[i][j] = tool.clip(SSEDQ_new_R[i][j], 255, 0)
            SSEDQ_new_G[i][j] = tool.clip(SSEDQ_new_G[i][j], 255, 0)
            SSEDQ_new_B[i][j] = tool.clip(SSEDQ_new_B[i][j], 255, 0)
    SSEDQ_new_R = np.uint8(SSEDQ_new_R)
    SSEDQ_new_G = np.uint8(SSEDQ_new_G)
    SSEDQ_new_B = np.uint8(SSEDQ_new_B)
    SSEDQ_new_img = cv2.merge([SSEDQ_new_B, SSEDQ_new_G, SSEDQ_new_R])
    
    SSEDQ_RGB_SSE = tool.SSE(R, SSEDQ_new_R) + tool.SSE(G, SSEDQ_new_G) + tool.SSE(B, SSEDQ_new_B)
    SSEDQ_psnr = cv2.PSNR(img, SSEDQ_new_img)
    f.write("SSEDQ_psnr = " + str(SSEDQ_psnr) + "\n")
    f.write("SSEDQ_SSE = " + str(SSEDQ_RGB_SSE) + "\n")
    print("psnr = ", SSEDQ_psnr)
    cv2.imwrite('output_image/' + str(a) + '_SSEDQ_output_test.jpg', SSEDQ_new_img)


f.close()
print()
