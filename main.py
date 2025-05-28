import cv2
import numpy as np
import os

def mask_img(img, mask):
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
    else:
        b, g, r, _ = cv2.split(img)
    b = np.multiply(b, mask).astype(img.dtype)
    g = np.multiply(g, mask).astype(img.dtype)
    r = np.multiply(r, mask).astype(img.dtype)
    return b, g, r

def resiz_img(img, height_size):
    height, width = img.shape[0], img.shape[1]
    scale = height/height_size
    width_size = int(width/scale)
    return cv2.resize(img, (width_size, height_size))

def instx_img_bkg(x, y, height_size, height_size2, rahmen, img, blur_size = 105):
    ## part1: add border
    resize_img = img = resiz_img(img, height_size)
    mask = rahmen[y:y+resize_img.shape[0], x:x+resize_img.shape[1], -1]

    b1, g1, r1 = mask_img(resize_img, np.where(mask>0, 0, 1))
    b2, g2, r2 = mask_img(rahmen[y:y+resize_img.shape[0], x:x+resize_img.shape[1]], np.where(mask>0, 1, 0))

    resize_img = cv2.merge((b1+b2, g1+g2, r1+r2, np.ones(b1.shape).astype(img.dtype)*255))
    rahmen[y:y+resize_img.shape[0], x:x+resize_img.shape[1], :] = resize_img            # org img add border


    ## add blur background img
    blur_image = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    if blur_image.shape[2] == 3:
        b, g, r = cv2.split(blur_image)
        blur_image = cv2.merge((b, g, r, np.ones(b.shape).astype(mask.dtype)*255))

    rahmen = resiz_img(rahmen, height_size2)

    h1, w1 = blur_image.shape[0]/2, blur_image.shape[1]/2
    h2, w2 = rahmen.shape[0]/2, rahmen.shape[1]/2

    b1, g1, r1 = mask_img(blur_image[int(h1-h2):int(h1-h2)+rahmen.shape[0], int(w1-w2):int(w1-w2)+rahmen.shape[1], :], np.where(rahmen[:, :, -1] > 0, 0, 1))
    b2, g2, r2 = mask_img(rahmen, np.where(rahmen[:, :, -1] > 0, 1, 0))

    rahmen = cv2.merge((b1+b2, g1+g2, r1+r2, np.ones(b1.shape).astype(img.dtype)*255))
    blur_image[int(h1-h2):int(h1-h2)+rahmen.shape[0], int(w1-w2):int(w1-w2)+rahmen.shape[1], :] = rahmen[:,:,:]
    return blur_image

if __name__ == '__main__':
    x, y = 392, 740             # 添加边框时添加图片在右上角的起始坐标(该数据与边框素材一一对应)
    height_size = 5900          # 添加边框时图片的高度
    height_size2 = 4200         # 添加模糊背景时带边框的图片的高度
    rahmen_dir = './1.png'      # 边框素材
    input_dir = './input'       # 要合成图片的文件夹
    output_dir = './output'     # 合成后保存图片的文件夹

    img_dir_list = os.listdir(input_dir)
    for img_dir in img_dir_list:
        rahmen = cv2.imread(rahmen_dir, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(input_dir + '/' + img_dir)
        if img.shape[0] < img.shape[1]:
            img = np.rot90(img, k = -1)
            cv2.imwrite(output_dir + '/' + img_dir, np.rot90(instx_img_bkg(x, y, height_size, height_size2, rahmen, img), k=1))
        else:
            cv2.imwrite(output_dir + '/' + img_dir, instx_img_bkg(x, y, height_size, height_size2, rahmen, img))