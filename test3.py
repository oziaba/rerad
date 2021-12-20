#!/usr/bin/python3
from multiprocessing import Process, Manager, Value

import os
import math
import queue
import numpy as np
from skimage.transform import resize
import tensorflow as tf
import keras
import sys
import cv2
from cv2 import imread, createCLAHE
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from keras.models import model_from_json
from array import *
from pydicom.uid import ExplicitVRLittleEndian
import gc
from pydicom.filereader import dcmread
from highdicom.sr.sop import ComprehensiveSR
from pydicom.uid import generate_uid
from keras.engine import base_layer_v1
import multiprocessing
import time
# Функция сегментации изображений 
# На вход подаются (исходное изображение, название для сегментации, загружаемая модель)
def segment_image(img, file, loaded_model):
    # Подготовка изображения для сегментации
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)[:,:,0]
    img = cv2.resize(img,(1024,1024))
    # Сегментация изображени
    segm_ret = loaded_model.predict(image_to_train(img), verbose=0)
    # Создание маски как изображение
    img = cv2.bitwise_and(img, img, mask=train_to_image(segm_ret))
    cv2.imwrite(os.path.join(way + "data/", file % "dicom"), img)
    return img

# Функция преобразования изображения для обучения 
def image_to_train(img):
    npy = img / 255
    npy = np.reshape(npy, npy.shape + (1,))
    npy = np.reshape(npy,(1,) + npy.shape)
    return npy

# Функция преобразования изображения для обучения (Перевод в 8 бит)
def train_to_image(npy):
    img = (npy[0,:, :, 0] * 255.).astype(np.uint8)
    return img

# Функция загрузки модели сегментации
def loadmodel(ModelName):
    # Загрузка модели сегментации
    print("---loading a " + ModelName + " segmentation model---" )
    # указать путь к json файлу
    json_file = open(resource_path(ModelName+".json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель
    loaded_model = model_from_json(loaded_model_json)
    # Загружаем сохраненные веса в модели
    loaded_model.load_weights(resource_path(ModelName+'.h5'))
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
    print("---" + ModelName + " segmentaton is complete---")
    return loaded_model

# Функция для создания наложения трех масок в исходное изображение (Маска1 - Синяя, Маска2 - Красная, Маска3 - Розовая)
def add_colored_mask3(image, mask_image, mask_image1, mask_image2):    
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)    
    mask_image_gray1 = cv2.cvtColor(mask_image1, cv2.COLOR_BGR2GRAY)    
    mask_image_gray2 = cv2.cvtColor(mask_image2, cv2.COLOR_BGR2GRAY)        
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)    
    mask1 = cv2.bitwise_and(mask_image1, mask_image1, mask=mask_image_gray1)    
    mask2 = cv2.bitwise_and(mask_image2, mask_image2, mask=mask_image_gray2)    
    mask_coord = np.where(mask!=[0,0,0])    
    mask_coord1 = np.where(mask1!=[0,0,0])    
    mask_coord2 = np.where(mask2!=[0,0,0])
    mask[mask_coord[0],mask_coord[1],:]=[0,0,255]    
    mask1[mask_coord1[0],mask_coord1[1],:]=[255,0,0]    
    mask2[mask_coord2[0],mask_coord2[1],:]=[255,0,255]
    ret = cv2.addWeighted(image, 1.0, mask, 0.2, 0)       
    ret = cv2.addWeighted(ret, 1.0, mask1, 0.4, 0)      
    ret = cv2.addWeighted(ret, 1.0, mask2, 0.2, 0)  
    return ret

# Высота Сердца
# Крайняя верхняя точка сердца
# Проходим по Х сверху - вниз находим первую точку со значением больше нуля 
def kr_verx(heartarray,vi,vj):
    vi.value = 0
    vj.value = 0
    for i in range(3,1020):
        for j in range(3,1020):
            if heartarray[i,j] > 7:
                vi.value = i
                vj.value = j
                break
        else:
            continue
        break
    return vi,vj

# Крайняя нижняя точка сердца
# Проходим по Х снизу - вверх находим первую точку со значением больше нуля 
def kr_nizh(heartarray,ni,nj):
    nj.value = 0
    ni.value = 0
    for i in range(1020,3,-1):
        for j in range(3,1020):
            if heartarray[i,j] > 7:
                ni.value = i
                nj.value = j
                break
        else:
            continue
        break
    return ni,nj

# Ширина сердца
# Крайняя левая точка сердца
# Проходим по У слева - направо находим первую точку со значением больше нуля 
def kr_leva(heartarray,li,lj,vi,ni):
    for j in range(3,1020):
        for i in range(int(vi),int(ni)):
            if heartarray[i,j] > 7:
                li = i
                lj = j
                break
        else:
            continue
        break
    return li,lj

# Крайняя правая точка сердца
# Проходим по У слева - направо находим последнюю точку со значением больше нуля 
def kr_prav(heartarray,pi,pj,vi,ni):
    for j in range(3,1020):
        for i in range(int(vi),int(ni)):
            if heartarray[i,j] > 7:
                pi = i
                pj = j
    return pi,pj

# Высота Легких
# Крайняя верхняя точка легких
# Проходим по Х сверху - вниз находим первую точку со значением больше нуля 
def lungv(predictarray):
    im = 0
    jm = 0
    for i in range(3,1020):
        for j in range(3,1020):
            if predictarray[i,j] > 7:
                im = i
                jm = j
                break
        else:
            continue
        break
    return (im,jm)

# Находится средняя высота между крайними точками ключиц
# Далее находится ближайшая остица, на этой остице ищем верхние и нижние точки 
# По этим точкам вычисляем срединную линию остиц
def ost_andrey2(img, yt1, yt2):
    y = (yt1 + yt2)//2
    kry = 0
    krx = 0
    for i in range(0,y):
        for j in range(3, 1020):
            if img[y + i, j] > 7:
                kry = y + i
                krx = j
                ostnx = krx
                ostny = kry
                ostvx = krx
                ostvy = kry
                for i in range(1,kry):
                    count = 0
                    for j in range(0, 1023):
                        if img[kry + i, j] > 7:
                            ostnx = j
                            ostny = kry + i
                            break
                        else:
                            count += 1
                    if count == 1023:
                        break
                for i in range(1,kry):
                    count = 0
                    for j in range(0, 1023):
                        if img[kry - i, j] > 7:
                            ostvx = j
                            ostvy = kry - i
                            break
                        else:
                            count += 1
                    if count == 1023:
                        break                   
                break
                
            elif img[y - i, j] > 7: 
                kry = y - i
                krx = j
                ostnx = krx
                ostny = kry
                ostvx = krx
                ostvy = kry
                for i in range(1,kry):
                    count = 0
                    for j in range(0, 1023):
                        if img[kry + i, j] > 7:
                            ostnx = j
                            ostny = kry + i
                            break
                        else:
                            count += 1
                    if count == 1023:
                        break
                for i in range(1,kry):
                    count = 0
                    for j in range(0, 1023):
                        if img[kry - i, j] > 7:
                            ostvx = j
                            ostvy = kry - i
                            break
                        else:
                            count += 1
                    if count == 1023:
                        break   
                break
        else:
            continue
        break  
    count1 = 0   
    for jl in range(0,50):
        if img[ostvy, ostvx + jl] < 8:
            ostvx = ostvx + count1//2
            break
        else:
            count1 += 1
    count1 = 0
    for jl in range(0,50):
        if img[ostny, ostnx + jl] < 8:
            ostnx = ostnx + count1//2
            break
        else:
            count1 += 1
    return (ostvx + ostnx) // 2

# Находятся нижние точки "островков безумия". Далее из них находим самые нижние два. 
# Между ними проводим прямую. От этой прямой находим ближайщие точки до ключиц, между этими точками будет срединная линия.
def kr_kl(img):
    x_max_spisok = []
    y_max_spisok = []
    x_spisok = []
    y_spisok = []
    n1 = 10
    n2 = 800
    for j in range(n1, n2, 1):
        kol = 0
        for i in range(n1, n2, 1):
            if img[i, j] == 0:
                kol = kol + 1
            if (img[i, j] > 0 ) and (img[i, j+1] == 0):
                x_spisok.append(i)
                y_spisok.append(j+1)
        if ((kol == (n2 - n1)) and (len(x_spisok) > 0)):
            x_max2 = max(x_spisok)
            y_max2 = y_spisok[x_spisok.index(x_max2)]
            x_max_spisok.append(y_max2)
            y_max_spisok.append(x_max2)

            x_spisok = []
            y_spisok = []

    if (len(x_spisok) > 0):
        x_max2 = max(x_spisok)
        y_max2 = y_spisok[x_spisok.index(x_max2)]
        x_max_spisok.append(y_max2)
        y_max_spisok.append(x_max2)

    y = x_max_spisok
    xx = y_max_spisok

    l = []
    k = []
    z = sorted(xx, reverse = True)
    if z[0] == z[1]:
        yt = y[xx.index(z[0])]
        y.remove(yt)
        xx.remove(z[0])
        l.append(z[0])
        k.append(yt)
        yt = y[xx.index(z[0])]
        l.append(z[0])
        k.append(yt)
        x1 = k[0]
        y1 = l[0]
        x2 = k[1]
        y2 = l[1]
    else:
        xx = y_max_spisok  
        y1 = z[0]
        y2 = z[1]
        x1 = x_max_spisok[y_max_spisok.index(y1)]
        x2 = x_max_spisok[y_max_spisok.index(y2)]
    
    x = (x1 + x2)//2
    ny = y1
    yt1 = y1
    yt2 = y2
    if (ny < y2):
        ny = y2
        yt1 = y2
        yt2 = y1

    if (x1 < x):
        nx1 = x1
        nx2 = x2
        xt1 = x1
        xt2 = x2
    else:
        nx1 = x2
        nx2 = x1
        xt1 = x2
        xt2 = x1
    for j in range(ny, ny - 100, -1):
        for i in range(x, nx1, -1):
            if (img[j, i] == 0) and (img[j, i-1]) > 5:
                if (xt1 < i):
                    xt1 = i
                    yt1 = j
    for j in range(ny, ny-100, -1):
        for i in range(x, nx2, 1):
            if (img[j, i] == 0) and (img[j, i+1]) > 5:
                if (xt2 > i):
                    xt2 = i
                    yt2 = j
    return ((xt1+xt2)//2, xt1, xt2, yt1, yt2, y1, x1)

# Количество ребер по прямой 
# На входе значение У от 3/4 левого ребра, значение смещения, маска ребер видимых в легких
# Подсчитывается по прямой линии количество ребер со смещением  на i
def kolvorebra(kld, i, predictarray3):
    kol = 0
    j = kld + i
    for i in range(3,1020):
        if (predictarray3[i-1,j] > 0) and (predictarray3[i,j] == 0):
            kol = kol + 1
    return(kol)


# Функция для нахождения самой длинной линии по вертикали на легких с левой стороны
# Эта линия показывает точку пересечения легкого с нижней диафрагмой
# На выходе точка пересечения нижней диафрагмы с левой частью легких ih, jh
def lungdlina(predictarray):
    jh = 0
    ih = 0
    maxline = 0
    for j in range(3,1020):
        summ = 0
        for i in range(3,1024):
            if predictarray[i,j] > 0:
                summ = summ + 1
        if maxline <= summ:
            maxline = summ
            jh = j
    for i in range(3,1024):
        if predictarray[i,jh] > 0:
            ih = i
    return ih, jh

# Функция определения крайних точек для нормальных легких. Для этого на высоте ih (точка 
# пересечения нижней диафрагмы с легкими) находятся крайние точки легких. 
# На выходе крайние точки легких
def lungnormwirina(predictarray, ih): 
    for i in range(3,1023):
        if predictarray[ih, i] > 0:
            lunli = i
            lunlj = ih
            break
    for i in range(1020, 0, -1):
        if predictarray[ih, i] > 0:
            lunpi = i
            lunpj = ih
            break
    return lunli, lunlj, lunpi, lunpj

  
# Функции для нахождения касательной к легким, для нахождения крайних точек легкого.
# Бывают случаи, когда на высоте ih крайние точки не определены. Тогда опускается касательная по краю легкого и находится пересечение с ih. 
# Данная функция находит функцию прямой (касательной). Касательная представляет собой, прямую по двум точкам, выбранных на определнных высотах сердца.

def lungnenormwirina1(predictarray, ih, vusota, vi):
# pointi - это высота одной из точек (выбранная опытным путем), 
# она равна половине высоты сердца + верхняя точка сердца.
# pointi-80 вторая точка (выбранная опытным путем).
    pointi = vusota//2 + vi
    for j in range(1020, 0, -1):
        if predictarray[pointi, j] > 0:
            point2i = pointi
            point2j = j
            break
    for j in range(1020, 0, -1):
        if predictarray[pointi-80, j] > 0:
            point1i = pointi-80
            point1j = j
            break
# Это условие когда касательная проходит перпендикулярно к ih, т.е. угловой коэффициент равен 0 (на 0 нельзя делить). 
# Находится Х точки пересечения ih с касательной. 
    if (point1j != point2j):
        a = (point1i - point2i) / (point1j - point2j)
        b = point1i - a * point1j
        c = (ih - b) // a
# Если угловой коэффициент равен 0, Х равен Х-у двух выбранных точек.
    else:
        c = point1j
# На выходе крайняя точка с правой стороны
    return point1i, point1j, int(c)
            
# Функция аналогичная вверхнему только с левой стороны
def lungnenormwirina2(predictarray, ih, vusota, vi):
    pointi = vusota//2 + vi
    for j in range(3,1020):
        if predictarray[pointi-80, j] > 0:
            point1i = pointi-80
            point1j = j
            break
    for j in range(3,1020):
        if predictarray[pointi, j] > 0:
            point2i = pointi
            point2j = j
            break
    if (point1j != point2j):
        a = (point1i - point2i) / (point1j - point2j)
        b = point1i - a * point1j
        c = (ih - b) // a
    else:
        c = point1j
# На выходе крайняя точка с левой стороны
    return point1i, point1j, int(c)
    
# Функция вычисления реберно-диафрагмального синуса
def sinus(lungarray, klg):
    for j in range(1020, 3, -1):
        for i in range(3, klg):
            if lungarray[j, i] > 7:
                sinl1i = i
                sinl1j = j
                break
        else:
            continue
        break

    for j in range(sinl1j-30, 3, -1):
        for i in range(3, klg):
            if lungarray[j, i] > 7:
                sinl11i = i
                sinl11j = j
                break
        else:
            continue
        break

    for j in range(1020, 3, -1):
        for i in range(klg, 3, -1):
            if lungarray[j, i] > 7:
                sinl2i = i
                sinl2j = j
                break
        else:
            continue
        break

    for j in range(sinl1j-30, 3, -1):
        for i in range(klg, 3, -1):
            if lungarray[j, i] > 7:
                sinl22i = i
                sinl22j = j
                break
        else:
            continue
        break

    for j in range(1020, 3, -1):
        for i in range(klg, 1020):
            if lungarray[j, i] > 7:
                sinp1i = i
                sinp1j = j
                break
        else:
            continue
        break

    for j in range(sinp1j-30, 3, -1):
        for i in range(klg, 1020):
            if lungarray[j, i] > 7:
                sinp11i = i
                sinp11j = j
                break
        else:
            continue
        break

    for j in range(1020, 3, -1):
        for i in range(1020, klg, -1):
            if lungarray[j, i] > 7:
                sinp2i = i
                sinp2j = j
                break
        else:
            continue
        break

    for j in range(sinp1j-30, 3, -1):
        for i in range(1020, klg, -1):
            if lungarray[j, i] > 7:
                sinp22i = i
                sinp22j = j
                break
        else:
            continue
        break

    aj = sinl1j - sinl11j 
    ai = sinl1i - sinl11i 
    bj = sinl2j - sinl22j 
    bi = sinl2i - sinl22i 

    cossinusl = (aj*bj + ai*bi)/(np.sqrt(aj**2 + ai**2) * np.sqrt(bj**2 + bi**2))
    sinusl = round(math.degrees(math.acos(cossinusl)), 2)

    aj = sinp1j - sinp11j 
    ai = sinp1i - sinp11i 
    bj = sinp2j - sinp22j 
    bi = sinp2i - sinp22i 

    cossinusp = (aj*bj + ai*bi)/(np.sqrt(aj**2 + ai**2) * np.sqrt(bj**2 + bi**2))
    sinusp = round(math.degrees(math.acos(cossinusp)), 2)
          
    return sinusl, sinusp, sinl1i, sinl1j, sinl11i, sinl11j, sinl2i, sinl2j, sinl22i, sinl22j, sinp1i, sinp1j, sinp11i, sinp11j, sinp2i, sinp2j, sinp22i, sinp22j 
    
# Функция нахождения кардиодиафрагмальных углов
def sinuskd(heartarray, bottomarray, botpi, botpj, botli, botlj):

    for j in range(3, 1020):
        if heartarray[botpi-30, j] > 7:
            sinkdp1j = j
            sinkdp1i = botpi - 30
            break
            
    for i in range(3, 1020):
        if bottomarray[i, botpj-30] > 7:
            sinkdp2i = i
            sinkdp2j = botpj-30
            break
            
    aj = sinkdp1j - botpj 
    ai = sinkdp1i - botpi 
    bj = sinkdp2j - botpj 
    bi = sinkdp2i - botpi 

    cossinusp = (aj*bj + ai*bi)/(np.sqrt(aj**2 + ai**2) * np.sqrt(bj**2 + bi**2))
    sinuskdp = round(math.degrees(math.acos(cossinusp)), 1)
    
    for j in range(1020, 3, -1):
        if heartarray[botli-30, j] > 7:
            sinkdl1j = j
            sinkdl1i = botli - 30
            break
            
    for i in range(3, 1020):
        if bottomarray[i, botlj+30] > 7:
            sinkdl2i = i
            sinkdl2j = botlj+30
            break
            
    aj = sinkdl1j - botlj 
    ai = sinkdl1i - botli 
    bj = sinkdl2j - botlj 
    bi = sinkdl2i - botli 

    cossinusl = (aj*bj + ai*bi)/(np.sqrt(aj**2 + ai**2) * np.sqrt(bj**2 + bi**2))
    sinuskdl = round(math.degrees(math.acos(cossinusl)), 1)    
    print("Кардиодиафрагмальный угол = " + str(sinuskdp) + ", " + str(sinuskdl))
    
    return sinkdp1j, sinkdp1i, sinkdp2j, sinkdp2i, sinkdl1j, sinkdl1i, sinkdl2j, sinkdl2i, sinuskdp, sinuskdl

# Ищем важные точки на сердце
def lavypavy(heartarray, vusota, vi, ni, li, lj, pi ,pj, klg):
    lt1y = vi + (vusota // 4)
    lt2y = ni - (vusota // 3)
    for j in range(3, 1020):
        if heartarray[lt1y,j] > 7:
            lt1x = j
            break
    for j in range(3, 1020):
        if heartarray[lt2y,j] > 7:
            lt2x = j
            break
    if lt2y > li:
        lt2y = li
        lt2x = lj
 
    minim = 0
    for i in range(lt1y + 1, lt2y):
        for j in range(3,klg):
            if heartarray[i,j] > 7:
                dlin = 0
                dlin = ((lt2y-lt1y)*j+(lt1x-lt2x)*i+(lt2x*lt1y-lt1x*lt2y))/np.sqrt((lt2y-lt1y)**2 + (lt1x-lt2x)**2)
                if (dlin >= 0) and (minim < dlin):
                    mini = i
                    minj = j
                    minim = dlin
                break
    
    vlpi = (mini-vi)//2 + vi 
    for j in range (3,1020):
        if heartarray[vlpi, j] > 7:
            vlpj = j
            break
            
    vlppi = vlpi
    for j in range(1020, klg, -1):
        if heartarray[vlppi, j] > 7:
            vlppj = j
            break

    pt1y = vi + 2*(vusota // 10)
    pt2y = pi
    for j in range(1020, 0, -1):
        if heartarray[pt1y,j] > 7:
            pt1x = j
            break

    pt2x = pj

    ppmini = 0
    minim = 0

    for i in range(pt1y + 1, pt2y):
        for j in range(1020, 0, -1):
            if heartarray[i,j] > 7:
                dlin = 0
                dlin = ((pt2y-pt1y)*j+(pt1x-pt2x)*i+(pt2x*pt1y-pt1x*pt2y))/np.sqrt((pt2y-pt1y)**2 + (pt1x-pt2x)**2)
                if  (minim < dlin):
                    ppmini = i
                    ppminj = j
                    minim = dlin
                break

    if ppmini != 0:
        pt2x = ppminj
        pt2y = ppmini
        
    minim = 0
    dlin = 0
    
    for i in range(pt1y + 1, pt2y):
        for j in range(1020, 0, -1):
            if heartarray[i,j] > 7:
                dlin = 0
                dlin = ((pt2y-pt1y)*j+(pt1x-pt2x)*i+(pt2x*pt1y-pt1x*pt2y))/np.sqrt((pt2y-pt1y)**2 + (pt1x-pt2x)**2)
                if  (minim < abs(dlin)):
                    minrasi = i
                    minrasj = j
                    minim = abs(dlin)
                    kk = dlin
                break
                
    minim = 0
    if (kk > 0):
        pt1x = minrasj
        pt1y = minrasi
        for i in range(pt1y + 1, pt2y):
            for j in range(1020, 0, -1):
                if heartarray[i,j] > 7:
                    dlin = 0
                    dlin = ((pt2y-pt1y)*j+(pt1x-pt2x)*i+(pt2x*pt1y-pt1x*pt2y))/np.sqrt((pt2y-pt1y)**2 + (pt1x-pt2x)**2)
                    if  (minim < abs(dlin)):
                        minrasi = i
                        minrasj = j
                        minim = abs(dlin)
                    break
                    
    lplj = minrasj + (pt2x - minrasj)//4
    for i in range(minrasi, pt2y):
        if heartarray[i, lplj] > 7:
            lpli = i
            break
            
    vppi = (minrasi - vi)//4 + vi
    for j in range (1020,klg, -1):
        if heartarray[vppi, j] > 7:
            vppj = j
            break
    
    rvp = abs(vppj - klg)
    rvl = abs(vlpj - klg)
    
    return mini, minj, minrasi, minrasj, vppi, vppj, vlpi, vlpj, rvl, rvp, lt1y, lt1x, lt2y, lt2x, lpli, lplj

# Точки пересечения диафрагмы и сердца
def bottompoint(heartarray, bottomarray, klg, ih, ni, nj):
    for j in range(1020, klg, -1):
        for i in range(3,ni):
            if (heartarray[i,j] > 7) and (bottomarray[i,j] > 7):
                botli = i
                botlj = j
                break
        else:
            continue
        break
        
    for j in range(3,klg):
        for i in range(3,ni):
            if (heartarray[i,j] > 7) and (bottomarray[i,j] > 7):
                botpi = i
                botpj = j
                break
        else:
            continue
        break
        
    if (botpi > ih):
        botpi = ih             
        for j in range(3,1020):
            if heartarray[botpi,j] > 7:
                botpj = j
                break
    return botli, botlj, botpi, botpj

# Верхняя точка нижней диафрагмы
def bottommaxi(bottomarray, klg):
    for i in range(3, 1020):
        for j in range(3, klg):
            if bottomarray[i,j] > 7:
                bmaxi = i
                break
        else: 
            continue
        break
    return bmaxi

# Вычисление индекса Мура
def moor(predictarray1, klg, minrasi, vi, lungwirina, pj):
    moori = minrasi - (minrasi-vi)//4 
    for j in range(pj, klg, -1):
        if predictarray1[moori, j] > 7:
            moorj = j
            break
    indexmoor = (moorj - klg)/(lungwirina/2) * 100 
    return indexmoor, moori, moorj   

# Высота от пересечения левой ключицы с легкими до правой точки сердца
def rastsrlkpipj(predictarray2, predictarray, pj):
    for j in range(1020, 3, -1):
        for i in range(3, 1020):
            if (predictarray2[i,j] > 7) and (predictarray[i,j] > 7):
                krkli = i
                krklj = j
                break
        else:
            continue
        break
    srklj = (krklj)
    srkli = krkli
    rastsr = srklj - pj
    return rastsr, srklj, srkli

# Координаты первой правой дуги
def pravdug1(predictarray1, mini, minj, vi, vj):
    k = []
    k.append(vi)
    k.append(vj)
    for i in range (vi, mini+1):
        for j in range (3, max(vj,minj)+1):
            if predictarray1[i,j] > 7:
                k.append(i)
                k.append(j)
                break
    return k

# Координаты второй правой дуги
def pravdug2(predictarray1, mini, minj, botpi, botpj):
    p = []    
    for i in range (botpi, mini-1, -1):
        for j in range (3,max(botpj,minj)+1):
            if predictarray1[i,j] > 7:
                p.append(i)
                p.append(j)
                break
    return p

# Координаты первой левой луги
def levdug1(predictarray1, d2i,d2j, vi,vj):
    m = []
    m.append(vi)
    m.append(vj)
    for i in range (vi, d2i+1):
        for j in range (1020,min(vj,d2j)-1,-1):
            if predictarray1[i,j] > 7:
                m.append(i)
                m.append(j)
                break
    return m

# Координаты второй левой дуги
def levdug2(predictarray1, minrasi, d2i,d2j, vj):

    mm = []    
    for i in range (d2i, minrasi+1):
        for j in range (1020,min(d2j, vj)-1,-1):
            if predictarray1[i,j] > 7:
                mm.append(i)
                mm.append(j)
                break
    return mm

# Координты третей левой дуги
def levdug3(predictarray1, minrasi, lpli, lplj, vj):
    n = []   

    for i in range (minrasi, lpli+1):
        for j in range (1020,min(lplj,vj)-1,-1):
            if predictarray1[i,j] > 7:
                n.append(i)
                n.append(j)
                break
    return n

# Координаты четвертой левой дуги
def levdug4(predictarray1, botli, botlj, lpli, lplj):
    q = []   
    
    for i in range (lpli, botli+1):
        for j in range (1020,min(botlj,lplj)-1,-1):
            if predictarray1[i,j] > 7:
                q.append(i)
                q.append(j)
                break
    return q

# Функция удаления "островков безумия". Остается только сердце
def removeOBheart(predictarray):
    contours, hierarchy = cv2.findContours(predictarray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    maxID=0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>max_area:
            cnt = contours[i]
            max_area = area
            maxID = i
    for ID in range(0,len(contours)):
        if ID != maxID:
            x, y, w, h = cv2.boundingRect(contours[ID])          
            for i in range(y,y+h):
                for j in range (x,x+w):
                    predictarray[i,j] = 0
    return predictarray

# Функция удаления "островков безумия". Остаются только два легких
def removeOBlung(predictarray1):
    contours, hierarchy = cv2.findContours(predictarray1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    ID1 = []
    maxID = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        ID1.append(i)
        if area>max_area:
            cnt = contours[i]
            max_area = area
            maxID = i
    ID1.remove(maxID)
    max_area = -1
    for i in ID1:
        area = cv2.contourArea(contours[i])
        if area>max_area:
            cnt = contours[i]
            max_area = area
            maxID = i
    ID1.remove(maxID)
    for ID in ID1:
            x, y, w, h = cv2.boundingRect(contours[ID]) 
            for i in range(y,y+h):
                for j in range (x,x+w):
                    predictarray1[i,j] = 0
    return predictarray1
    
# Вычисление дуги пучка
def dugpuchka(predictarray1, vppi, vppj, moori, moorj, pj, vj):
    Amin = 0
    for i in range(vppi,moori):
        for j in range(pj, vj, -1):
            if predictarray1[i,j] > 7:
                A = abs((vppi-moori)*j+(moorj-vppj)*i+vppj*moori-moorj*vppi)/np.sqrt((vppi-moori)**2 + (moorj-vppj)**2)
                if A > Amin:
                    d2i = i
                    d2j = j
                    Amin = A
                break
    return d2i, d2j

#     В данной функции убираются лишние ребра которые нашлись за пределами грудной клетки
def lungrebra(predictarray, predictarray3):
    lungrebraarray = np.zeros((1024,1024),np.uint16)
    for i in range(0,1023):
        for j in range(0,1023):
            if (predictarray[i,j] != 0) and (predictarray3[i,j] != 0):
                lungrebraarray[i,j] = predictarray3[i,j]
    npy = lungrebraarray
    ima = (npy).astype(np.uint16)
    fileres = way + "data/dicom_ribslung.jpg"
    cv2.imwrite(fileres, ima)
#     На выходе массив с ребрами в пределах грудной клетки
    return(lungrebraarray)

#     Функция подсчета количества ребер
def rebra(predictarray3, klg , lungli):
#     Создаем лист количества ребер, будем выбиравть из них максимальное
    kolvo = []
#     Находим опытным путем прямую от которой будем смещаться влево, вправо на i и подсчитывать количество ребер
    kld = (klg - lungli)//2 + lungli
    for i in range(0, (klg - lungli)//2):
#         Подсчитываем количество ребер по прямой
        kolvo1 = kolvorebra(kld, i, predictarray3)
        kolvo.append(kolvo1)
        kolvo2 = kolvorebra(kld, -i, predictarray3)
        kolvo.append(kolvo2)
#   Находим максимум из подсчитанных количеств ребер - это и есть количество ребер
    maxkolvo = max(kolvo)
    print("Количество ребер = ",maxkolvo + 2)
    
    return maxkolvo + 2

#     Основная функция программы где запускаются другие функции, 
# обрабатывается изображение и создается новый dicom файл и отправляется на сервер
# На вход подаются (dicom файл, имя файла, папка где находится файл)
def predict(imagedicom , filename):   
       
    imagedicomanalyz = imagedicom
#     Получаем изображение из dicom файла
    image = imagedicom.pixel_array

    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)

    print("Directory Path:", Path().absolute())
#     Соханяем его в формате jpg
    print(way + "data/dicom.jpg")
    # imageio.imwrite(way + "data\\" + filename + ".jpg", image)
    im = Image.fromarray(image)
    im.save(way + "data/dicom.jpg")
    folderjpg = way + "data/dicom.jpg"
#     Открываем изображение которое сохранили
    img = Image.open(folderjpg).convert("RGB")
    
#     Загрузка модели сегментации сердца
    def heart(predarray):
        loaded_model_heart = loadmodel("heart")
    #     Привязка для сохранения сегментированной области
        fileh = "%s_heart.jpg"
    #     Сегментация сердца из исходного изображения и сохранение
        predictimage = segment_image(img, fileh, loaded_model_heart)
        del loaded_model_heart
    #     Присвоение массива полученной области
        heartarray = predictimage
        heartarray = removeOBheart(heartarray)
        predarray["heart"]=heartarray
    # def diaphragm(predarray):
    #     Зазгрузка модели сегментации нижней части
        loaded_model_bottom = loadmodel("diaphragm")
    #     Привязка для сохранения сегментированной области
        fileb = "%s_diaphragm.jpg"
    #     Сегментация нижней части из исходного изображения и сохранение
        predictimage5 = segment_image(img, fileb, loaded_model_bottom)
        del loaded_model_bottom
    #     Присвоение массива полученной области
        bottomarray = predictimage5
        predarray["diaphragm"]=bottomarray
    
    def lung(predarray):
#     Загрузка модели сегментации легких
        loaded_model_lung = loadmodel("lung")
    #     Привязка для сохранения сегментированной области
        filel = "%s_lung.jpg"
    #     Сегментация легких из исходного изображения и сохранение
        predictimage1 = segment_image(img, filel, loaded_model_lung)
        del loaded_model_lung
    #     Присвоение массива полученной области
        lungarray = predictimage1
        lungarray = removeOBlung(lungarray)
        predarray["lung"]=lungarray

    # def ribs(predarray):
    #     Загрузка модели сегментации ребер
        loaded_model_rebra = loadmodel("ribs")
    #     Привязка для сохранения сегментированной области
        filep = "%s_ribs.jpg"   
    #     Сегментация ребер из исходного изображения и сохранение
        predictimage3 = segment_image(img, filep, loaded_model_rebra)
        del loaded_model_rebra
    #     Присвоение массива полученной области
        ribsarray = predictimage3
        #     Создаем массив ребер только в области грудной клетки
        rebralungarray  = lungrebra(lungarray, ribsarray)
        predarray["ribs"]=rebralungarray
        
    def kluchisa(predarray):
#     Загрузка модели сегментации ключицы
        loaded_model_kluchisa = loadmodel("clavicles")
    #     Привязка для сохранения сегментированной области
        filek = "%s_clavicles.jpg"      
    #     Сегментация ключиц из исходного изображения и сохранение
        claviclesarray = segment_image(img, filek, loaded_model_kluchisa)
        del loaded_model_kluchisa
        filetrain2 = way + "data/dicom_clavicles.jpg"
        predictimage2 = cv2.imread(filetrain2)
        #     Массив ключиц преобразуем в 8 бит
        claviclesarray = np.array(predictimage2, dtype= np.uint8)[:, :, 0]
        predarray["clavicles"]=claviclesarray

    def ostisa(predarray):
    #    Загрузка модели сегментации остиц
        loaded_model_ost = loadmodel("ostica")
    #     Привязка для сохранения сегментированной области
        fileo = "%s_ostica.jpg"
    #     Сегментация остиц из исходного изображения и сохранение
        predictimage4 = segment_image(img, fileo, loaded_model_ost)
        del loaded_model_ost
    #     Присвоение массива полученной области
        ostarray = predictimage4
        predarray["ostica"]=ostarray


    manager = Manager()

    predarray = manager.dict()

    p1 = Process(target=heart, args=(predarray,))
    p1.start()

    p2 = Process(target=lung, args=(predarray,))
    p2.start()

    p3 = Process(target=kluchisa, args=(predarray,))
    p3.start()

    p4 = Process(target=ostisa, args=(predarray,))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    heartarray = predarray["heart"]
    lungarray = predarray["lung"]
    claviclesarray = predarray["clavicles"]
    ribsarray = predarray["ribs"]
    ostarray = predarray["ostica"]
    bottomarray = predarray["diaphragm"]

#     Название файлов с привязками по сегментированным областям
    filetrain = way + "data/dicom_lung.jpg"
    filetrain1 = way + "data/dicom_heart.jpg"
    filetrain2 = way + "data/dicom_clavicles.jpg"
    # filetrain3 = way + "data/dicom_ribs.jpg"
    filetrain4 = way + "data/dicom_ostica.jpg"
    # filetrain5 = way + "data/dicom_diaphragm.jpg"
    
#     Открываем сохраненные изображения 
    predictimage = cv2.imread(filetrain)
    predictimage1 = cv2.imread(filetrain1)  
    predictimage2 = cv2.imread(filetrain2)    
    # predictimage3 = cv2.imread(filetrain3)
    predictimage4 = cv2.imread(filetrain4)
    # predictimage5 = cv2.imread(filetrain5)
        
#     Открываем исзодное изображение
    imageo = cv2.imread(folderjpg)
#     Изменяем размер изображения 
    imag = cv2.resize(imageo, (1024, 1024))

    filetrain6 = way + "data/dicom_ribslung.jpg"
    predictimage6 = cv2.imread(filetrain6)

    vi = Value('i', 0, lock=False)
    vj = Value('i', 0)
    p4 = Process(target=kr_verx, args=(heartarray,vi,vj,))
    p4.start()
    #     vi, vj - крайняя верхняя точка сердца (vi - У, vj - Х)
    # vi, vj = kr_verx(heartarray)
    ni = Value('i', 0, lock=False)
    nj = Value('i', 0)
    p5 = Process(target=kr_nizh, args=(heartarray,ni,nj,))
    p5.start()

    p4.join()
    p5.join()

    vi = vi.value
    vj = vj.value
    ni = ni.value
    nj = nj.value

#     ni, nj - крайняя верхняя точка сердца (ni - У, nj - Х)
    # ni, nj = kr_nizh(heartarray)
    # li = Value('i', 0)
    # lj = Value('i', 0)
    # p6 = Process(target=kr_leva, args=(heartarray,li,lj,vi,ni))
    # p6.start()
#     li, lj - крайняя левая точка сердца (li - У, lj - Х)

    li = 0
    lj = 0
    li, lj = kr_leva(heartarray,li,lj,vi,ni)
    # pi = Value('i', 0)
    # pj = Value('o', 0)
    # p7 = Process(target=kr_prav, args=(heartarray,pi,pj,vi,ni))
    # p7.start()
#     pi, pj - крайняя левая точка сердца (pi - У, pj - Х)
    pi = 0
    pj = 0



    pi, pj = kr_prav(heartarray,pi,pj,vi,ni)
    # p6.join()
    # p7.join()

#     Нахождение высоты сердца
    vusota = abs(ni - vi)
#     Нахождение ширины сердца
    wirina = abs(pj - lj)
    #      Вычиление срединной линии
#      klg - срединная линия
    klg, xt1, xt2, yt1, yt2, y1, x1 = kr_kl(claviclesarray)
#     Нахождение точки пересечения нижней диафрагмы с левой частью легких. ih - высота этой точки
    ih, jh = lungdlina(lungarray)
#     Нахождение крайних точек легких на высоте ih 
    lungli, lunglj, lungpi, lungpj = lungnormwirina(lungarray, ih)
#     Получаем крайние точки когда на ih не существет пересечения крайней части легких, путем проведения касательной
#     с - левая, сс - правая точки легких на высоте ih.
    point1i, point1j, c = lungnenormwirina1(lungarray, ih, vusota, vi)
    point2i, point2j, cc = lungnenormwirina2(lungarray, ih, vusota, vi)
#     Если крайней точки легкого не существует, то она смещается на нашу точку пересечения нижней диафрагмы с легкими. 
#     Далее идет сравнение этой крайней точки с точкой пересечения нижней диафрагмы с легкими. 
#     +-40 потому что бывает случаи когда нижняя точка имеет вид прямой и выбрана одна из них. 
#     Если не существует крайних точек с двух сторон, то крайние точки выявляются по касетельным
    if (jh-40<=lungli<=jh+40) and (jh-40<=lungpi<=jh+40):
        lungli = cc
        lungpi = c
#      Условие для не существующей точки слева
    elif (jh-40<=lungli<=jh+40) and (jh+40<lungpi):
        lungli = cc
#      Условие для не существующей точки слева
    elif (jh-40>lungli) and (jh-40<=lungpi<=jh+40):
        lungpi = c
#      Вычисление ширины легких
    lungwirina = abs(lungli - lungpi)
#      Вычисление линии по остицам
    ostline = ost_andrey2(ostarray , yt1, yt2)
#         Вычисление ПАВУ и ЛАВУ
    mini, minj, minrasi, minrasj, vppi, vppj, vlpi, vlpj, rvl, rvp, lt1y, lt1x, lt2y, lt2x, lpli, lplj = lavypavy(heartarray, vusota, vi, ni, li, lj, pi ,pj, klg)   
#     Вычисление нижних точек пересечения стени сердца с диафрагмой 
    botli, botlj, botpi, botpj = bottompoint(heartarray, bottomarray, klg, ih, ni, nj)   
#      Вычисление Кардиоторакльного индекса = отношение ширины сердца к ширину легих
    kti = wirina/lungwirina*100
#      Округление полученной величины
    KTI = round(kti, 2)
    print("Кардиоторакальный индекс = ",KTI,"%")
#     Вычиление Правопредсердного коэффициента = Отношение длины правого поперечника сердца к половине диаметра грудной клетки
    prprk = (abs(klg - lj) * 100)/0.5/lungwirina
#      Округление полученной величины
    prprk = round(prprk, 2)
    print("Правопредсердный коэффициент = ",prprk,"%")
#     Вычисление Отношения поперечного размера сердца = отношение правой части поперечника сердца на левую часть
    poprc = ((abs(klg - lj))/(abs(pj - klg))) * 100
#      Округление полученной величины
    poprc = round(poprc, 2)
    print("Отношение поперечного размера сердца = ", poprc,"%")
    
    zoom = np.sqrt((imagedicom.Columns/1024)**2 + (imagedicom.Rows/1024)**2)
    zoomi = imagedicom.Rows/1024
    zoomj = imagedicom.Columns/1024
    
    L = np.sqrt((mini*zoomi - botli*zoomi)**2 + (minj*zoomj - botlj*zoomj)**2) * imagedicom.PixelSpacing[0]

    vsp = abs(vi-mini) * imagedicom.PixelSpacing[0] * imagedicom.Rows / 1024
    vsp = round(vsp,2)
    print("Высота сосудистого пучка, см = ", vsp)
    
    vspp = abs(botpi - mini) * imagedicom.PixelSpacing[0] * imagedicom.Rows / 1024
    vspp = round(vspp,2)
    print("Высота сегмента правого предсердия, см = ", vspp)

    alk = rvp/(0.5 * lungwirina)*100
    alk = round(alk,2)
    print("Аортолегочный коэффициент, см = ", alk, "%")

    hdtpp = np.sqrt((mini*zoomi - botpi*zoomi)**2 + (minj*zoomj - botpj*zoomj)**2)
    hdtpp = round(hdtpp * imagedicom.PixelSpacing[0]/10, 1)
    print("Хорда дуги тени правого предсердия, см = ", hdtpp)

    hdtlj = np.sqrt((lpli*zoomi - botli*zoomi)**2 + (lplj*zoomj - botlj*zoomj)**2)
    hdtlj = round(hdtlj * imagedicom.PixelSpacing[0]/10, 1)
    print("Хорда дуги тени левого желудочка, см = ", hdtlj)

    indexmoor, moori, moorj = moor(heartarray, klg, minrasi, vi, lungwirina, pj)
    indexmoor = round(indexmoor, 2)
    print("Индекс Мура = ", indexmoor, "%")
    
    Q1 = abs((botli-mini)*zoomi*lplj*zoomj+(minj-botlj)*zoomj*lpli*zoomi+botlj*zoomj*mini*zoomi-minj*zoomj*botli*zoomi)/np.sqrt((mini*zoomi-botli*zoomi)**2 + (minj*zoomj-botlj*zoomj)**2)
    Q2 = abs((botli-mini)*zoomi*botpj*zoomj+(minj-botlj)*zoomj*botpi*zoomi+botlj*zoomj*mini*zoomi-minj*zoomj*botli*zoomi)/np.sqrt((mini*zoomi-botli*zoomi)**2 + (minj*zoomj-botlj*zoomj)**2)
    
    qlj = minj*zoomj + (botlj-minj)*zoomj*((lplj-minj)*(botlj-minj)*zoomj*zoomj+(lpli-mini)*(botli-mini)*zoomi*zoomi)/((botlj-minj)*(botlj-minj)*zoomj*zoomj+(botli-mini)*(botli-mini)*zoomi*zoomi)
    qli = mini*zoomi + (botli-mini)*zoomi*((lplj-minj)*(botlj-minj)*zoomj*zoomj+(lpli-mini)*(botli-mini)*zoomi*zoomi)/((botlj-minj)*(botlj-minj)*zoomj*zoomj+(botli-mini)*(botli-mini)*zoomi*zoomi)
    qpj = minj*zoomj + (botlj-minj)*zoomj*((botpj-minj)*(botlj-minj)*zoomj*zoomj+(botpi-mini)*(botli-mini)*zoomi*zoomi)/((botlj-minj)*(botlj-minj)*zoomj*zoomj+(botli-mini)*(botli-mini)*zoomi*zoomi)
    qpi = mini*zoomi + (botli-mini)*zoomi*((botpj-minj)*(botlj-minj)*zoomj*zoomj+(botpi-mini)*(botli-mini)*zoomi*zoomi)/((botlj-minj)*(botlj-minj)*zoomj*zoomj+(botli-mini)*(botli-mini)*zoomi*zoomi)
    
    qlj = qlj/zoomj
    qli = qli/zoomi
    qpj = qpj/zoomj
    qpi = qpi/zoomi
    print("Q1 = ", round(Q1*imagedicom.PixelSpacing[0],2))
    print("Q2 = ", round(Q2*imagedicom.PixelSpacing[0],2))
    Fa = L  * (Q1 + Q2)*imagedicom.PixelSpacing[0]* 0.735
    Fa = Fa/100
    Fa = round(Fa, 2)
    print("Площадь фронтального силуэта, см2 = ", Fa)

    V = 0.53 * pow((L * (Q1 + Q2) * imagedicom.PixelSpacing[0] * np.pi/4), 3/2)
    V = V/1000
    V = round(V, 2)
    print("Объем сердца, см3 = ", V)
    L = round(L/10,1)
    print("L = ", L)

    Amin = 0
    for y in range(mini, botpi):
        for x in range(lj, klg):
            if heartarray[y, x] > 7:
                A = abs((botpi-mini)*zoomi*x*zoomj+(minj-botpj)*zoomj*y*zoomi-minj*zoomj*botpi*zoomi+botpj*zoomj*mini*zoomi)/np.sqrt((botpi*zoomi-mini*zoomi)**2 + (botpj*zoomj-minj*zoomj)**2)
                if A > Amin:
                    ptii = y
                    ptij = x
                    Amin = A
                break

    pti = Amin/(lungwirina * zoomj /2)
    pti = round(pti, 2)
    print("Предсердно-торакальный индекс, ПТИ =", pti)
    
    bmaxi = bottommaxi(bottomarray, klg)
    vgk = abs(bmaxi - yt1) * imagedicom.PixelSpacing[0] * imagedicom.Rows / 1024
    vgk = round(vgk,1)
    print("Высота грудной клетки = ", vgk )
    
    tanalphaL = (abs(botli-mini)*zoomi)/(abs(botlj-minj)*zoomj)
    alphaL = math.atan(tanalphaL)
    alphaL = round(alphaL*180/np.pi, 1)
    print("Угол наклона длинника сердца к горизонтали = ", alphaL)
    
    rastsr, srklj, srkli = rastsrlkpipj(claviclesarray, lungarray, pj)
    rastr = round(rastsr*imagedicom.PixelSpacing[0]*imagedicom.Columns / 1024, 1 )
    print("Расстояние от верхушки сердца до середины ключицы = ", rastr)
    
    d2i,d2j = dugpuchka(heartarray, vppi, vppj, moori, moorj, pj, vj)

#     Подсчитываем количество ребер
    kolvoreber = rebra(ribsarray, klg, lungli)
    
    k = pravdug1(heartarray, mini, minj, vi, vj)
    p = pravdug2(heartarray, mini, minj, botpi, botpj)
    m = levdug1(heartarray, d2i,d2j,vi,vj)
    mm = levdug2(heartarray, minrasi, d2i,d2j, vj)
    n = levdug3(heartarray, minrasi, lpli, lplj, vj)
    q = levdug4(heartarray, botli, botli, lpli, lplj)   

#     Создаем изображение с наложением сегментированных областей на исходную 
    ima = add_colored_mask3(imag, predictimage, predictimage1, predictimage2)
    img = Image.fromarray(ima.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(resource_path("style.otf"), 16)
    fill = "rgb(120,25,0)"
    for i in range(0,len(k)-3,2):
        draw.line ((k[i+1], k[i], k[i+3], k[i+2]), fill = "rgb(0, 255, 0)", width=2, joint='curve')
    for i in range(0,len(p)-3,2): 
        draw.line ((p[i+1], p[i], p[i+3], p[i+2]), fill = "rgb(0, 255, 255)", width=2, joint='curve')
    for i in range(0,len(m)-3,2):  
        draw.line ((m[i+1], m[i], m[i+3], m[i+2]), fill = "rgb(255, 255, 0)", width=2, joint='curve')        
    for i in range(0,len(mm)-3,2):
        draw.line ((mm[i+1], mm[i], mm[i+3], mm[i+2]), fill = "rgb(255, 0, 255)", width=2, joint='curve')
    for i in range(0,len(n)-3,2):
        draw.line ((n[i+1], n[i], n[i+3], n[i+2]), fill = "rgb(0, 0, 255)", width=2, joint='curve')
    for i in range(0,len(q)-3,2):
        draw.line ((q[i+1], q[i], q[i+3], q[i+2]), fill = "rgb(255, 0, 0)", width=2, joint='curve')        
#     Получаем рамеры исходного изображения для сохранения в таком же размере нового dicom файла с результатами
    k , l , r = imageo.shape
    
    otnvspkvspp = vsp/vspp
    otnvspkvspp = round(otnvspkvspp, 2)
    rvl = rvl * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024
    rvl = round(rvl/10, 1)
    rvp = rvp * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024
    rvp = round(rvp/10, 1)
    
#     Проводим линии используемых величин
    draw.line ((lj, li, klg, li), fill = "rgb(0, 0, 255)", width=2, joint='curve')
    draw.text(((lj+klg)//2, li-10),str(round((klg - lj) * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024 / 10, 1)), fill=fill, font=font)
    draw.line ((pj, pi, klg, pi), fill = "rgb(0, 0, 255)", width=2, joint='curve')
    draw.text(((pj+klg)//2 , pi-10),str(round((pj - klg) * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024 /10, 1)), fill=fill, font=font)
    draw.line ((lplj,lpli, qlj, qli), fill = "rgb(255, 0, 255)", width=2, joint='curve')
    draw.line ((botpj,botpi, qpj, qpi), fill = "rgb(255, 0, 255)", width=2, joint='curve')
    
    draw.line ((klg, vi, klg, ni), fill = "rgb(0, 0, 255)", width=2, joint='curve')
    draw.line ((lungli, ih, lungpi, ih), fill = "rgb(0, 255, 0)", width=2, joint='curve')
    draw.line ((vppj, vppi, klg, vppi), fill = "rgb(255, 0, 0)", width=2, joint='curve')
    draw.line ((vlpj, vlpi, klg, vlpi), fill = "rgb(255, 0, 0)", width=2, joint='curve')  
    
    draw.line ((pj,pi,pj, srkli), fill = "rgb(255, 0, 0)", width=1, joint='curve')
    draw.line ((pj, srkli, srklj, srkli), fill = "rgb(255, 0, 0)", width=1, joint='curve')
    draw.text((srklj-20, srkli+20),str(rastr), fill=fill, font=font)
    
    draw.line ((klg,moori,moorj, moori), fill = "rgb(255, 255, 0)", width=2, joint='curve')
    draw.text((klg + 20, moori + 20),"ИМ = " + str(indexmoor) + "% (<30%)", fill=fill, font=font)

    txj = ((moori-d2i+(d2i-minrasi)/(d2j-minrasj)*d2j)*(d2j-minrasj))/(d2i-minrasi)

    txj = int(round(txj, 0))

    zoomi = imagedicom.Rows/1024
    zoomj = imagedicom.Columns/1024
    
    resx = ""
    aminx = ""
    if (moorj - txj > 0):
        Amin = 0
        vmoori = d2i
        vmoorj = pj
        for y in range(d2i, minrasi):
            for x in range(pj, klg,-1):
                if heartarray[y, x] > 7:
                    A = abs((minrasi-d2i)*zoomi*x*zoomj+(d2j-minrasj)*zoomj*y*zoomi+(minrasj*zoomj*d2i*zoomi-d2j*zoomj*minrasi*zoomi))/np.sqrt((minrasi*zoomi-d2i*zoomi)**2 + (d2j*zoomj-minrasj*zoomj)**2)
                    if A > Amin:
                        Amin = A
                        vmoori = y
                        vmoorj = x
                    break
        print("Выпуклость легочного сегмента = " + str(round(Amin/10,1)) + "мм ")
        aminx = "Выпуклость легочного сегмента = " + str(round(Amin/10,1)) + "мм (0-1мм)"
        if Amin/10 >= 2:
            resx = "Признак легочной гипертензии. " 
            xmoor = minrasj + (d2j-minrasj)*((vmoorj-minrasj)*(d2j-minrasj) + (vmoori - minrasi)*(d2i-minrasi))/((d2j -minrasj)*(d2j-minrasj) + (d2i-minrasi)*(d2i-minrasi))
            ymoor = minrasi + (d2i-minrasi)*((vmoorj-minrasj)*(d2j-minrasj) + (vmoori - minrasi)*(d2i-minrasi))/((d2j -minrasj)*(d2j-minrasj) + (d2i-minrasi)*(d2i-minrasi))
            xmoor = round(xmoor, 0)
            ymoor = round(ymoor, 0)
    
            draw.ellipse((xmoor-1,ymoor-1,xmoor+1,ymoor+1), fill="black")
            draw.line((xmoor, ymoor, vmoorj, vmoori), fill = "rgb(0, 100, 0)", width=1, joint='curve')
            draw.line((minrasj, minrasi, d2j, d2i), fill = "rgb(0, 100, 0)", width=1, joint='curve')
        draw.text((vmoorj + 10, vmoori-10),"ВЛС = " + str(round(Amin/10,1)) + "мм", fill=fill, font=font)
        draw.ellipse((vmoorj-1,vmoori-1,vmoorj+1,vmoori+1), fill="black")
    draw.line((botlj, botli, minj, mini), fill = "rgb(0, 170, 0)", width=2, joint='curve')
    draw.text((botlj - 150, botli + 20),"Угол длинника сердца = " + str(alphaL), fill=fill, font=font)
    draw.text((botlj - 150, botli + 40),"Длина длинника сердца = " + str(L) + " (<15.5см)", fill=fill, font=font)
    
    draw.ellipse((minj-1,mini-1,minj+1,mini+1), fill="black")
    draw.ellipse((botlj-1,botli-1,botlj+1,botli+1), fill="black")
    draw.ellipse((minrasj-1,minrasi-1,minrasj+1,minrasi+1), fill="black")
    draw.ellipse((d2j-1,d2i-1,d2j+1,d2i+1), fill="black")
    draw.ellipse((botpj-1,botpi-1,botpj+1,botpi+1), fill="black")
    draw.ellipse((lplj-1,lpli-1,lplj+1,lpli+1), fill="black")
    
    draw.text((vlpj+20, vlpi-10),str(rvl) + " (3-4см)", fill=fill, font=font)
    draw.text((klg+20, vppi-10),str(rvp) + " (3-4см)", fill=fill, font=font)
    draw.text((klg + 20, vppi + 20),"АЛК = " + str(alk) + "% (<20%)", fill=fill, font=font)
    draw.text((klg + 20, (vi+ni)//2+50),"КТИ = " + str(KTI) + "%", fill=fill, font=font)
    draw.text((40, 900),"Правопредсердный коэффициент = " + str(prprk) + "% (20-30%)", fill=fill, font=font)

    pol = ''
    if imagedicom.PatientSex == "F":
        draw.text((40, 930),"Площадь фронтально силуэта сердца = " + str(Fa) + " (жен.108.8-118.0см2)", fill=fill, font=font)
        pol = 'F'
    elif imagedicom.PatientSex == "M":
        draw.text((40, 930), "Площадь фронтально силуэта сердца = " + str(Fa) + " (муж.105.8-143.7см2)", fill=fill, font=font)
        pol = 'M'
    else:
        draw.text((40, 930), "Площадь фронтально силуэта сердца = " + str(Fa) + " (м.105.8-143.7см2 ж.108.8-118.0см2)" , fill=fill, font=font)
    
    if imagedicom.PatientSex == "F":
        draw.text((40, 870), "Объем сердца = " + str(V) + " (жен.508-647см3)" , fill=fill, font=font)
        pol = 'F'
    elif imagedicom.PatientSex == "M":
        draw.text((40, 870), "Объем сердца = " + str(V) + " (муж.656-882см3)" , fill=fill, font=font)
        pol = 'M'
    else:
        draw.text((40, 870), "Объем сердца = " + str(V) + " (м.656-882см3 ж.508-647см3)" , fill=fill, font=font)

    draw.text((40, 960), "Отношение высоты сосуд.пучка к высоте сегмента правого предсердия = " + str(otnvspkvspp) + " (1:1)" , fill=fill, font=font)
       
    ima = np.array(img,dtype = np.uint8)
#    Изменяем размер как в исходном изображении
    ima = cv2.resize(ima,(l,k))
#     Сохраняем его dicom файл в тег Pixel Data
    im = Image.fromarray(ima)
    im.save(way + "data/dicom_result.jpg")
    imagedicom.PixelData = ima.tobytes()
#     Изменяем его модальность на CR
    imagedicom.Modality = "CR"
#     Сохраняем его Высоту и Ширину 
    imagedicom.Rows,imagedicom.Columns = ima.shape[0], ima.shape[1]
#     Изменяем количество каналов
    imagedicom.SamplesPerPixel = 3
#     Сообщаем, что это RGB изображение
    imagedicom.PhotometricInterpretation = "RGB"
#     Сообщаем, что битность равна 8
    imagedicom.BitsAllocated = 8
    imagedicom.BitsStored = 8
    imagedicom.HighBit = 7 
    imagedicom.PlanarConfiguration = 0
    imagedicom.WindowCenter = 128
    imagedicom.WindowWidth = 256
    imagedicom.SeriesNumber = 5
#     Изменяем SOPInstanceUID, чтобы сервер принимал его как другой файл 
    imagedicom[0x8,0x18].value = (imagedicom[0x8,0x18].value) + "7"
#     Изменяем SeriesInstanceUID, чтобы не было серией исходного изображения
    imagedicom[0x20,0xe].value = (imagedicom[0x20,0xe].value) + "7"
#     Функция которая позволяет сохранить в dicom архивированные изображения
    imagedicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
#     Сохраняем в dicom формате 
    imagedicom.save_as(way + "dicom/resultdicom") 
#     Отправка на сервер с помощью DCMTK
    # os.system("storescu -aec ARCHIMED 10.80.4.207 104 /home/ard/orthanc/dicom/resultdicom")                                                                                                                                                                                               



    sinusl, sinusp, sinl1i, sinl1j, sinl11i, sinl11j, sinl2i, sinl2j, sinl22i, sinl22j, sinp1i, sinp1j, sinp11i, sinp11j, sinp2i, sinp2j, sinp22i, sinp22j   = sinus(lungarray, klg)
    sinkdp1j, sinkdp1i, sinkdp2j, sinkdp2i, sinkdl1j, sinkdl1i, sinkdl2j, sinkdl2i, sinuskdp, sinuskdl = sinuskd(heartarray, bottomarray, botpi, botpj, botli, botlj)

    imaanalyz = add_colored_mask3(imag, predictimage6, predictimage4, predictimage2)
    imganalyz = Image.fromarray(imaanalyz.astype('uint8'), 'RGB')
    
    draw1 = ImageDraw.Draw(imganalyz)
    font = ImageFont.truetype(resource_path("style.otf"), 24)
    fill = "rgb(120,25,0)"
    draw1.line ((klg, 0, klg, 1023), fill = "rgb(0, 0, 255)", width=2, joint='curve')
    draw1.line ((ostline, 0, ostline, 1023), fill = "rgb(48, 213, 200)", width=2, joint='curve')
    draw1.text((200, 840),"Количество ребер = " + str(kolvoreber), fill=fill, font=font)
    draw1.line((sinkdp1j, sinkdp1i, botpj, botpi), fill = "rgb(128, 0, 128)", width=2, joint='curve')
    draw1.line((sinkdp2j, sinkdp2i, botpj, botpi), fill = "rgb(128, 0, 128)", width=2, joint='curve')
    
    draw1.line((sinkdl1j, sinkdl1i, botlj, botli), fill = "rgb(128, 0, 128)", width=2, joint='curve')
    draw1.line((sinkdl2j, sinkdl2i, botlj, botli), fill = "rgb(128, 0, 128)", width=2, joint='curve')
    
    razn = (klg-ostline) * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024
    popheart = hdtlj + hdtpp
    popheart = round(popheart, 1)
    
    if sinusl > 0:
        draw1.line ((sinl1i, sinl1j, sinl11i, sinl11j), fill = "rgb(0, 0, 255)", width=2, joint='curve')
        draw1.line ((sinl2i, sinl2j, sinl22i, sinl22j), fill = "rgb(48, 213, 200)", width=2, joint='curve')
    if sinusp > 0:
        draw1.line ((sinp1i, sinp1j, sinp11i, sinp11j), fill = "rgb(0, 0, 255)", width=2, joint='curve')
        draw1.line ((sinp2i, sinp2j, sinp22i, sinp22j), fill = "rgb(48, 213, 200)", width=2, joint='curve')
    tpov = ""
    vdoh = ""
    respov = ""
    if (abs(razn) <= 5):
        tpov = "Установка пациента по вертикальной оси правильная"
        respov = "Установка правильная, разворота нет. "
        draw1.text((ostline+10, vi + 30),"Установка пациента правильная", fill=fill, font=font)
    elif (abs(razn) > 5) and (abs(razn) < 10):
        tpov = "Установка пациента по вертикальной оси немного отклонена"
        draw1.text((ostline+10, vi + 30),"Установка пациента с небольшим поворотом", fill=fill, font=font)
        respov = "Установка пациента с небольшим поворотом"
        if razn > 0:
            draw1.text((ostline+10, vi + 60),"Повернут влево", fill=fill, font=font)
            tpov = tpov + " влево"
            respov = respov + " влево. "
            draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
        else:
            tpov = tpov + " вправо"
            respov = respov + " вправо. "
            draw1.text((ostline+10, vi + 60),"Повернут вправо", fill=fill, font=font)
            draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
    else:
        tpov = "Установка пациента по вертикальной оси отклонена"
        draw1.text((ostline+10, vi + 30),"Установка пациента с поворотом", fill=fill, font=font)
        respov = "Установка пациента с поворотом"
        if razn > 0:
            draw1.text((ostline+10, vi + 60),"Повернут влево", fill=fill, font=font)
            tpov = tpov + " влево"
            respov = respov + " влево. "
            draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
        else:
            tpov = tpov + " вправо"
            respov = respov + " вправо. "
            draw1.text((ostline+10, vi + 60),"Повернут вправо", fill=fill, font=font)
            draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
            
    draw1.text((ostline+10, vi),"Разница = " + str(abs(round(razn, 2))) + "мм", fill=fill, font=font)
    
    if (kolvoreber >= 12):
        vdoh = "Глубокий вдох. "
        draw1.text((200, 870),"Глубокий вдох", fill=fill, font=font)
    elif (kolvoreber >= 9):
        vdoh = "Обычный вдох. "
        draw1.text((200, 870),"Обычный вдох", fill=fill, font=font)
    else:
        vdoh = "Недостаточный вдох. "
        draw1.text((200, 870),"Недостаточный вдох", fill=fill, font=font)
        draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
        
    texts = ""
    sinusresl = "острый"
    sinusresp = "острый"

    if sinusl > 90:
        sinusresl = "закруглен"
    if sinusp > 90:
        sinusresp = "закруглен"
    if sinusp == 0:
        texts = "\r\nНижний отдел грудной клетки срезан"
        sinusresp = "Не определен"
    if sinusl == 0:
        texts = "\r\nНижний отдел грудной клетки срезан"
        sinusresl = "Не определен"
    if sinusp == -1:
        texts = "\r\nНижний отдел грудной клетки срезан"
        sinusresp = "Не определен"
    if sinusl == -1:
        texts = "\r\nНижний отдел грудной клетки срезан"
        sinusresl = "Не определен"
    if "отдел" in texts:
        draw1.rectangle((10, 10, 100, 100), fill=(250, 0, 0), outline=(255, 255, 255))
    
    if sinusl > 90 and sinusp > 90:
        ressin = "Легочно-плевральные синусы облитерированы с обеих сторон. "
    elif (sinusl <= 90) and (sinusp <= 90):
        ressin = "Легочно-плевральные синусы свободны с обеих сторон. "
    elif sinusl > 90 and sinusp <=90:
        ressin = "Правый синус облитерирован. Левый синус свободен. "
    else:
        ressin = "Левый синус облитерирован. Правый синус свободен. "
        
    textppk = ""
    if prprk > 30:
    	textppk = "Признаки увеличения правого предсердия. "

    imaanalyz = np.array(imganalyz,dtype = np.uint8)
    im = Image.fromarray(imaanalyz)
    im.save(way + "data/dicom_analyze.jpg")
#     Изменяем размер как в исходном изображении
    imaanalyz = cv2.resize(imaanalyz,(l,k))
#     Сохраняем его dicom файл в тег Pixel Data
    imagedicomanalyz.PixelData = imaanalyz.tobytes()
#     Изменяем его модальность на CR
    imagedicomanalyz.Modality = "CR"
#     Сохраняем его Высоту и Ширину 
    imagedicomanalyz.Rows,imagedicomanalyz.Columns = imaanalyz.shape[0], imaanalyz.shape[1]
#     Изменяем количество каналов
    imagedicomanalyz.SamplesPerPixel = 3
#     Сообщаем, что это RGB изображение
    imagedicomanalyz.PhotometricInterpretation = "RGB"
#     Сообщаем, что битность равна 8
    imagedicomanalyz.BitsAllocated = 8
    imagedicomanalyz.BitsStored = 8
    imagedicomanalyz.HighBit = 7 
    imagedicomanalyz.PlanarConfiguration = 0
    imagedicomanalyz.WindowCenter = 128 
    imagedicomanalyz.WindowWidth = 256
    imagedicomanalyz.SeriesNumber = 4
#     Изменяем SOPInstanceUID, чтобы сервер принимал его как другой файл 
    imagedicomanalyz[0x8,0x18].value = (imagedicom[0x8,0x18].value) + "8"
#     Изменяем SeriesInstanceUID, чтобы не было серией исходного изображения
    imagedicomanalyz[0x20,0xe].value = (imagedicom[0x20,0xe].value) + "8"
#     Функция которая позволяет сохранить в dicom архивированные изображения
    imagedicomanalyz.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
#     Сохраняем в dicom формате 
    imagedicomanalyz.save_as(way + "dicom/resultanalyz") 
#     Отправка на сервер с помощью DCMTK
    # os.system("storescu -aec ARCHIMED 10.80.4.207 104 /home/ard/orthanc/dicom/resultanalyz")

    srimage = dcmread(resource_path('SR_empty'))
    if pol == 'F':
        text1 = "Площадь фронтального силуэта сердца, см2 = " + str(Fa) + " (жен.108.8-118.0см2)\r\nОбъем сердца, см3 = " + str(V) + " (жен.508-647см3)\r\n"
    elif pol == 'M':
        text1 = "Площадь фронтального силуэта сердца, см2 = " + str(Fa) + " (муж.105.8-143.7см2)\r\nОбъем сердца, см3 = " + str(V) + " (муж.656-882см3)\r\n"
    else:
        text1 = "Площадь фронтального силуэта сердца, см2 = " + str(Fa) + " (муж.105.8-143.7см2 жен.108.8-118.0см2)\r\nОбъем сердца, см3 = " + str(V) + " (муж.656-882см3 жен.508-647см3)\r\n"
        
    textangle = ''
    if (alphaL >= 45) and (alphaL < 46):
        textangle = 'Косое (Нормостеник)'
    elif alphaL >= 46:
        textangle = 'Вертикальное (Астеник)'
    else:
        textangle = 'Горизонтальное (Гиперстеник)'
    
    resmoor = ""
    if indexmoor > 40:
        resmoor = "III степень расширения ствола легочной артерии. "
    elif indexmoor > 35:
        resmoor = "II степень расширения ствола легочной артерии. "
    elif indexmoor > 30:
        resmoor = "I степень расширения ствола легочной артерии. "
    
    srimage.ContentSequence[0].TextValue = "\r\nКоличество ребер = " + str(kolvoreber) + "\r\n" + vdoh + "\r\n" + "\r\nРасстояние между срединной линией, проведенной по остистым отроскам, и линией, расположенной между грудинно-ключичными сочленениями, мм = " + str(abs(round(razn, 2)))+ "\r\n" + tpov + "\r\n" + "\r\nРеберно-диафрагмальный синус правый = "+ str(sinusresl) + "\r\nРеберно-диафрагмальный синус левый = "+ str(sinusresp) + texts + "\r\n"
    
    textright = "По правому контуру сердца:\r\n" + "\r\nПредсердно-торакальный Индекс = " + str(pti) + "(<0.3)\r\nПравопредсердный коэффициент = " + str(prprk) + " % (20-30%)\r\nХорда дуги правого предсердия, см = " + str(hdtpp) + "\r\nПравая часть поперечного размера аорты, см = " + str(rvl) + "(3-4см)" + "\r\n" + "Правая часть поперечника сердца = " + str(round((klg - lj) * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024 / 10, 1)) + "\r\n"
    
    textleft = "\r\nПо левому контуру сердца:\r\n" +"\r\nИндекс Мура = " + str(indexmoor) + " % (<30%)\r\nАортолегочный Коэффициент = " + str(alk) + " % (<20%)\r\nХорда дуги левого желудочка, см = " + str(hdtlj) + "\r\nЛевая часть поперечного размера аорты, см = " + str(rvp) + "(3-4см)" +"\r\n" + str(aminx) + "\r\n" + "Левая часть поперечника сердца = " + str(round((pj - klg) * imagedicom.PixelSpacing[1] * imagedicom.Columns / 1024 / 10, 1)) + "\r\n" 
    
    textall = "\r\nОбщие параметры:\r\n" + "\r\nПоложение сердца = " + textangle + "\r\nКардиоторакальный Индекс = " + str(KTI) + " % (~50%)" + "\r\nДлинник сердца, см = " + str(L) + "(<15.5см)\r\nПоперечник сердца, см = " + str(popheart) + "\r\nУгол наклона длинника сердца = " + str(alphaL)  + "\r\nВысота сосудистого пучка, см = " +str(vsp) +  "\r\nОтношение высоты сосуд.пучка к высоте тени правого предсердия = " + str(otnvspkvspp) + " (1:1)\r\n" + str(text1) + "\r\n"
    
    srimage.ContentSequence[1].TextValue  = textall + textright + textleft
    
    reskti = ""
    if KTI < 50:
        reskti = "Тень сердца не расширена. "
    else:
        reskti = "Признаки расширения тени сердца. КТИ = " + str(KTI) + "%. "
    
    resalk = ""
    if alk > 20:
        resalk = "Признаки расширения дуги аорты. "
    resvao = ""
    if rvl > 4:
        resvao = "Расширение восходящего отдела аорты. "

    srimage.ContentSequence[2].TextValue = "\r\n" + respov + vdoh + ressin + reskti + textppk + resmoor + resalk + resvao + resx
    sr_dataset = ComprehensiveSR(
        evidence=[imagedicom],
        content=srimage,
        series_number=1,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        instance_number=1)

    sr_dataset.SeriesNumber = 6
    sr_dataset.ContentDate = imagedicom.StudyDate
    sr_dataset.ContentTime = imagedicom.StudyTime
    sr_dataset.SeriesDescription = "RERAD Neural Network Report"
    sr_dataset[0x8,0x18].value = (imagedicom[0x8,0x18].value) + "9"
    sr_dataset.save_as(way + "dicom/SR_dicom")
    # os.system("storescu -aec ARCHIMED 10.80.4.207 104 /home/ard/orthanc/dicom/SR_dicom")

#     Закрываем сессию keras
    keras.backend.clear_session()
#     Удаляем из памяти загруженные модели

    gc.collect()

#     Возвращаем имя файла, КТИ, Правопредсердный коэффициент, Отношение поперечного размера сердца
    return KTI

def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
way = "./"

def clicked():
    filename = askopenfilename()   
    # Открываем dicom файл
    imagedicom = dcmread(filename)
    label1 = tk.Label(text=filename + " - файл прочитан")
    label1.pack()
    label2 = tk.Label(text="Идет обработка файла...")
    label2.pack()


    t1 = time.time()
    KTI = predict(imagedicom, filename)
    print("Обработка файла " +filename+ " прошла успешно!")
    t2 = time.time()
    print('TIME = ', t2-t1)
    label3 = tk.Label(text="Обработка прошла успешно!")
    label3.pack()

window = tk.Tk()
window.title("RERAD")
# window.iconbitmap('svg.ico')
window.geometry('400x250')

btn = tk.Button(window, \
text="Выберите файл",\
bg="blue", \
fg="white", \
command=clicked)
btn.pack()
window.mainloop()