import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def letra_seleccionada(img):
        total_pixeles = img.size
        pixeles_cero = np.count_nonzero(img == 0)
        
        # Calcular el porcentaje de píxeles iguales a cero
        porcentaje_cero = (pixeles_cero / total_pixeles) * 100
        if porcentaje_cero >= 15:
            return False
        else:
            return True

def count_false(dic):
        sum = 0
        for i in dic.values():
            if i is False:
                sum = sum+1
        return sum

def corregir(img):
    index_respuestas = {"y1":140, "y2":1038}
    img_respuestas = img[index_respuestas["y1"]:index_respuestas["y2"]]
    
    # Estos arreglos los hacemos porque no podíamos identificar bien las "A" por columna
    img_respuestas = np.where(img_respuestas > 49, 255, img_respuestas)
    img_respuestas = np.where(img_respuestas < 50, 0, img_respuestas)

    black = 0
    img_zeros = img_respuestas==black
                            
    img_row_zeros = img_zeros.any(axis=1)
    img_row_zeros_idxs = np.argwhere(img_zeros.any(axis=1))

    x = np.diff(img_row_zeros)          
    renglones_indxs = np.argwhere(x)

    # Genero estructura de datos para guardar datos de renglones
    r_idxs = np.reshape(renglones_indxs, (-1,2))
    renglones = []
    for ir, idxs in enumerate(r_idxs):
        renglones.append({
            "ir": ir+1,
            "cord": idxs,
            "punto_medio":idxs[0] + int(abs(idxs[0]-idxs[1])/2),
            "img": img_respuestas[idxs[0]:idxs[1], :]
        })

    # COLUMNAS
    # Busquemos ahora inicio y fin de cada "columna" (número de pregunta, punto, A, B, C, D y E)

    letras = []
    il = -1
    for ir, renglon in enumerate(renglones):
        renglon_zeros = renglon["img"]==0

        # Analizo columnas del renglón 
        ren_col_zeros = renglon_zeros.any(axis=0)
        ren_col_zeros_idxs = np.argwhere(renglon_zeros.any(axis=0))
            
        # Encontramos inicio y final de cada letra
        x = np.diff(ren_col_zeros)
        letras_indxs = np.argwhere(x) 
        # *** Modifico índices ***********
        ii = np.arange(0,len(letras_indxs),2)
        letras_indxs[ii]+=1

        letras_indxs = letras_indxs.reshape((-1,2))
        
        for irl, idxs in enumerate(letras_indxs):
            il+=1
            letras.append({
                "ir":ir+1,
                "irl":irl+1,
                "il": il,
                "cord": [renglon["cord"][0], idxs[0], renglon["cord"][1], idxs[1]],
                "punto_medio":idxs[0] + int(abs(idxs[0]-idxs[1])/2),
                "img": renglon["img"][:, idxs[0]:idxs[1]]
            })

    # Nos quedamos con todas las "letras" que tienen anchura mayor que 18 (en general tienen 20 píxeles de ancho)
    # Y decidimos si está seleccionada (True) o no (False) según qué porcentaje tiene de píxeles negros.
    letras_ok = []
    for i in letras:
        if i["cord"][3]-i["cord"][1]>17:
            letras_ok.append(i)
            i["seleccionada"] = letra_seleccionada(i["img"])
        

    # Calculamos dónde empiezan las "A"

    min = letras_ok[0]["cord"][1]
    for letra in letras_ok:
        if min>letra["cord"][1]:
            min = letra["cord"][1]

    # Y los puntos medios de las columnas
    punto_medio_A = min+9
    punto_medio_B = punto_medio_A+29
    punto_medio_C = punto_medio_B+29
    punto_medio_D = punto_medio_C+29
    punto_medio_E = punto_medio_D+29


    for r in renglones:
        respuestas = {}
        for i in letras_ok:
            if i["cord"][0]<r["punto_medio"] and i["cord"][2]>r["punto_medio"]:
                if i["cord"][1]<punto_medio_A and i["cord"][3]>punto_medio_A:
                    respuestas["A"] = i["seleccionada"]
                elif i["cord"][1]<punto_medio_B and i["cord"][3]>punto_medio_B:
                    respuestas["B"] = i["seleccionada"]
                elif i["cord"][1]<punto_medio_C and i["cord"][3]>punto_medio_C:
                    respuestas["C"] = i["seleccionada"]
                elif i["cord"][1]<punto_medio_D and i["cord"][3]>punto_medio_D:
                    respuestas["D"] = i["seleccionada"]
                elif i["cord"][1]<punto_medio_E and i["cord"][3]>punto_medio_E:
                    respuestas["E"] = i["seleccionada"]
        
        LETRAS = ["A", "B", "C", "D", "E"]
        for l in LETRAS:
            if l not in respuestas:
                respuestas[l] = True
        r["respuestas"]=respuestas


    respuestas_correctas = ["A", "A", "B", "A", "D", "B", "B", "C", "B", "A", "D", "A", "C", "C", "D", "B", "A", "C", "C", "D", "B", "A", "C", "C", "C"]
    cantidad_de_correctas = 0


    entry = {}
    # Ahora sí, respondemos al enunciado:
    for i in range(25):
        if renglones[i]["respuestas"][respuestas_correctas[i]]==True and count_false(renglones[i]["respuestas"])==4:
            cantidad_de_correctas = cantidad_de_correctas+1
    
    if cantidad_de_correctas>=20:
        entry["aprobado"]="Sí"
    else:
        entry["aprobado"]="No"
    



    # TOMAR NOMBRE
    # Indexamos el renglón, y el resto de datos en ese renglón
    index_renglon = {"y1":109, "y2":129}
    index_name = {"x1":98, "x2":280}

    img_name = img[index_renglon["y1"]:index_renglon["y2"],index_name["x1"]:index_name["x2"]]

    carpeta = 'images'
    # Crear la carpeta si no existe
    import os
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Guardar la imagen y el dato que allí leemos con el procesador óptico
    ruta_imagen = os.path.join(carpeta, 'name.png')
    plt.imsave(ruta_imagen, img_name, cmap='gray')
    imagen = Image.open(ruta_imagen)
    name = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()

    entry["nombre"]=name
    return entry