import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ==========================================
# PROBLEMA 2 - Corrección de multiple choice
# ==========================================

"""
n = int(input("Ingrese un número del 1 al 5: "))
while n not in range(1,6):
	n = int(input("Error: ingrese un número del 1 al 5: "))
img = cv2.imread(f'multiple_choice_{n}.png',cv2.IMREAD_GRAYSCALE)
img.shape
"""


img = cv2.imread(f'multiple_choice_1.png',cv2.IMREAD_GRAYSCALE)
img.shape






# ITEM A
# Usamos la gráfica para delimitar la zona de las preguntas
#plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
#plt.show()

index_respuestas = {"y1":140, "y2":1038}
img_respuestas = img[index_respuestas["y1"]:index_respuestas["y2"]]
# Estos arreglos los hacemos porque no podíamos identificar bien las "A" por columna
img_respuestas = np.where(img_respuestas > 49, 255, img_respuestas)
img_respuestas = np.where(img_respuestas < 50, 0, img_respuestas)
plt.figure(), plt.imshow(img_respuestas, cmap='gray'), plt.show(block=False)
plt.show()


print(f"Tamaño de imagen donde están las respuestas: {img_respuestas.shape}")


black = 0
img_zeros = img_respuestas==black
                           
img_row_zeros = img_zeros.any(axis=1)
img_row_zeros_idxs = np.argwhere(img_zeros.any(axis=1))

#plt.figure(), plt.imshow(escrito, cmap='gray'), plt.show()
#plt.show()

x = np.diff(img_row_zeros)          
renglones_indxs = np.argwhere(x)
print(f"Cantidad de renglones: {int(len(renglones_indxs)/2)}")


# Visualización
"""
ii = np.arange(0,len(renglones_indxs),2)    # 0 2 4 ... X --> X es el último nro par antes de len(renglones_indxs)
renglones_indxs[ii]+=1

xri = np.zeros(img_respuestas.shape[0])
xri[renglones_indxs] = (img_respuestas.shape[1]-1)
yri = np.arange(img_respuestas.shape[0])            
plt.figure(), plt.imshow(img_respuestas, cmap='gray'), plt.plot(xri, yri, 'r'), plt.title("Renglones - Inicio y Fin"), plt.show()
"""

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

# Exploramos el dato y lo visualizamos.
#print(renglones[0])
#plt.figure(), plt.imshow(renglones[24]["img"], cmap='gray'), plt.show()




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

def letra_seleccionada(img):
    total_pixeles = img.size
    pixeles_cero = np.count_nonzero(img == 0)
    
    # Calcular el porcentaje de píxeles iguales a cero
    porcentaje_cero = (pixeles_cero / total_pixeles) * 100
    if porcentaje_cero >= 15:
        return False
    else:
        return True

# Nos quedamos con todas las "letras" que tienen anchura mayor que 18 (en general tienen 20 píxeles de ancho)
# Y decidimos si está seleccionada (True) o no (False) según qué porcentaje tiene de píxeles negros.
letras_ok = []
for i in letras:
    if i["cord"][3]-i["cord"][1]>17:
        letras_ok.append(i)
        i["seleccionada"] = letra_seleccionada(i["img"])
        #print(i["cord"])
        #print(i["seleccionada"])

"""
# Visualización
from matplotlib.patches import Rectangle        # Matplotlib posee un módulo para dibujar rectángulos (https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)
plt.figure(), plt.imshow(img_respuestas, cmap='gray')
for il, letra in enumerate(letras_ok):
    yi = letra["cord"][0]                       # Rectangle() toma como entrada
    xi = letra["cord"][1]                       # las coordenadas (x,y) de la esquina superior izquierda, 
    W = letra["cord"][2] -letra["cord"][0]      # el ancho y el alto.
    H = letra["cord"][3] -letra["cord"][1]      #
    rect = Rectangle((xi,yi), H, W, linewidth=1, edgecolor='r', facecolor='none')    # Creamos el objeto rectángulo.
    ax = plt.gca()          # Obtengo el identificador de los ejes de la figura (handle)...
    ax.add_patch(rect)      # ... Agrego el objeto (patch) a los ejes.
plt.show()
"""

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



"""
# Visualización con identificación de columnas
from matplotlib.patches import Rectangle        # Matplotlib posee un módulo para dibujar rectángulos (https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)
plt.figure(), plt.imshow(img_respuestas, cmap='gray')
for il, letra in enumerate(letras_ok):
    yi = letra["cord"][0]                       # Rectangle() toma como entrada
    xi = letra["cord"][1]                       # las coordenadas (x,y) de la esquina superior izquierda, 
    W = letra["cord"][2] -letra["cord"][0]      # el ancho y el alto.
    H = letra["cord"][3] -letra["cord"][1]      #
    rect = Rectangle((xi,yi), H, W, linewidth=1, edgecolor='r', facecolor='none')    # Creamos el objeto rectángulo.
    ax = plt.gca()          # Obtengo el identificador de los ejes de la figura (handle)...
    ax.add_patch(rect)      # ... Agrego el objeto (patch) a los ejes.

plt.axvline(x=punto_medio_A, color='b', linestyle='--')
plt.axvline(x=punto_medio_B, color='b', linestyle='--')
plt.axvline(x=punto_medio_C, color='b', linestyle='--')
plt.axvline(x=punto_medio_D, color='b', linestyle='--')
plt.axvline(x=punto_medio_E, color='b', linestyle='--')
plt.show()
"""

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



"""
# Visualización
from matplotlib.patches import Rectangle        # Matplotlib posee un módulo para dibujar rectángulos (https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)
plt.figure(), plt.imshow(img_respuestas, cmap='gray')
for il, letra in enumerate(letras_ok):
    yi = letra["cord"][0]                       # Rectangle() toma como entrada
    xi = letra["cord"][1]                       # las coordenadas (x,y) de la esquina superior izquierda, 
    W = letra["cord"][2] -letra["cord"][0]      # el ancho y el alto.
    H = letra["cord"][3] -letra["cord"][1]      #
    rect = Rectangle((xi,yi), H, W, linewidth=1, edgecolor='r', facecolor='none')    # Creamos el objeto rectángulo.
    ax = plt.gca()          # Obtengo el identificador de los ejes de la figura (handle)...
    ax.add_patch(rect)      # ... Agrego el objeto (patch) a los ejes.
plt.show()

"""

# Vemos cómo quedaron nuestros datos:
for i in range(10):
    print(renglones[i]["respuestas"])

def count_false(dic):
    sum = 0
    for i in dic.values():
        if i is False:
            sum = sum+1
    return sum

respuestas_correctas = ["A", "A", "B", "A", "D", "B", "B", "C", "B", "A", "D", "A", "C", "C", "D", "B", "A", "C", "C", "D", "B", "A", "C", "C", "C"]
cantidad_de_correctas = 0

# Ahora sí, respondemos al enunciado:
for i in range(25):
    if renglones[i]["respuestas"][respuestas_correctas[i]]==True and count_false(renglones[i]["respuestas"])==4:
        print(f"Pregunta {renglones[i]["ir"]}: OK")
        cantidad_de_correctas = cantidad_de_correctas+1
    else:
        print(f"Pregunta {renglones[i]["ir"]}: MAL")

print(f"Cantidad de respuestas correctas: {cantidad_de_correctas}")
if cantidad_de_correctas>=20:
    print("Examen aprobado.")
else:
    print("Examen desaprobado.")









"""
# ITEM B (NOS PUSIMOS CREATIVOS)
# Exploración de imágenes


# Usamos la gráfica para delimitar el renglón con la información relevante
#plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
#plt.show()

# Indexamos el renglón, y el resto de datos en ese renglón
index_renglon = {"y1":109, "y2":129}
index_name = {"x1":98, "x2":280}
index_id = {"x1":332, "x2":430}
index_code = {"x1":432, "x2":565}
index_date = {"x1":650, "x2":767}


# Verificamos que los recortes funcionen
img_renglon = img[index_renglon["y1"]:index_renglon["y2"]]
#plt.figure(), plt.imshow(img_renglon, cmap='gray'), plt.show(block=False)
#plt.show()

img_name = img[index_renglon["y1"]:index_renglon["y2"],index_name["x1"]:index_name["x2"]]
img_id = img[index_renglon["y1"]:index_renglon["y2"],index_id["x1"]:index_id["x2"]]
# A code le dejamos el título, sino la información de la imagen era muy poca y no se podía leer.
img_code = img[index_renglon["y1"]:index_renglon["y2"],index_code["x1"]:index_code["x2"]]
img_date = img[index_renglon["y1"]:index_renglon["y2"],index_date["x1"]:index_date["x2"]]

img_datos = [img_name,img_id,img_code,img_date]
for i in range(4):
	#plt.subplot(1, 4, i+1)
	#plt.imshow(img_datos[i], cmap='gray')
    pass
#plt.show()

# Usaremos PyTesseract y Poppler (adaptación de C++) como OCR (optical characters recognition)
# para ayudarnos a leer las imágenes.


# Directorio para guardar imagenes auxiliares
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

ruta_imagen = os.path.join(carpeta, 'id.png')
plt.imsave(ruta_imagen, img_id, cmap='gray')
imagen = Image.open(ruta_imagen)
id = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()
# A veces aparecían comas al principio o al final de id
id = id.lstrip(",")
id = id.rstrip(",")

ruta_imagen = os.path.join(carpeta, 'code.png')
plt.imsave(ruta_imagen, img_code, cmap='gray')
imagen = Image.open(ruta_imagen)
code = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()
code = code[-1]

ruta_imagen = os.path.join(carpeta, 'date.png')
plt.imsave(ruta_imagen, img_date, cmap='gray')
imagen = Image.open(ruta_imagen)
date = pytesseract.image_to_string(imagen, lang="eng+spa+fra").strip()



if len(name.split())==2 and len(name)<=25:
    print(f"Nombre ({name}): OK")
else:
    print(f"Nombre ({name}): MAL")
if len(id.split())==1 and len(id)==8:
    print(f"Id ({id}): OK")
else:
    print(f"Id ({id}): MAL")
if len(code)==1:
    print(f"Code ({code}): OK")
else:
    print(f"Code ({code}): MAL")
if len(date.split())==1 and len(date)==8:
    print(f"Date ({date}): OK")
else:
    print(f"Date ({date}): MAL")

	"""






# ITEM C
# Se hace ejecutando el input del principio sobre el item A.





"""

# ITEM D 
from TP_2_as_function import corregir
from PIL import Image, ImageDraw, ImageFont

lista = []
for i in range(1,6):
    img = cv2.imread(f'multiple_choice_{i}.png',cv2.IMREAD_GRAYSCALE)
    entry = corregir(img)
    lista.append(entry)

df = pd.DataFrame(lista)    

# Crear una imagen vacía
width, height = 600, 300
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Escribir el DataFrame en la imagen
font = ImageFont.truetype("arial.ttf", 20)
text = df.to_string(index=False)  # No incluir el índice en la tabla
lines = text.split('\n')
y_pos = 10


for line in lines:
    if 'No' in line:  # Colorear las filas con 'No' en rojo
        draw.text((10, y_pos), line, fill='red', font=font)
    else:
        draw.text((10, y_pos), line, fill='black', font=font)
    y_pos += 20

# Guardar la imagen como un archivo PNG
img.save('tabla.png')


"""









# ITEM B 2.0
# Exploración de imágenes


# Usamos la gráfica para delimitar el renglón con la información relevante
#plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)
#plt.show()

import cv2

# Función para contar componentes conectadas y sus separaciones horizontales
def count_chars_and_spaces(img):
    _, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Obtener las posiciones horizontales y centroids de los caracteres con área mínima 40
    positions = []
    centroids_filtered = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 3:
            positions.append((x, y, w, h))
            centroids_filtered.append(centroids[i])

    # Dibujar las cajas alrededor de los caracteres detectados
    for (x, y, w, h) in positions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Contar caracteres y espacios basados en la distancia entre centroides
    num_chars = len(positions)
    num_spaces = 0
    
    # Calcular las distancias entre centroides y posiciones
    for i in range(1, len(centroids_filtered)):
        distancia1 = centroids_filtered[i][0] - centroids_filtered[i-1][0]
        distancia2 = positions[i][0] - (positions[i-1][0] + positions[i-1][2])
        promedio = (distancia1 + distancia2) / 2
        if promedio > 21:
            num_spaces += 1
    
    return num_chars, num_spaces, positions

# Función para verificar las condiciones especificadas
def verificar_condiciones(img_datos):
    resultados = {
        "name": False,
        "id": False,
        "code": False,
        "date": False
    }

    # Verificación de name
    img_name = img_datos[0]
    num_chars_name, num_spaces_name, _ = count_chars_and_spaces(img_name)
    if num_chars_name <= 25 and num_spaces_name == 1:
        resultados["name"] = True
        print("Name: OK")
    else:
        print("Name: MAL")
        print(f"El nombre tiene {num_chars_name} caracter(es) y {num_spaces_name} espacio(s).")

    # Verificación de id
    img_id = img_datos[1]
    num_chars_id, num_spaces_id, positions_id = count_chars_and_spaces(img_id)
    if num_chars_id == 8 and num_spaces_id == 0:
        resultados["id"] = True
        print("Id: OK")
    else:
        print("Id: MAL")
        print(f"El id tiene {num_chars_id} caracter(es) y {num_spaces_id} espacio(s).")

    # Verificación de code
    img_code = img_datos[2]
    num_chars_code, num_spaces_code, _ = count_chars_and_spaces(img_code)
    if num_chars_code == 1 and num_spaces_code == 0:
        resultados["code"] = True
        print("Code: OK")
    else:
        print("Code: MAL")
        print(f"El code tiene {num_chars_code} caracter(es) y {num_spaces_code} espacio(s).")

    # Verificación de date
    img_date = img_datos[3]
    num_chars_date, num_spaces_date, _ = count_chars_and_spaces(img_date)
    if num_chars_date == 8 and num_spaces_date == 0:
        resultados["date"] = True
        print("Date: OK")
    else:
        print("Date: MAL")
        print(f"La fecha tiene {num_chars_date} caracter(es) y {num_spaces_date} espacio(s).")

    return resultados, positions_id

# Segmentación y verificación de imágenes
for i in range(1, 6):
    img = cv2.imread(f'multiple_choice_{i}.png', cv2.IMREAD_GRAYSCALE)
    
    # Indexamos el renglón, y el resto de datos en ese renglón
    index_renglon = {"y1": 109, "y2": 129}
    index_name = {"x1": 98, "x2": 280}
    index_id = {"x1": 332, "x2": 430}
    index_code = {"x1": 490, "x2": 565}
    index_date = {"x1": 650, "x2": 767}

    img_name = img[index_renglon["y1"]:index_renglon["y2"], index_name["x1"]:index_name["x2"]]
    img_id = img[index_renglon["y1"]:index_renglon["y2"], index_id["x1"]:index_id["x2"]]
    img_code = img[index_renglon["y1"]:index_renglon["y2"], index_code["x1"]:index_code["x2"]]
    img_date = img[index_renglon["y1"]:index_renglon["y2"], index_date["x1"]:index_date["x2"]]

    img_datos = [img_name, img_id, img_code, img_date]
    
    print(f"Chequeo de imagen {i}: ")
    # Verificamos las condiciones
    resultados, positions_id = verificar_condiciones(img_datos)
    print("#####################################")
