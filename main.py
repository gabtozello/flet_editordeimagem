import cv2
import numpy as np
import os.path
import flet as ft
from flet import *
import sys
import random
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
previous_image_path = None

def main(page: ft.Page):
    page.scroll = "none"

    def uploadnow(e: FilePickerFileType):
        global previous_image_path
        app = QApplication(sys.argv)
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.png *.jpeg)")

        if file_path:
            file_name = os.path.basename(file_path)
            pixmap = QPixmap(file_path)
            save_path = os.path.join("./myUploads/", file_name)
            pixmap.save(save_path)
            print(f"Image saved to {save_path}")

            if previous_image_path != save_path:
                previous_image_path = save_path
                image(save_path)

            sys.exit(app.exec_())

    file_picker = ft.FilePicker(on_result=uploadnow)
    page.overlay.append(file_picker)

    page.appbar = ft.AppBar(
        leading=ft.IconButton(
            icon=icons.ADD_A_PHOTO,
            on_click=uploadnow,
        ),
        title=ft.Text("Editor de fotos"),
        center_title=True,
        bgcolor=ft.colors.SURFACE_VARIANT,
        actions=[ft.IconButton(
            icon=icons.DOWNLOAD,
            on_click=lambda _: save_image('temporario.jpg'),
        )]
        )

    page.add(file_picker)


    def image(save_path):
        print("Received save path:", save_path)
        imagem = ft.Container(
                    content=ft.Image(src=save_path, fit='contain'),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.top_center,
                    width=400,
                    height=400,
                    border_radius=10,
                )
        images = ft.Row(width=600, wrap=False, scroll="always")
        for i in range(1, 21):
            button_name = f"{i}"
            button_image = ft.Image(
                src=f"filtrosExemplos/exemplo{i}.jpg",
                width=80,
                height=150,
                fit=ft.ImageFit.COVER,
                repeat=ft.ImageRepeat.NO_REPEAT,
                border_radius=ft.border_radius.all(10),
            )
            button_container = ft.Container(
                on_click=lambda e, name=button_name: aplicarFiltro(name, save_path),
                content=button_image,
                alignment=ft.alignment.center,
                margin=10,
            )
            images.controls.append(button_container)
        page.clean()
        page.add(ft.Column([imagem, images]))


    def aplicarFiltro(button_name, save_path):
        print(f"Button clicked: {button_name}")
        print(f"Recebido:{save_path}")
        if not hasattr(aplicarFiltro, 'original_save_path'):
            aplicarFiltro.original_save_path = save_path
        save_path = aplicarFiltro.original_save_path
        print(save_path)
        img = cv2.imread(save_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Filtro 1: imagem Equalizada do gray
        if button_name == '1':
           img_edit = cv2.equalizeHist(img_g)
        # Filtro 2: Filtro vermelho
        elif button_name == '2':
            red_filter = np.zeros_like(img)
            red_filter[:, :, 2] = img[:, :, 2]
            img_edit = red_filter
        # Filtro 3: Filtro verde
        elif button_name == '3':
            print('ta aqui no 3')
            green_filter = np.zeros_like(img)
            green_filter[:, :, 1] = img[:, :, 1]
            img_edit = green_filter
        # Filtro 4:
        elif button_name == '4':
            sobelx = cv2.Sobel(img_g, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_g, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            img_edit = magnitude
        # Filtro 5: Filtro relevo colorido
        elif button_name == '5':
            emboss_filter = np.array([[-2, -1, 0],
                                      [-1, 1, 1],
                                      [0, 1, 2]])
            embossed = cv2.filter2D(img, -1, emboss_filter)
            img_edit = embossed
        # Filtro 6: Filtro relevo com imagem em gradiente
        elif button_name == '6':

            emboss_filter = np.array([[-2, -1, 0],
                                      [-1, 1, 1],
                                      [0, 1, 2]])

            embossed = cv2.filter2D(img_g, -1, emboss_filter)
            color_start = (255, 20, 147)  # Rosa
            color_end = (255, 165, 0)    # Laranja
            gradient_size = 256
            gradient = np.zeros((gradient_size, 1, 3), dtype=np.uint8)
            for i in range(gradient_size):
                r = int(color_start[0] + (color_end[0] - color_start[0]) * i / gradient_size)
                g = int(color_start[1] + (color_end[1] - color_start[1]) * i / gradient_size)
                b = int(color_start[2] + (color_end[2] - color_start[2]) * i / gradient_size)
            gradient[i] = (r, g, b)
            embossed_color = cv2.applyColorMap(embossed, gradient)
            img_edit = embossed_color
        # Filtro 7: Filtro de relevo
        elif button_name == '7':
            kernel_emboss = np.array([[-2, -1, 0],
                                 [-1,  1, 1],
                                 [ 0,  1, 2]])
            img_emboss = cv2.filter2D(img_g, -1, kernel_emboss)
            img_emboss = np.clip(img_emboss, 0, 255).astype(np.uint8)
            img_edit = img_emboss
        # Filtro 8: Cada parte da imagem de uma cor
        elif button_name == '8':
            # Divide a imagem em 4
            height, width, _ = img.shape
            quadrant_width = width // 2
            quadrant_height = height // 2
            quadrant_width_adjusted = quadrant_width + width % 2
            quadrant_height_adjusted = quadrant_height + height % 2
            # Extrai os quadrantes da imagem original
            quadrant1 = img[0:quadrant_height_adjusted, 0:quadrant_width_adjusted]
            quadrant2 = img[0:quadrant_height_adjusted, quadrant_width:width]
            quadrant3 = img[quadrant_height:height, 0:quadrant_width_adjusted]
            quadrant4 = img[quadrant_height:height, quadrant_width:width]

            quadrant1[:, :, 1:] = 0  # Manter apenas o canal azul
            quadrant2[:, :, 0] = 0  # Manter apenas o canal verde e vermelho
            quadrant3[:, :, 2:] = 0  # Manter apenas o canal azul
            quadrant4[:, :, :2] = 0  # Manter apenas o canal verde e vermelho
            # Cria uma imagem vazia do tamanho do quadrado composto
            square_image = np.zeros((height, width, 3), dtype=np.uint8)
            # Insere os quadrantes na imagem vazia
            square_image[0:quadrant_height_adjusted, 0:quadrant_width_adjusted] = quadrant1
            square_image[0:quadrant_height_adjusted, quadrant_width:width] = quadrant2
            square_image[quadrant_height:height, 0:quadrant_width_adjusted] = quadrant3
            square_image[quadrant_height:height, quadrant_width:width] = quadrant4
            img_edit = square_image
        # Filtro 9: Filtro negativo
        elif button_name == '9':
            img_edit = cv2.bitwise_not(img)
        # Filtro 10: Textura de madeira
        elif button_name == '10':
            emboss_filter = np.array([[-2, -1, 0],
                                 [-1, 1, 1],
                                 [0, 1, 2]])
            embossed = cv2.filter2D(img, -1, emboss_filter)
            wood_texture = cv2.imread('imagens/textura_de_madeira.jpg')
            wood_texture = cv2.cvtColor(wood_texture, cv2.COLOR_RGB2BGR)
            # Redimensionar a textura para o tamanho da imagem original
            wood_texture_resized = cv2.resize(wood_texture, (embossed.shape[1], embossed.shape[0]))
            alpha = 0.5
            embossed = cv2.cvtColor(embossed, cv2.COLOR_RGB2GRAY)
            embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
            # Aplicar a textura de madeira na imagem original
            wood_filter = cv2.addWeighted(embossed, 1-alpha, wood_texture_resized, alpha, 0)
            img_edit = wood_filter
        # Filtro 11:
        elif button_name == '11':
            img = cv2.bitwise_not(img)
            img_edit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Filtro 12: Bordas de fogo
        elif button_name == '12':
            sobelx = cv2.Sobel(img_g, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_g, cv2.CV_64F, 0, 1, ksize=3)
            # Calcula a magnitude do gradiente
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Normaliza a magnitude
            magnitude_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
            # Aplica o efeito de blend nas bordas
            blended = cv2.applyColorMap((magnitude_normalized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            img_edit = blended
        # Filtro 13: Bordas em roxo
        elif button_name == '13':
             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,0)
             sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
             sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
             magnitude = cv2.magnitude(sobelx, sobely)
             _, threshold = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
             image_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
             image_color[np.where(threshold != 0)] = [250, 0, 100]
             img_edit = image_color
        # Filtro 14: Filtro desenho
        elif button_name == '14':
             laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
             sketch = 255 - laplacian
             img_edit = sketch
        # Filtro 15: Filtro azul
        elif button_name == '15':
             blue_filter = np.zeros_like(img)
             blue_filter[:, :, 0] = img[:, :, 0]
             img_edit = blue_filter
        # Filtro 16: Imagens com cores diferents cada uma
        elif button_name == '16':
             height, width, _ = img.shape
             # Cria uma imagem quadrada com quatro partes iguais
             square_image = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
             # Cria um filtro verde
             green_filter = np.zeros_like(img)
             green_filter[:, :, 1] = img[:, :, 1]
             # Define a cor roxa
             purple_color = (128, 0, 128)
             # Aplica a cor roxa ao filtro
             purple_filter = np.zeros_like(img) + purple_color
             purple_filter[:, :, 1] = green_filter[:, :, 1]
             # Defin a cor amarela
             yellow_color = (0, 255, 255)
             # Aplicar a cor amarela ao filtro
             yellow_filter = np.zeros_like(img) + yellow_color
             yellow_filter[:, :, :2] = green_filter[:, :, :2]
             # Azul
             blue_filter = np.zeros_like(img)
             blue_filter[:, :, 0] = img[:, :, 1]
             # Vermelho
             red_filter = np.zeros_like(img)
             red_filter[:, :, 2] = img[:, :, 2]
             # Preencher cada parte do quadrado com a imagem original
             square_image[0:height, 0:width] = purple_filter
             square_image[0:height, width:(2 * width)] = yellow_filter
             square_image[height:(2 * height), 0:width] = blue_filter
             square_image[height:(2 * height), width:(2 * width)] = red_filter
             img_edit = square_image
        # Filtro 17:
        elif button_name == '17':
             sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
             sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
             magnitude = np.sqrt(sobelx**2 + sobely**2)
             img_edit = magnitude
        # Filtro 18:
        elif button_name == '18':
             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
             kernel_size = random.randint(20, 40)
             img_edit = cv2.blur(img, (kernel_size, kernel_size))
        # Filtro 19:
        elif button_name == '19':
             granulacao = 500
             noise = np.zeros_like(img, dtype=np.int16)
             cv2.randn(noise, -granulacao, granulacao)
             img_edit = np.clip(img + noise, 50, 255).astype(np.uint8)
        # Filtro 20:
        elif button_name == '20':
            tamanho_kernel = 5
            intensidade_bordas = 100
            blurred = cv2.GaussianBlur(img, (tamanho_kernel, tamanho_kernel), 0)
            edges = cv2.Canny(blurred, intensidade_bordas, intensidade_bordas * 2)
            img_bordas = cv2.bitwise_and(img, img, mask=edges)
            img_edit = cv2.addWeighted(img, 0.7, img_bordas, 0.3, 0)
        page.update()
        output_path = 'temporario.jpg'
        cv2.imwrite(output_path, img_edit)
        image(output_path)


    def save_image(output_path):
        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Salvar Imagem")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Arquivos JPEG (*.jpg);;Arquivos PNG (*.png);;Todos os Arquivos (*)")
        if file_dialog.exec_() == QFileDialog.Accepted:
           file_path = file_dialog.selectedFiles()[0]
           image = cv2.imread(output_path)
           cv2.imwrite(file_path, image)
ft.app(target=main)
