import shutil
import sys
import os
import subprocess
import time
from basicsr.archs.rrdbnet_arch import RRDBNet

import cv2
from PIL import Image

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QWidget, QLineEdit, QComboBox, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar, \
    QMessageBox, QCheckBox

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer, QRectF
import datetime
from PySide6.QtCore import QThread, Signal
from realesrgan import RealESRGANer

from gfpgan import GFPGANer

folder_to_delete = "tmp_img"

import urllib.request

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

# Función para descargar modelos de GFPGAN según la selección
def download_gfpgan_model(version):
    model_urls = {
        "GFPGANCleanv1-NoCE-C2": "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
        "GFPGANv1.3": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "RestoreFormer": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    }
    model_path = f"{version}.pth"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_urls[version], model_path)
    return model_path



# # RealESRGAN Model Download
# realesrgan_model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
# realesrgan_model_path = 'RealESRGAN_x2plus.pth'
# download_model(realesrgan_model_url, realesrgan_model_path)


class EnhanceThread_Upscale(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(str)

    def __init__(self, command, output_file_path):
        super().__init__()
        self.command = command
        self.output_file_path = output_file_path

    def run(self):
        startupinfo = None
        if sys.platform == 'win32':  # Solo si el sistema operativo es Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # Evita que se muestre la ventana
            startupinfo.wShowWindow = subprocess.SW_HIDE  # Esto esencialmente esconde la ventana
        print(self.command)

        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW  # Esto previene que se cree la ventana
        )
        while process.poll() is None:
            line = process.stdout.readline().strip()
            # print(line)
            if "%" in line:
                progress = float(line.split('%')[0].replace(',', '.'))
                self.progress_signal.emit(progress)  # Emitiendo señal de progreso
        process.wait()
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
        self.progress_signal.emit(100)  # Emitiendo señal de progreso completo
        self.finished_signal.emit(self.output_file_path)  # Emitiendo señal de finalización con la ruta de archivo de salida




class EnhanceThread(QThread):
    progress_signal = Signal(int)
    finished_signal = Signal(str)

    def __init__(self, input_file_path, output_file_path, gfpgan_options):
        super().__init__()
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.gfpgan_options = gfpgan_options

        # Initialize GFPGAN model
        self.gfpgan_restorer = GFPGANer(
            model_path=gfpgan_options['model_path'],
            upscale=gfpgan_options['upscale'],
            arch=gfpgan_options['arch'],
            channel_multiplier=gfpgan_options['channel_multiplier'],
            bg_upsampler=None  # Set to None because we will use RealESRGAN separately
        )


    def run(self):
        # Read input image
        img = cv2.imread(self.input_file_path)
        self.progress_signal.emit(25)  # Emit progress complete signal


        # Use GFPGAN for face enhancement
        _, _, restored_img = self.gfpgan_restorer.enhance(img, has_aligned=self.gfpgan_options['aligned'],
                                                           only_center_face=self.gfpgan_options['only_center_face'],
                                                           paste_back=True)
        self.progress_signal.emit(75)  # Emit progress complete signal

        # Save the enhanced image
        cv2.imwrite(self.output_file_path, restored_img)
        print("Acabo de escribir la imagen")

        # Emit finished signal
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
        self.progress_signal.emit(100)  # Emit progress complete signal
        self.finished_signal.emit(self.output_file_path)  # Emit finished signal with output file path

class ImageLabel(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self._pixmap = None

    def set_image(self, pixmap_or_path):
        if isinstance(pixmap_or_path, QPixmap):
            self._pixmap = pixmap_or_path
        else:
            self._pixmap = QPixmap(pixmap_or_path)

        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._image_item.setPixmap(scaled_pixmap)
            self.setSceneRect(scaled_pixmap.rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()



class RealESRGANApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("aliscaller App")
        self.setGeometry(100, 100, 800, 600)


        # Widgets
        self.original_image_label = ImageLabel()
        self.enhanced_image_label = ImageLabel()


        self.select_button = QPushButton("Selecciona una Image", self)
        self.select_button.clicked.connect(self.select_file)

        self.clear_button = QPushButton("Limpiar Image", self)
        self.clear_button.clicked.connect(self.clear_image_label)

        self.model_label = QLabel("Modelo:", self)
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["GFPGANCleanv1-NoCE-C2", "GFPGANv1.3", "GFPGANv1.4", "RestoreFormer"])
        self.model_selector.setCurrentIndex(0)  # Establecer realesrgan-x4plus como seleccionado por defecto

        self.model_label_enhance = QLabel("Modelo:", self)
        self.model_selector_enhance = QComboBox(self)
        self.model_selector_enhance.addItems(["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"])
        self.model_selector_enhance.setCurrentIndex(1)  # Establecer realesrgan-x4plus como seleccionado por defecto

        self.use_realersGAN = QCheckBox("RealersGAN", self)
        self.use_gfpgan = QCheckBox("GFPGAN", self)
        self.use_gfpgan.setChecked(True)
        # Conecta las señales stateChanged a una función que ajuste el estado del otro QCheckBox
        self.use_realersGAN.stateChanged.connect(self.realersgan_changed)
        self.use_gfpgan.stateChanged.connect(self.gfgan_changed)

        self.format_label = QLabel("Formato:", self)
        self.format_selector = QComboBox(self)
        self.format_selector.addItems(["jpg", "png", "webp"])
        self.format_selector.setCurrentIndex(0)  # Establecer realesrgan-x4plus como seleccionado por defecto

        self.scale_label = QLabel("Escala:", self)
        self.scale_label.setToolTip("Escala solo funciona con realesr-animevideov3")
        self.scale_input = QLineEdit(self)
        self.scale_input.setToolTip("Escala solo funciona con realesr-animevideov3")
        self.scale_input.setText("2")

        self.open_output_folder_button = QPushButton("Carpeta Salida", self)
        self.open_output_folder_button.clicked.connect(self.open_output_folder)

        self.enhance_button = QPushButton("Mejorar!!", self)
        self.enhance_button.clicked.connect(self.enhance_image)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)  # El rango será de 0 a 100
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Layout
        option_layout = QHBoxLayout()
        option_layout.addWidget(self.select_button)
        option_layout.addWidget(self.clear_button)  # Añadiendo el nuevo botón aquí
        # option_layout.addWidget(self.model_label)
        # self.model_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Etiqueta con tamaño fijo
        option_layout.addWidget(self.model_selector)
        self.model_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Selector expansible horizontalmente
        option_layout.addWidget(self.use_gfpgan)

        # option_layout.addWidget(self.model_label_enhance)
        # self.model_label_enhance.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Etiqueta con tamaño fijo
        option_layout.addWidget(self.model_selector_enhance)
        self.model_selector_enhance.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Selector expansible horizontalmente

        option_layout.addWidget(self.use_realersGAN)
        option_layout.addWidget(self.format_label)
        self.format_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Etiqueta con tamaño fijo
        option_layout.addWidget(self.format_selector)
        self.format_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Selector expansible horizontalmente


        button_layout = QHBoxLayout()
        button_layout.addWidget(self.scale_label)
        button_layout.addWidget(self.scale_input)
        button_layout.addWidget(self.enhance_button)
        button_layout.addWidget(self.open_output_folder_button)

        # Layout
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.original_image_label)
        images_layout.addWidget(self.enhanced_image_label)

        prog_bar_layout = QHBoxLayout()
        self.ali_laber = QLabel("<b><i>Hecho por Ali @learnwithaali ¡¡prohibida su venta!! </i></b>", self)
        prog_bar_layout.addWidget(self.progress_bar)
        prog_bar_layout.addWidget(self.ali_laber)


        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addLayout(option_layout)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(prog_bar_layout)

        container = QWidget()
        container.setLayout(main_layout)
        # Creando y configurando las etiquetas para superponerse sobre las vistas de imagen
        self.original_image_label_label = QLabel("Real", self.original_image_label)
        self.original_image_label_label.setStyleSheet("background-color: white;")  # Ajusta el estilo según necesites
        self.original_image_label_label.move(10, 10)  # Ajusta la posición según necesites

        self.enhanced_image_label_label = QLabel("Preview", self.enhanced_image_label)
        self.enhanced_image_label_label.setStyleSheet("background-color: white;")  # Ajusta el estilo según necesites
        self.enhanced_image_label_label.move(10, 10)  # Ajusta la posición según necesites


        self.setCentralWidget(container)

        self.selected_file = ""
        self.output_folder = "output"
        os.makedirs(self.output_folder, exist_ok=True)

    def realersgan_changed(self, state):
        print(state)
        # Si RealersGAN se marca, asegurarse de que GFPGAN esté desmarcado
        if state == Qt.Checked.value and self.use_gfpgan.isChecked():
            self.use_gfpgan.setChecked(False)

    def gfgan_changed(self, state):
        print(state)
        # Si GFPGAN se marca, asegurarse de que RealersGAN esté desmarcado
        if state == Qt.Checked.value and self.use_realersGAN.isChecked():
            self.use_realersGAN.setChecked(False)


    def enhance_image(self):
        if not self.selected_file:
            QMessageBox.warning(self, "Error", "No image selected")
            return


        # Primero verifica si se debe usar RealESRGAN
        if self.use_realersGAN.isChecked():
            self.enhance_image_with_realersgan(self.selected_file, None, None)
            return
        # Después verifica si se debe usar GFPGAN
        if self.use_gfpgan.isChecked():
            self.enhance_image_with_gfpgan(self.selected_file)
            return





    # Modificar la función enhance_image para adaptarla a la selección de GFPGAN
    def enhance_image_with_gfpgan(self,output_file_path_prev = None):
        if not self.selected_file:
            QMessageBox.warning(self, "Error", "No image selected")
            return

        selected_version = self.model_selector.currentText()
        gfpgan_model_path = download_gfpgan_model(selected_version)

        # Configuración específica del modelo seleccionado
        if selected_version in ["GFPGANCleanv1-NoCE-C2", "GFPGANv1.3", "GFPGANv1.4"]:
            arch = 'clean'
            channel_multiplier = 2
        elif selected_version == "RestoreFormer":
            arch = 'RestoreFormer'
            channel_multiplier = 2
        else:
            QMessageBox.warning(self, "Error", "Modelo GFPGAN no reconocido")
            return
        if self.scale_input.text() == "" or  int(self.scale_input.text()) < 2 or int(self.scale_input.text()) > 4:
            QMessageBox.warning(self, "Error", "Escala incorrecta, debe ser un número entre 2 y 4")
            return
        scaler = int(self.scale_input.text())


        # Opciones para GFPGAN
        gfpgan_options = {
            'model_path': gfpgan_model_path,
            'upscale': scaler,
            'arch': arch,
            'channel_multiplier': channel_multiplier,
            'only_center_face': False,
            'aligned': False
        }


        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_format = self.format_selector.currentText()
        output_file_path = os.path.join(self.output_folder, f"enhanced_{timestamp}.{output_format}")

        inp_img = self.selected_file
        if output_file_path_prev:
            inp_img = output_file_path_prev

        # Iniciar el proceso de mejora
        self.enhance_thread = EnhanceThread(inp_img, output_file_path, gfpgan_options)
        self.enhance_thread.progress_signal.connect(self.progress_bar.setValue)

        self.enhance_thread.finished_signal.connect(self.show_enhanced_image)
        self.enhance_thread.start()

    def enhance_image_with_realersgan(self, input_file_path, intermediate_file_path=None, final_output_file_path=None):
        if not self.selected_file:
            return
        if not self.use_realersGAN.isChecked():
            return


        model = self.model_selector_enhance.currentText()
        output_format = self.format_selector.currentText()

        scale = self.scale_input.text()
        if not scale.isdigit():
            scale = "2"  # Default value if the input is not valid
        scale = int(scale)
        if scale < 2:
            scale = 2
        elif scale > 4:
            scale = 4
        selected_model = self.model_selector_enhance.currentText()
        if "x4" in selected_model:
            scale = 4
        self.scale_input.setText(str(scale))

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        orignal_img_ext = self.selected_file.split(".")[-1]
        output_file_path = os.path.join(self.output_folder, f"output_{scale}_{timestamp}.{output_format}")




        # Execute RealESRGAN
        command = ["./realesrgan-ncnn-vulkan.exe", "-i", self.selected_file, "-o", output_file_path, "-s", str(scale),
                   "-n", model, "-f", output_format]

        if orignal_img_ext == "png" and (output_format == "jpg" or output_format == "webp"):
            os.makedirs(folder_to_delete, exist_ok=True)
            new_tmp_img_path = os.path.join(folder_to_delete, f"temp_{timestamp}.jpg")
            original_image = Image.open(self.selected_file)
            if original_image.mode == "RGBA":
                original_image = original_image.convert("RGB")
            # Guarda la imagen original como un archivo JPG
            original_image.save(new_tmp_img_path, "JPEG")


            # Actualiza self.selected_file para que apunte a la nueva imagen temporal
            command = ["./realesrgan-ncnn-vulkan.exe", "-i", new_tmp_img_path, "-o", output_file_path, "-s", str(scale),
                       "-n", model, "-f", output_format]



        self.enhance_thread = EnhanceThread_Upscale(command, output_file_path)
        self.enhance_thread.progress_signal.connect(self.progress_bar.setValue)  # Conectar la señal de progreso a la barra de progreso
        if self.use_gfpgan.isChecked():
            self.enhance_thread.finished_signal.connect(lambda: self.enhance_image_with_gfpgan(output_file_path))
        else:
            self.enhance_thread.finished_signal.connect(self.show_enhanced_image)  # Conectar la señal de finalización para mostrar la imagen mejorada
        self.enhance_thread.start()


    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, "Seleccione una imagen", "", "Images  (*.jpg *.png );;All Files (*)",
                                   options=options)
        if file_dialog.exec() == QFileDialog.Accepted:
            self.selected_file = file_dialog.selectedFiles()[0]
            self.original_image_label.set_image(self.selected_file)


    def clear_image_label(self):
        self.original_image_label._scene.clear()
        self.enhanced_image_label._scene.clear()

    def open_output_folder(self):
        output_folder_path = os.path.abspath(self.output_folder)
        subprocess.Popen(f'explorer {output_folder_path}')  # Esto abrirá la carpeta de salida en el explorador



    def show_enhanced_image(self, output_file_path):
        if os.path.exists(output_file_path):
            print("se ha hecho el enhance")
            # Cargar y mostrar la imagen original
            self.original_image_label.set_image(self.selected_file)

            # Cargar y mostrar la imagen mejorada
            self.enhanced_image_label.set_image(output_file_path)
        else:
            QMessageBox.warning(self, "Advertencia", "Lo mas probable es que la imagen si esté en la carpeta output")
            self.original_image_label.set_image(self.selected_file)
            # Cargar y mostrar la imagen mejorada
            self.enhanced_image_label.set_image(output_file_path)



    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            local_file_path = mime_data.urls()[0].toLocalFile()
            self.selected_file = local_file_path

            # Cargar y mostrar la imagen arrastrada en el label original
            self.original_image_label.set_image(local_file_path)
            # Puedes también limpiar la imagen mejorada cuando una nueva imagen es soltada
            self.enhanced_image_label._scene.clear()


def main():
    app = QApplication(sys.argv)
    # Cargar y aplicar el archivo CSS
    with open("styles.css", "r") as file:
        app.setStyleSheet(file.read())
    window = RealESRGANApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
