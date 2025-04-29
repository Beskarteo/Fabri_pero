from math import dist
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import ImageTk, Image
import sys
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog


class App:
    def __init__(self):
        self.win = tk.Tk()
        photo = tk.PhotoImage(file='icon1.png') #_internal\
        self.win.title("Обработка данных интерферометра Фабри-перо")
        self.win.iconphoto(False, photo)
        self.win.config(bg='#696969')
        self.win.geometry("1210x755")
        self.win.resizable(False, False)
        self.entry_var1 = tk.StringVar()
        self.entry_var2 = tk.StringVar()
        self.threshold = tk.StringVar()
        self.d = tk.StringVar()
        self.v = tk.StringVar()
        self.enabled = tk.IntVar(value=1)
        self.tip_graf = tk.IntVar(value=0)
        self.x1 = '0'
        self.x2 = '0'
        self.file = ''
        self.win.protocol("WM_DELETE_WINDOW", self.closing)

        self.fc = tk.Button(text="Загрузить изображение", command=self.im_in, bg='#9c9c9c').place(x=10, y=345)

        self.im_i = tk.Button(text="Найти центр", command=self.find_cent, bg='#9c9c9c').place(x=200, y=345)

        self.gra = tk.Button(text="Построить график", command=self.graf, bg='#9c9c9c').place(x=311, y=345)

        self.get = tk.Button(text="Получить значение", command=self.get_x, bg='#9c9c9c').place(x=840, y=345)

        self.get = tk.Button(text="Рассчитать V", command=self.formula, bg='#9c9c9c').place(x=1050, y=346)

        self.max_checkbutton = tk.Checkbutton(text="Максимумы", variable=self.enabled, command=self.checkbutton_changed,bg='#9c9c9c').place(x=465, y=345)
        self.max_checkbutton = tk.Checkbutton(text="График без обработки", variable=self.tip_graf, command=self.checkbutton_changed,bg='#9c9c9c').place(x=1005, y=140)

        self.X_lb = tk.Label(text="x центра", bg='#9c9c9c').place(x=466, y=35)
        self.X_tf = tk.Entry().place(x=466, y=60, width=70)
        self.Y_lb = tk.Label(text="y центра", bg='#9c9c9c').place(x=466, y=85)
        self.Y_tf = tk.Entry().place(x=466, y=110, width=70)
        self.radius_lb = tk.Label(text="радиус", bg='#9c9c9c').place(x=466, y=135)
        self.radius_tf = tk.Entry().place(x=466, y=160, width=70)

        self.xgr1_lb = tk.Label(text="от R", bg='#9c9c9c').place(x=600, y=345)
        self.xgr1_tf = tk.Entry(textvariable=self.entry_var1).place(x=640, y=345, width=70)
        self.xgr2_lb = tk.Label(text="до R", bg='#9c9c9c').place(x=720, y=345)
        self.xgr2_tf = tk.Entry(textvariable=self.entry_var2).place(x=763, y=345, width=70)

        self.threshold_lb = tk.Label(text="Порог бинаризации (0-255)", bg='#9c9c9c').place(x=1000, y=55)
        self.threshold_tf = tk.Entry(textvariable=self.threshold).place(x=1063, y=95, width=70)

        self.d_lb = tk.Label(text="Введите d", bg='#9c9c9c').place(x=1060, y=230)
        self.d_tf = tk.Entry(textvariable=self.d).place(x=1063, y=270, width=70)
        self.d_lb = tk.Label(text="см", bg='#696969', font=("Arial", 13, "bold")).place(x=1136, y=269)
        self.v_tf = tk.Entry(textvariable=self.v).place(x=1063, y=385, width=70)
        self.d_lb = tk.Label(text="см⁻¹", bg='#696969', font=("Arial", 13, "bold")).place(x=1136, y=384)

        # Добавим изображение
        self.canvas1 = tk.Canvas(self.win, height=300, width=450, highlightbackground='#000000')
        self.canvas1.config(bg='#808080')
        self.canvas1.place(x=10, y=35)

        # Добавим изображение
        self.canvas2 = tk.Canvas(self.win, height=300, width=450, highlightbackground='#000000')
        self.canvas2.config(bg='#808080')
        self.canvas2.place(x=540, y=35)

        # Добавим изображение
        self.canvas3 = tk.Canvas(self.win, height=363, width=980, highlightbackground='#000000')
        self.canvas3.config(bg='#808080')
        self.canvas3.place(x=10, y=380)

        self.win.mainloop()

    def im_in(self):
        self.file = filedialog.askopenfilename(filetypes=[("Image File", '.jpg .png .jpeg')])
        if self.file != "":
            self.image = Image.open(self.file)
            width, height = self.image.size
            self.width = width
            self.height = height
            while (width>450 or height>300) == 1:
                width = int(width//1.01)
                height = int(height//1.01)
            self.image_out1 = self.image.resize((width, height))
            self.photo = ImageTk.PhotoImage(self.image_out1)
            self.c_image = self.canvas1.create_image(225, 150, anchor="center", image=self.photo)
            self.x1 = '0'
            self.x2 = '0'
            self.threshold = tk.StringVar()
            self.threshold_tf = tk.Entry(textvariable=self.threshold).place(x=1063, y=95, width=70)

        else:
            print("Не выбран файл")

    def find_cent(self):
        self.img = Image.open(self.file)
        image = self.img
        data = []
        cx = 0
        cy = 0
        r = 10
        sp_x = []
        sp_y = []
        bok = []
        prot = []
        self.rad = 100

        # бинаризация
        self.img = self.img.convert("L")
        # проверять пустое ли окно self.threshold.get()
        data = self.get_data()
        self.threshold = tk.StringVar(self.win, self.threshold)
        self.threshold_tf = tk.Entry(textvariable=self.threshold).place(x=1063, y=95, width=70)

        data_np = np.array(data)

        clustering = DBSCAN(eps=3).fit(data_np)
        labels = clustering.labels_

        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)  # исключаем шум (-1)

        large_clusters = unique_labels[counts >= 5000]  # Фильтрация кластеров с >=5000 точками

        unique_labels = set(large_clusters) - {-1}  # Получаем уникальные метки (исключая шум -1)

        clusters = [data_np[labels == label].tolist() for label in unique_labels]  # Создаем список кластеров: каждый кластер — массив его точек

        cl = max(clusters, key=len)
        inf = [max(cl), min(cl), max(cl, key=lambda x: x[1]), min(cl, key=lambda x: x[1])]  # крайние точки по x и y

        kr = []  # kr p, l, v, n alg pelevin
        for i in range(len(inf)):
            if i == 0:
                kr.append([p for p in cl if p[0] == inf[i][0]])
            if i == 1:
                kr.append([p for p in cl if p[0] == inf[i][0]])
            if i == 2:
                kr.append([p for p in cl if p[1] == inf[i][1]])
            if i == 3:
                kr.append([p for p in cl if p[1] == inf[i][1]])
        tip = len([1 for kr_i in kr if len(kr_i) > 20])
        # print(tip, 'tip')
        if tip == 0:
            cx = (inf[0][0] + inf[1][0]) / 2
            cy = (inf[2][1] + inf[3][1]) / 2
            cen = [cx, cy]
            r = (dist(cen, inf[0]) + dist(cen, inf[1]) + dist(cen, inf[2]) + dist(cen, inf[3])) / 4
        if tip == 1 or tip == 2:  # найти наиб срез и взять противоположный
            kr_max = max(kr, key=len)
            if kr[0] == kr_max:
                prot = kr[1]
                bok = [kr[2], kr[3]]
            if kr[1] == kr_max:
                prot = kr[0]
                bok = [kr[2], kr[3]]
            if kr[2] == kr_max:
                prot = kr[3]
                bok = [kr[0], kr[1]]
            if kr[3] == kr_max:
                prot = kr[2]
                bok = [kr[0], kr[1]]

            k = dist(prot[0], kr_max[0])
            l = dist(prot[0], kr_max[-1])
            n = dist(kr_max[-1], kr_max[0])
            r = (k * l * n) / (4 * (k ** 2) * (l ** 2) - (k ** 2 + l ** 2 - n ** 2) ** 2) ** 0.5
            if kr[0] == kr_max:
                sp_y = [prot[0][1]]
                if dist(bok[0][1], bok[1][1]) > 1.9 * r:  # смотрим содержет ли центр
                    sp_x = [prot[0][1] - r]
                    sp_y.append(bok[0][0][0])
                    sp_y.append(bok[1][0][0])
                    sp_x.append(bok[0][0][1])
                    sp_x.append(bok[1][0][1])
                else:
                    sp_x = [prot[0][0]]
                    sp_x.append(prot[0][0] + 2 * r)
                    sp_y.append(bok[0][0][1])
                    sp_y.append(bok[1][0][1])
            if kr[1] == kr_max:
                sp_y = [prot[0][1]]
                if dist(bok[0][1], bok[1][1]) > 1.9 * r:  # смотрим содержет ли центр
                    sp_x = [prot[0][1] + r]
                    sp_y.append(bok[0][0][0])
                    sp_y.append(bok[1][0][0])
                    sp_x.append(bok[0][0][1])
                    sp_x.append(bok[1][0][1])
                else:
                    sp_x = [prot[0][0]]
                    sp_x.append(prot[0][0] - 2 * r)
                    sp_y.append(bok[0][0][1])
                    sp_y.append(bok[1][0][1])
            if kr[2] == kr_max:
                sp_x = [prot[0][0]]
                if dist(bok[0][0], bok[1][0]) > 1.9 * r:  # смотрим содержет ли центр
                    sp_y = [prot[0][1] + r]
                    sp_x.append(bok[0][0][0])  # +r
                    sp_x.append(bok[1][0][0])  # -r
                    sp_y.append(bok[0][0][1])
                    sp_y.append(bok[1][0][1])
                else:
                    sp_y = [prot[0][1]]
                    sp_x.append(bok[0][0][0])
                    sp_x.append(bok[1][0][0])
                    sp_y.append(prot[0][1] + 2 * r)
            if kr[3] == kr_max:
                sp_x = [prot[0][0]]
                if dist(bok[0][0], bok[1][0]) > 1.9 * r:  # смотрим содержет ли центр
                    sp_y = [prot[0][1] - r]
                    sp_x.append(bok[0][0][0] - r)
                    sp_x.append(bok[1][0][0] + r)
                    sp_y.append(bok[0][0][1])
                    sp_y.append(bok[1][0][1])
                else:
                    sp_y = [prot[0][1]]
                    sp_x.append(bok[0][0][0])
                    sp_x.append(bok[1][0][0])
                    sp_y.append(prot[0][1] - 2 * r)
            cx = sum(sp_x) / len(sp_x)
            cy = sum(sp_y) / len(sp_y)
        if tip == 3:
            pass
        self.cen = [cx, cy]
        self.rad = int(r)
        self.radius = tk.StringVar()
        self.radius.set(str(int(r)))
        self.radius_tf = tk.Entry(textvariable=self.radius)
        self.radius_tf.place(x=466, y=160, width=70)

        plt.clf()
        plt.imshow(image)
        plt.axis('off')  # Отключение осей
        plt.plot(cx, cy, 'ro')
        ax = plt.gca()
        circle = plt.Circle((cx, cy), r, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        plt.savefig('out_image.jpg', bbox_inches='tight', facecolor='#808080')
        plt.clf()

        self.x = tk.IntVar()
        self.x.set(int(cx))
        self.X_tf = tk.Entry(textvariable=self.x).place(x=466, y=60, width=70)
        self.y = tk.IntVar()
        self.y.set(int(cy))
        self.Y_tf = tk.Entry(textvariable=self.y).place(x=466, y=110, width=70)

        self.image_out = Image.open('out_image.jpg')

        width_out, height_out = self.image_out.size
        while (width_out > 450 or height_out > 300) == 1:
            width_out = int(width_out // 1.01)
            height_out = int(height_out // 1.01)
        self.image_out2 = self.image_out.resize((width_out, height_out))
        self.photo_out = ImageTk.PhotoImage(self.image_out2)
        self.cn_image_out = self.canvas2.create_image(225, 150, anchor="center", image=self.photo_out)

        self.max_radius = int(dist(self.cen, [0, 0]))  # максимальное расстояние до края изображения

    def graf(self):
        # загрузка изображения (преобразование в градации серого)
        image = Image.open(self.file).convert('L')
        image_np = np.array(image)

        cx, cy = self.cen

        # радиусы и средняя яркость
        radii = []
        mean_brightness = []
        self.data_max = []
        data_min = []

        # проход по радиусам от 0 до максимального расстояния
        for r in range(int(self.rad*1.1), int(self.max_radius)):
            brightness_for_r = [] # яркость точек на текущем радиусе

            # проход по всем углам
            for angle in range(360):
                # Перевод угла в радианы
                alpha = np.deg2rad(angle)

                # Координаты точки на окружности радиуса r
                x = int(cx + r * np.cos(alpha))
                y = int(cy + r * np.sin(alpha))

                # Проверка, чтобы точка не выходила за пределы изображения
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Добавляем яркость точки в список
                    brightness_for_r.append(image_np[y, x])

            # Если на данном радиусе есть точки, вычисляем среднюю яркость
            if brightness_for_r:
                mean_brightness.append(np.mean(brightness_for_r))
                radii.append(r)

        # преобразуем списки в массивы и берем квадраты x
        if self.x2 != '0':  # обновление краев графика  
            radii = np.array([rad ** 2 for rad in radii])
            mean_brightness = np.array(mean_brightness)
            sp_ind = np.array([ind for ind in range(len(radii)) if radii[ind] > int(self.x1) and radii[ind] < int(self.x2)])
            radii = radii[sp_ind]
            mean_brightness = mean_brightness[sp_ind]
        else:
            radii = np.array([rad ** 2 for rad in radii])
            mean_brightness = np.array(mean_brightness)

        # Поиск максимумов
        peaks, _ = find_peaks(mean_brightness, prominence=3) # нахожу индексы максимумов и отсеиваю лишние 3
        peaks_min, _ = find_peaks(-mean_brightness, prominence=1)

        if self.tip_graf.get() == 0: # обработка графика
            ind_min = np.where(radii > radii[peaks_min][0])[0]  # Индексы элементов > radii[peaks_min][0]
            radii = radii[ind_min]
            mean_brightness = mean_brightness[ind_min]
            self.distance = abs(radii[peaks-ind_min[0]][0] - radii[peaks-ind_min[0]][1])*1.2 # расстояние между первыми двумя максимумами
            self.distance_eps = abs(radii[peaks-ind_min[0]][0] - radii[peaks-ind_min[0]][2])*1.2 # расстояние между первым и третим максимумами

            for x, y in zip(radii[peaks-ind_min[0]], mean_brightness[peaks-ind_min[0]]):
                self.data_max.append([x, y])

            for peak in self.data_max.copy():
                sosed = [p1 for p1 in self.data_max.copy() if abs(p1[0]-peak[0])<self.distance]
                if len(sosed) == 1:
                    self.data_max = [i for i in self.data_max if i < peak] # сохранение всех макс до первого одиночного
                    break

            for x, y in zip(radii[peaks_min-ind_min[0]], mean_brightness[peaks_min-ind_min[0]]):
                data_min.append([x, y])

            data_min.sort()
            ind = 0
            while data_min[ind] < max(self.data_max):
                ind += 1
            ind_max = np.where(radii < data_min[ind][0])[0]  # Индексы элементов < radii[peaks_min][0]
            radii = radii[ind_max]
            mean_brightness = mean_brightness[ind_max]
        else:
            for x, y in zip(radii[peaks], mean_brightness[peaks]):
                self.data_max.append([x, y])

        # Создаем фигуру Matplotlib
        self.fig = Figure(figsize=(9.8, 3.1), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Строим график
        self.ax.plot(radii, mean_brightness, label="Средняя яркость")
        self.ax.set_ylabel("Средняя яркость")
        self.ax.set_title("Зависимость средней яркости от радиуса")
        self.ax.grid(True)
        self.ax.legend()
        if self.enabled.get() == 1:
            for x, y in self.data_max:
                self.ax.text(x, y, str(int(y)), fontsize=7, ha='center', va='bottom')
                self.ax.scatter(x, y, color='red', label='Максимумы')

        self.plot_frame = tk.Frame()
        self.plot_frame.place(x=13, y=383, width=978, height=360)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Добавляем панель инструментов
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def get_data(self):
        data = []
        if self.threshold.get() == '':
            for thr in range(250, 20, -10):
                self.threshold = str(thr)
                min_data = thr*1660
                img_thr = self.img.point(lambda x: 255 if x > int(self.threshold) else 0)

                img_arr = np.asarray(img_thr)

                for stroka in range(len(img_arr)):
                    for stolb in range(len(img_arr[stroka])):
                        # значение пикселя, stroka - номер строчки, stolb - номер столбца
                        if img_arr[stroka][stolb] == 255:
                            data.append([stolb, stroka])
                #print(len(data), '- пикселей')

                if len(data) > 3_000_000 or len(data) < min_data:
                    data = []
                    continue
                else:
                    return data
            return None
        else:
            self.threshold = str(self.threshold.get())
            img_thr = self.img.point(lambda x: 255 if x > int(self.threshold) else 0)

            img_arr = np.asarray(img_thr)

            for stroka in range(len(img_arr)):
                for stolb in range(len(img_arr[stroka])):
                    # значение пикселя, stroka - номер строчки, stolb - номер столбца
                    if img_arr[stroka][stolb] == 255:
                        data.append([stolb, stroka])
            # print(len(data), '- пикселей')
            return data

    def get_x(self):
        self.x1 = self.entry_var1.get()
        self.x2 = self.entry_var2.get()
        if self.x1 == '':
            self.x1 = '0'
        if self.x2 == '':
            self.x2 = str(int(self.max_radius)**2)
        self.graf()

    def checkbutton_changed(self):
        self.graf()

    def formula(self):
        inp_d = float(self.d.get())
        sp_v = []
        for i in range(0, len(self.data_max) - 3, 2):
            sp_v.append((self.data_max[i + 3][0] - self.data_max[i + 2][0]) / (self.data_max[i + 3][0] - self.data_max[i+1][0]) / (2 * inp_d))
            #print(self.data_max[i + 3][0], self.data_max[i + 2][0],  self.data_max[i + 3][0], self.data_max[i+1][0], (2 * inp_d))
        self.v = int((sum(sp_v) / len(sp_v)) * 10000) / 10000
        self.v_out = tk.IntVar()
        self.v_out.set(self.v)
        self.v_tf = tk.Entry(textvariable=self.v_out).place(x=1065, y=385, width=70)

    def closing(self):
        sys.exit()


app = App()
