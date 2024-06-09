import matplotlib.pyplot as plt     # бібліотека для роботи з графіками
import numpy as np                  # бібліотека для роботи з масивами
import scipy as sp                  # бібліотека використовується для згладжування
import scipy.fftpack

from matplotlib.widgets import Button, TextBox, RadioButtons

# бібліотеки для роботи з OriginPro
import OriginExt
import originpro as op              

# бібліотеки для роботи з файлами
import os
import sys

###############################################

parametr = 16       # "радіус" області усереднення RGB значень
base_line = 10      # кількість точок базової лінії
st = 4              # крок між точками в областях пошуку
npts = 3            # параметр FFT згладжування
save_like = '.opj'  # тип файлу для збереженняя

###############################################

# Отримуємо імена файлів з зображеннями
file_list = [file for file in os.listdir() if file.endswith('.jpg')]

# Масив з матрицями RGB значень кожного з зображень
images = np.array([plt.imread(file) for file in file_list])

# Номери зображень
Number = range(1, len(file_list) + 1)

pic = images[0].copy()  # Копія даних першого зображення

x_size, y_size = pic.shape[1], pic.shape[0]

max_points = []             # Точки з максимальними відгуками
difference_markers = []     # Маркери для збереження Мульти-відгуку



####################################################
# Функції для пошуку відгуків
####################################################

def smooth_fft(x, data, order=3):
    """Функція повертає згладжені значення data"""
    n = len(data)
    dx = x[1]-x[0]

    freq_cutoff = 1 / (2*order*dx)      # Максимальна частота пропускання
    freq = sp.fftpack.rfftfreq(n, dx)   # Масив частот

    # Утворюємо вагові коефіцієнти
    low_pass_weights = np.zeros(n)
    # Обрізаємо частоти, які більші за f_cutoff
    low_pass_weights[np.abs(freq) <= freq_cutoff] = 1.0
    # Задаємо "параболічну форму" коефіцієнтам
    low_pass_weights = low_pass_weights * np.array([1-(f/freq_cutoff)**2 for f in freq])

    # Застосовуємо перетворення Фур'є та вагові коефіцієнти
    fft_data = sp.fftpack.rfft(data)
    filtered_fft_data = fft_data * low_pass_weights

    # Використовуємо обернене перетворення Фур'є
    return sp.fftpack.irfft(filtered_fft_data)


def generate_points(up_point, down_point, step=16, image_size=None):
    """Функція приймає координати кутів прямокутної області.
        up_point --- верхній лівий кут прямокутника,
        down_point --- нижній правий кут прямокутника.
        Ця прямокутна область розбивається на сітку з точок.
        Відстань між сусідніми точками дорівнює step.
        Повертаються масиви з координатами цих точок"""
    x_1, y_1 = up_point
    x_2, y_2 = down_point

    x_coords = np.array(range(x_1, x_2, step))
    y_coords = np.array(range(y_1, y_2, step))

    # Видаляємо координати, які потім можуть призвести
    # до помилок при усередненні RGB значень

    if image_size != None:
        x_max = image_size[0]
        y_max = image_size[1]
        
        x_coords = x_coords[(x_coords >= parametr) & (x_coords <= x_max-1-parametr)] 
        y_coords = y_coords[(y_coords >= parametr) & (y_coords <= y_max-1-parametr)]

    return (x_coords, y_coords)

def mean_by_3_max(array):
    """Функція повертає середнє трьох
    найбільших значень масиву array"""
    return np.mean(np.sort(array)[-3:])


def find_max_response_in_region(up_point, down_point, step, num=1):
    """Функція приймає координати кутів прямокутної області.
        up_point --- верхній лівий кут прямокутника,
        down_point --- нижній правий кут прямокутника.
        Ця прямокутна область розбивається на сітку з точок.
        Відстань між сусідніми точками дорівнює step.
        В цій області шукають num точок з найбільшими відгуками
        за критерієм функції mean_by_3_max()"""
    
    # Отримуємо координати пікселів в цій області із кроком step
    x_coords, y_coords = generate_points(up_point, down_point, step, image_size=(x_size, y_size))

    # Знаходимо усереднені значення RGB в околі
    # "радіусом" parametr навколо цих пікселів
    R = np.array([[[int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 0])) for img in images] for x in x_coords] for y in y_coords])
    G = np.array([[[int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 1])) for img in images] for x in x_coords] for y in y_coords])
    B = np.array([[[int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 2])) for img in images] for x in x_coords] for y in y_coords])

    L = np.sqrt(R**2 + G**2 + B**2)

    Rcos = R/L
    Gcos = G/L
    Bcos = B/L

    # Середні значення RGB по базовій лінії
    R0 = np.mean(R[:,:,:base_line], axis=2)
    G0 = np.mean(G[:,:,:base_line], axis=2)
    B0 = np.mean(B[:,:,:base_line], axis=2)
    L0 = np.sqrt(R0**2 + G0**2 + B0**2)
    
    R0 = np.repeat(R0[:,:,np.newaxis], repeats=len(file_list), axis=2)
    G0 = np.repeat(G0[:,:,np.newaxis], repeats=len(file_list), axis=2)
    B0 = np.repeat(B0[:,:,np.newaxis], repeats=len(file_list), axis=2)
    L0 = np.repeat(L0[:,:,np.newaxis], repeats=len(file_list), axis=2)

    # Обраховуємо інтегральний відгук
    S = np.sqrt((R/L - R0/L0)**2 +
                (G/L - G0/L0)**2 +
                (B/L - B0/L0)**2 )

    # Знаходимо згладжені відгуки
    S_smooth = np.zeros_like(S)

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            S_smooth[j, i] = smooth_fft(Number, S[j,i], order=npts)

    # Сортуємо відгуки та повертаємо дані про num найбільших
    responses = [((i, j), mean_by_3_max(S_smooth[j,i])) for i in range(len(x_coords)) for j in range(len(y_coords))]
    res_sort = sorted(responses, key = lambda item: item[1], reverse=True)

    result = [[x_coords[i], y_coords[j], S[j,i], S_smooth[j,i], R[j,i], G[j,i], B[j,i]] for (i,j), _ in res_sort[:num]]
    return result

####################################################
# Функції для роботи із зображенням в окремому вікні
####################################################

def delete_last_region():
    """Функція видаляє останню
    виділену користувачем область"""

    # Умова, щоб видаляло лише виділені області,
    # але не початкове зображення
    if len(list_pictures) > 1:
        # Видалення
        list_pictures.pop()
        regions.pop()

        # Оновлення зображення, яке демонструється
        current_pic = list_pictures[-1]
        ax.clear()
        ax.imshow(current_pic)
        ax.set_title('Виділіть курсором області пошуку відгуку')
        fig.canvas.draw()
        
def work_with_regions(regions):
    """Функція будує графіки найбільших відгуків
    в точках вибраних користувачем областей"""

    number_of_regions = len(regions)    # Кількість областей

    # Функція нічого не робитиме, якщо області не вибрані 
    if number_of_regions == 0: return ([], [])
    
    global max_points
    global difference_markers

    max_points = []             # Точки з максимальними відгуками
    difference_markers = []     # Маркери для збереження Мульти-відгуку

    # Підготовка масиву для Мульти-відгуку
    multi_diff_S = np.zeros(len(file_list))
        
    plt.figure()

    # Для кожної області піксель з найбільшим
    # відгуком та будуємо графіки
    for i in range(number_of_regions):
        # Координати кутів області
        up_point = regions[i][0]
        down_point = regions[i][1]

        # Отримуємо дані про найбільший відгук
        x, y, S, S_smooth, R, G, B = find_max_response_in_region(up_point, down_point, step=st, num=1)[0]

        # Додаємо координати пікселя у список
        max_points.append((x,y))

        L = np.sqrt(R**2 + G**2 + B**2)

        Rcos = R/L
        Gcos = G/L
        Bcos = B/L

        R0 = np.mean(R[:base_line])
        G0 = np.mean(G[:base_line])
        B0 = np.mean(B[:base_line])
        L0 = np.sqrt(R0**2 + G0**2 + B0**2)

        # Шукаємо найбільшу різницю колірних компонент для Мульти-відгуку
        RG = np.abs((Rcos - R0/L0) - (Gcos - G0/L0))
        GB = np.abs((Gcos - G0/L0) - (Bcos - B0/L0))
        BR = np.abs((Bcos - B0/L0) - (Rcos - R0/L0))

        differences = np.array([RG, GB, BR])
        # Знаходимо найбільшу різницю за критерієм
        # найбільшої суми впродовж "часу спостережнь"
        flag = [sum(diff) == max(sum(RG), sum(GB), sum(BR)) for diff in differences]
        biggest_diff = differences[flag][0]
        difference_markers.append(np.array(['R-G', 'G-B', 'B-R'])[flag][0])

        # Додаємо найбільшу різницю до Мульти-відгуку
        multi_diff_S += biggest_diff**2
            
        # Будуємо графіки
        plt.subplot(2, number_of_regions+1, i+1)
        plt.plot(Number, S, linestyle='-', linewidth=1, marker='s', markersize=5, color='k', label='Відгук')
        plt.plot(Number, S_smooth, linestyle='-', linewidth=1, marker='o', markersize=5, color='r', label='Smooth')
        plt.title(f'Область №{i+1}$, x={x}$, $y={y}$')
        plt.xlabel('Час, відносні одиниці')
        plt.legend()

        plt.subplot(2, number_of_regions+1, number_of_regions+1 + i+1)
        plt.plot(Number, Rcos,  linestyle='-', linewidth=1, marker='s', markersize=5, color='r', label='R')
        plt.plot(Number, Gcos,  linestyle='-', linewidth=1, marker='o', markersize=5, color='g', label='G')
        plt.plot(Number, Bcos,  linestyle='-', linewidth=1, marker='^', markersize=5, color='b', label='B')
        plt.xlabel('Час, відносні одиниці')
        plt.legend()

    # Графік Мульти-відгуку
    if number_of_regions > 1:
        plt.subplot(2, number_of_regions+1, number_of_regions + 1)

        multi_diff_S = np.sqrt(multi_diff_S)
            
        plt.plot(Number, multi_diff_S, linestyle='-', linewidth=1, marker='s', markersize=5, color='k', label='Відгук')
        plt.plot(Number, smooth_fft(Number, multi_diff_S, order=npts), linestyle='-', linewidth=1, marker='o', markersize=5, color='r', label='Smooth')
        plt.title(f'Мульти-відгук')
        plt.xlabel('Час, відносні одиниці')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

    # Для подальшого збереження повертаємо координати пікселів
    # та інформацію про складові Мульти-відгуку
    return (max_points, difference_markers)

def save_data(points, markers):
    """Функція зберігає дані вказаних у
    масиві points точок в файлі .opj"""

    # На випадок, якщо немає що зберігати, функція нічого не зробить
    if max_points == []: return     

    if save_like == '.opj':
        
        ###########################################

        # Very useful, especially during development, when you are
        # liable to have a few uncaught exceptions.
        # Ensures that the Origin instance gets shut down properly.
        # Note: only applicable to external Python.

        def origin_shutdown_exception_hook(exctype, value, traceback):
            '''Ensures Origin gets shut down if an uncaught exception'''
            op.exit()
            sys.__excepthook__(exctype, value, traceback)
        if op and op.oext:
            sys.excepthook = origin_shutdown_exception_hook
        

        # Set Origin instance visibility.
        # Important for only external Python.
        # Should not be used with embedded Python. 
        if op.oext:
            op.set_show(True)
        ###########################################

        # Створюємо новий проєкт та новий worksheet
        op.new()
        book = op.new_book('w')

        if len(points) > 1:
            mult = book.add_sheet('Multi-response')
            mult.from_list(0, Number)
        
        for i in range(len(points)):
            # Створюємо Worksheet
            sheet_name = str(points[i])
            wks = book.add_sheet(sheet_name)
            
            x, y = points[i]
            
            R = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 0])) for img in images])
            G = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 1])) for img in images])
            B = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 2])) for img in images])

            # Додаємо дані в колонки
            wks.from_list(0, Number)
            wks.from_list(1, R, lname = 'R')
            wks.from_list(2, G, lname = 'G')
            wks.from_list(3, B, lname = 'B')

            # Задаємо формули для обробки даних
            wks.set_formula(4, 'sqrt(col(B)^2 + col(C)^2 + col(D)^2)')
            wks.set_label(4, 'L')

            # Обрахунок косинусів
            wks.set_formula(5, 'col(B) / col(E)')
            wks.set_label(5, 'Rcos')

            wks.set_formula(6, 'col(C) / col(E)')
            wks.set_label(6, 'Gcos')

            wks.set_formula(7, 'col(D) / col(E)')
            wks.set_label(7, 'Bcos')

            # Пошук значень базової лінії

            R0 = sum(R[:base_line]) / base_line
            G0 = sum(G[:base_line]) / base_line
            B0 = sum(B[:base_line]) / base_line
            L0 = (R0**2 + G0**2 + B0**2)**(1/2)

            base_Rcos = R0 / L0
            base_Gcos = G0 / L0
            base_Bcos = B0 / L0


            # Обрахунок відгуку
            wks.set_formula(8, f'col(F) - {base_Rcos}')
            wks.set_label(8, 'Rcos\'')

            wks.set_formula(9, f'col(G) - {base_Gcos}')
            wks.set_label(9, 'Gcos\'')

            wks.set_formula(10, f'col(H) - {base_Bcos}')
            wks.set_label(10, 'Bcos\'')

            wks.set_formula(11, 'sqrt(col(I)^2 + col(J)^2 + col(K)^2)')
            wks.set_label(11, 'S')

            # Згладжування fft
            #wks.lt_exec(f'smooth -r 2 iy:=(1,11) oy:=(1,12) method:=fft npts:={npts};')

            # Додавання даних в таблицю Мульти-відгуку
            if len(points) > 1:
                mark = markers[i]
                
                mult.from_list(3*i+1, wks.to_list(8 + {'R':0, 'G':1, 'B':2}[mark[0]]))
                mult.set_label(3*i+1, f'point_{i+1}'+mark[0]+'cos\'')
                
                mult.from_list(3*i+2, wks.to_list(8 + {'R':0, 'G':1, 'B':2}[mark[-1]]))
                mult.set_label(3*i+2, f'point_{i+1}'+mark[-1]+'cos\'')

                mult.set_formula(3*i+3, f'col({3*i+1+1}) - col({3*i+2+1})')

        if len(points) > 1:
            mult.set_formula(3*len(points)+1, 'sqrt(' + ''.join([f'col({3*i+3+1})^2 + ' for i in range(len(points))])[:-3] + ')')
            mult.set_label(3*len(points)+1, 'mult_S')

        # Зберігаємо проєкт
        # Утворення адреси поточної папки,
        # яка знадобиться для збереження даних
        files = [file for file in os.listdir() if (file.endswith('.opj') and ('best_response' in file))]
        files.sort(key=lambda file: int(file[15:-5]))
        if files == []:
            file_name = 'best_response (1)'
        else:
            file_name = f'best_response ({int(files[-1][15:-5]) + 1})'


        def get_script_path():
            if getattr(sys, 'frozen', False):
                # Виконання з .exe (PyInstaller)
                return os.path.dirname(sys.executable)
            else:
                # Виконання зі скрипту .py
                return os.path.dirname(os.path.abspath(__file__))
            
        base_path = get_script_path()
        file_path = os.path.join(base_path, file_name + '.opj')

        op.save(file_path)
        # Exit running instance of Origin.
        # Required for external Python but don't use with embedded Python.
        if op.oext:
            op.exit()

    elif save_like == '.dat':
        for point in points:
            x, y = point
            
            R = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 0])) for img in images])
            G = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 1])) for img in images])
            B = np.array([int(np.mean(img[y-parametr:y+parametr+1, x-parametr:x+parametr+1, 2])) for img in images])

            f = open(str(point)+'.dat', 'w')
            f.write('\tR\tG\tB\n')
            f.write('\n'.join(['\t'.join([str(n), str(r), str(g), str(b)]) for (n,r,g,b) in zip(range(1, len(R)+1),R,G,B)]))

            f.close()





##########################################################
# Отримання координат границь області при натисканні мишки
##########################################################

def onpress(event):
    """Функція зберігає координати
    курсору в момент натискання мишки"""

    if event.xdata == None: return
    
    global x0
    global y0
    x0 = round(event.xdata)
    y0 = round(event.ydata)



def onrelease(event):
    """Функція зберігає координати
    курсору в момент відпускання мишки мишки"""

    if event.xdata == None: return
    
    x1 = round(event.xdata)
    y1 = round(event.ydata)

    # Визначення верхнього лівого і нижнього правого куьів області
    up_point = [min(x0, x1), min(y0, y1)]
    down_point = [max(x0, x1), max(y0, y1)]

    if up_point != down_point:

        # Візуалізація прямокутника, який відповідає виділеній області
        current_pic = list_pictures[-1].copy()

        current_pic[up_point[1], up_point[0]:down_point[0]+1, :] = 0
        current_pic[down_point[1], up_point[0]:down_point[0]+1, :] = 0
        current_pic[up_point[1]:down_point[1]+1, up_point[0], :] = 0
        current_pic[up_point[1]:down_point[1]+1, down_point[0], :] = 0

        # Оновлення зображення, яке демонструється
        ax.clear()
        ax.imshow(current_pic)
        ax.set_title('Виділіть курсором області пошуку відгуку')
        fig.canvas.draw()

        # Додавання в масиви дані про виділену область
        list_pictures.append(current_pic)
        regions.append((up_point, down_point))

    
def keyboard_press(event):
    """Функція виконує відповідні дії
    при натисканні клавіш клавіатури"""
    
    if event.key == 'backspace':
        delete_last_region()
            
    elif event.key == 'enter':
        work_with_regions(regions)

    elif event.key == 'control':
        save_data(max_points, difference_markers)

#########################################
# Утворення початкового зображення
#########################################

# Чорні квадратики на початковому зображеннні
visual_points = [item[:2] for item in find_max_response_in_region([100, 100], [x_size-100, y_size-100], 16, num=100)]

for (x,y) in visual_points:
    pic[y:y+4, x:x+4, :] = 0

list_pictures = [pic.copy()]
regions = []

    
fig, ax = plt.subplots()
ax.imshow(pic)

############################################################
# Викоричтання функцій при взаємодії з мишкою чи клавіатурою
############################################################

cid = fig.canvas.mpl_connect('button_press_event', onpress)
cid = fig.canvas.mpl_connect('button_release_event', onrelease)
cid = fig.canvas.mpl_connect('key_press_event', keyboard_press)

#########################################
# Аналогічні функції для кнопок у вікні
#########################################

def delete_button(event):
    delete_last_region()

def plot_button(event):
    work_with_regions(regions)

def save_button(event):
    save_data(max_points, difference_markers)

def radio_button(label):
    global save_like
    save_like = label


ax_del = fig.add_axes([0.1, 0.925, 0.25, 0.075])
del_butt = Button(ax_del, 'Видалити область,\nBackspace')
del_butt.on_clicked(delete_button)

ax_plot = fig.add_axes([0.4, 0.925, 0.175, 0.075])
plot_butt = Button(ax_plot, 'Обрахувати,\nEnter')
plot_butt.on_clicked(plot_button)

ax_save = fig.add_axes([0.625, 0.925, 0.175, 0.075])
save_butt = Button(ax_save, 'Зберегти дані,\nCtrl')
save_butt.on_clicked(save_button)

ax_radio = fig.add_axes([0.85, 0.925, 0.1, 0.075])
radio = RadioButtons(ax_radio, ['.opj', '.dat'], activecolor='red')
radio.on_clicked(radio_button)

###############################################
# Додавання можливості редагувати параметри
###############################################
def set_radius(value):
    value = int(value)
    global parametr
    if value >= 0:
        parametr = value
    else:
        parametr = 0
        text_radius.set_val(str(parametr))

def set_base_line(value):
    value = int(value)
    global base_line
    if value >= 1:
        base_line = value
    else:
        base_line = 1
        text_base_line.set_val(str(base_line))

def set_step(value):
    value = int(value)
    global st
    if value >= 1:
        st = value
    else:
        st = 1
        text_step.set_val(str(st))

def set_npts(value):
    value = int(value)
    global npts
    if value >= 1:
        npts = value
    else:
        npts = 1
        text_npts.set_val(str(npts))
        
    
    
ax_radius = fig.add_axes([0.16, 0.03, 0.05, 0.05])
text_radius = TextBox(ax_radius, "Радіус\nусереднення:", textalignment="center")
text_radius.on_submit(set_radius)
text_radius.set_val(str(parametr))

ax_base_line = fig.add_axes([0.45, 0.03, 0.05, 0.05])
text_base_line = TextBox(ax_base_line, "Кількість точок\nбазової лінії:", textalignment="center")
text_base_line.on_submit(set_base_line)
text_base_line.set_val(str(base_line))

ax_step = fig.add_axes([0.65, 0.03, 0.05, 0.05])
text_step = TextBox(ax_step, "Крок\nрозбиття:", textalignment="center")
text_step.on_submit(set_step)
text_step.set_val(str(st))

ax_npts = fig.add_axes([0.9, 0.03, 0.05, 0.05])
text_npts = TextBox(ax_npts, "Параметр\nзгладжування:", textalignment="center")
text_npts.on_submit(set_npts)
text_npts.set_val(str(npts))

ax.set_title('Виділіть курсором області пошуку відгуку')
plt.show()
    
