import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from numba import njit, prange
from threading import Thread
import time
try:
    from PIL import Image
except Exception:
    Image = None

plt.rcParams['toolbar'] = 'None'

def calcular_pixeles(ancho, alto, max_iter, x_min, x_max, y_min, y_max, tipo, it_sierp, jx, jy, modo_smooth, exp_custom=2.0):
    x_range = np.linspace(x_min, x_max, ancho, dtype=np.float64)
    y_range = np.linspace(y_min, y_max, alto, dtype=np.float64)
    cx_grid, cy_grid = np.meshgrid(x_range, y_range)
    return _calcular_pixeles_njit(cx_grid, cy_grid, max_iter, tipo, it_sierp, jx, jy, modo_smooth, exp_custom)

@njit(parallel=True, fastmath=True)
def _calcular_pixeles_njit(cx_grid, cy_grid, max_iter, tipo, it_sierp, jx, jy, modo_smooth, exp_custom=2.0):
    alto, ancho = cx_grid.shape
    fractal = np.empty((alto, ancho), dtype=np.float64)

    if tipo == 3:
        for i in prange(alto):
            for j in range(ancho):
                x = cx_grid[i, j]
                y = cy_grid[i, j]
                inside = False
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    tx = x
                    ty = y
                    count = 0
                    while count < it_sierp:
                        if 0.333333 < tx < 0.666666 and 0.333333 < ty < 0.666666:
                            break
                        tx = (tx * 3.0) % 1.0
                        ty = (ty * 3.0) % 1.0
                        count += 1
                    if count == it_sierp:
                        inside = True
                fractal[i, j] = 1.0 if inside else 0.0
        return fractal

    for i in prange(alto):
        for j in range(ancho):
            x = cx_grid[i, j]
            y = cy_grid[i, j]
            if tipo == 1:
                zx = x
                zy = y
                sx = jx
                sy = jy
            else:
                zx = 0.0
                zy = 0.0
                sx = x
                sy = y

            it = 0
            if tipo == 2:
                while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                    nx = zx*zx - zy*zy + sx
                    ny = abs(2.0*zx*zy) + sy
                    zx = nx
                    zy = ny
                    it += 1
            else:
                if exp_custom == 2.0:
                    while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                        nx = zx*zx - zy*zy + sx
                        ny = 2.0*zx*zy + sy
                        zx = nx
                        zy = ny
                        it += 1
                else:
                    while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                        r = np.sqrt(zx*zx + zy*zy)
                        theta = np.arctan2(zy, zx) * exp_custom
                        r_n = r ** exp_custom
                        zx = r_n * np.cos(theta) + sx
                        zy = r_n * np.sin(theta) + sy
                        it += 1

            if it < max_iter and modo_smooth:
                log_zn = np.log(zx*zx + zy*zy) * 0.5
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                fractal[i, j] = (it + 1.0 - nu) / max_iter
            else:
                fractal[i, j] = it / max_iter

    return fractal


@njit(parallel=True, fastmath=True)
def _mandelbrot_julia_njit(cx_grid, cy_grid, max_iter, is_julia, jx, jy, modo_smooth):
    alto, ancho = cx_grid.shape
    fractal = np.empty((alto, ancho), dtype=np.float64)
    for i in prange(alto):
        for j in range(ancho):
            x = cx_grid[i, j]
            y = cy_grid[i, j]
            if is_julia:
                zx = x; zy = y; sx = jx; sy = jy
            else:
                zx = 0.0; zy = 0.0; sx = x; sy = y
            it = 0
            while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                nx = zx*zx - zy*zy + sx
                ny = 2.0*zx*zy + sy
                zx = nx; zy = ny; it += 1
            if it < max_iter and modo_smooth:
                log_zn = np.log(zx*zx + zy*zy) * 0.5
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                fractal[i, j] = (it + 1.0 - nu) / max_iter
            else:
                fractal[i, j] = it / max_iter
    return fractal


@njit(parallel=True, fastmath=True)
def _burning_ship_njit(cx_grid, cy_grid, max_iter, jx, jy, modo_smooth):
    alto, ancho = cx_grid.shape
    fractal = np.empty((alto, ancho), dtype=np.float64)
    for i in prange(alto):
        for j in range(ancho):
            x = cx_grid[i, j]
            y = cy_grid[i, j]
            zx = 0.0; zy = 0.0; sx = x; sy = y
            it = 0
            while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                nx = zx*zx - zy*zy + sx
                ny = abs(2.0*zx*zy) + sy
                zx = nx; zy = ny; it += 1
            if it < max_iter and modo_smooth:
                log_zn = np.log(zx*zx + zy*zy) * 0.5
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                fractal[i, j] = (it + 1.0 - nu) / max_iter
            else:
                fractal[i, j] = it / max_iter
    return fractal


@njit(parallel=True, fastmath=True)
def _exp_fractal_njit(cx_grid, cy_grid, max_iter, is_julia, jx, jy, modo_smooth, exp_custom):
    alto, ancho = cx_grid.shape
    fractal = np.empty((alto, ancho), dtype=np.float64)
    for i in prange(alto):
        for j in range(ancho):
            x = cx_grid[i, j]
            y = cy_grid[i, j]
            if is_julia:
                zx = x; zy = y; sx = jx; sy = jy
            else:
                zx = 0.0; zy = 0.0; sx = x; sy = y
            it = 0
            while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                r = np.sqrt(zx*zx + zy*zy)
                theta = np.arctan2(zy, zx) * exp_custom
                r_n = r ** exp_custom
                zx = r_n * np.cos(theta) + sx
                zy = r_n * np.sin(theta) + sy
                it += 1
            if it < max_iter and modo_smooth:
                log_zn = np.log(zx*zx + zy*zy) * 0.5
                nu = np.log(log_zn / np.log(2.0)) / np.log(2.0)
                fractal[i, j] = (it + 1.0 - nu) / max_iter
            else:
                fractal[i, j] = it / max_iter
    return fractal



def gen_vicsek(x, y, tam, it, lista):
    if it == 0:
        lista.append(([x, x+tam, x+tam, x, x], [y, y, y+tam, y+tam, y]))
        return
    nt = tam / 3
    gen_vicsek(x + nt, y + nt, nt, it-1, lista)
    gen_vicsek(x, y + nt, nt, it-1, lista)
    gen_vicsek(x + 2*nt, y + nt, nt, it-1, lista)
    gen_vicsek(x + nt, y, nt, it-1, lista)
    gen_vicsek(x + nt, y + 2*nt, nt, it-1, lista)

def gen_hexflake(cx, cy, r, it, lista):
    if it == 0:
        px = []
        py = []
        for i in range(7):
            ang = i * np.pi / 3
            px.append(cx + r * np.cos(ang))
            py.append(cy + r * np.sin(ang))
        lista.append((px, py))
        return
        
    r_n = r / 3.0
    dist = r * (2.0 / 3.0)
    
    gen_hexflake(cx, cy, r_n, it - 1, lista)
    

    for i in range(6):
        ang = i * np.pi / 3
        nx = cx + dist * np.cos(ang)
        ny = cy + dist * np.sin(ang)
        gen_hexflake(nx, ny, r_n, it - 1, lista)





def gen_koch(p1, p2, it, lx, ly):
    if it == 0:
        lx.append(p1[0]); ly.append(p1[1]); return
    v = (p2 - p1) / 3
    pa, pc = p1 + v, p1 + 2 * v
    ang = np.deg2rad(-60)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pb = pa + np.dot(rot, v)
    gen_koch(p1, pa, it-1, lx, ly); gen_koch(pa, pb, it-1, lx, ly)
    gen_koch(pb, pc, it-1, lx, ly); gen_koch(pc, p2, it-1, lx, ly)


def gen_hilbert(x, y, xi, xj, yi, yj, it, lx, ly):
    if it == 0:
        lx.append(x + (xi + yi) / 2)
        ly.append(y + (xj + yj) / 2)
        return
    gen_hilbert(x, y, yi/2, yj/2, xi/2, xj/2, it-1, lx, ly)
    gen_hilbert(x + xi/2, y + xj/2, xi/2, xj/2, yi/2, yj/2, it-1, lx, ly)
    gen_hilbert(x + xi/2 + yi/2, y + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, it-1, lx, ly)
    gen_hilbert(x + xi/2 + yi, y + xj/2 + yj, -yi/2, -yj/2, -xi/2, -xj/2, it-1, lx, ly)



# ----





def gen_dragon(p1, p2, it, signo, lx, ly):
    if it == 0:
        lx.append(p1[0]); ly.append(p1[1]); return
    v = p2 - p1
    pm = p1 + 0.5 * v + signo * 0.5 * np.array([-v[1], v[0]])
    gen_dragon(p1, pm, it-1, 1, lx, ly); gen_dragon(pm, p2, it-1, -1, lx, ly)

def gen_sierpinski(p1, p2, p3, it, lista):
    if it == 0:
        lista.append(([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]])); return
    m1, m2, m3 = (p1+p2)/2, (p2+p3)/2, (p3+p1)/2
    gen_sierpinski(p1, m1, m3, it-1, lista); gen_sierpinski(m1, p2, m2, it-1, lista); gen_sierpinski(m3, m2, p3, it-1, lista)

def gen_pentagonos(c, r, ang, it, lista):
    phi = (1 + 5**0.5) / 2
    if it == 0:
        px = [c[0] + r * np.cos(ang + i*2*np.pi/5) for i in range(6)]
        py = [c[1] + r * np.sin(ang + i*2*np.pi/5) for i in range(6)]
        lista.append((px, py)); return
    r_n = r / (1 + phi); dist = r - r_n
    for i in range(5):
        a = ang + i * 2 * np.pi / 5
        nc = c + dist * np.array([np.cos(a), np.sin(a)])
        gen_pentagonos(nc, r_n, ang, it-1, lista)


def gen_arbol(p1, p2, it, lista, alpha=0.5):
    if it == 0: return
    v = p2 - p1
    w = np.array([-v[1], v[0]])
    p3, p4 = p2 + w, p1 + w
    
    pm = p4 + alpha * (p3 - p4) + np.sqrt(alpha * (1 - alpha)) * np.array([-(p3[1]-p4[1]), (p3[0]-p4[0])])
    
    lista.append(([p1[0], p2[0], p3[0], p4[0], p1[0]], [p1[1], p2[1], p3[1], p4[1], p1[1]]))
    if it > 1:
        gen_arbol(p4, pm, it-1, lista, alpha)
        gen_arbol(pm, p3, it-1, lista, alpha)





# ----




class AppFractales:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8), facecolor='black')
        self.julia_c = complex(-0.7, 0.27) 
        self.colores_lin = ['#00ffff', '#ff00ff', '#ffffff', '#00ff00', '#ffaa00', '#ff4444']
        self.paletas = [
            'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'gnuplot2',
            'ocean', 'gist_earth', 'terrain', 'cubehelix', 'hot', 'coolwarm', 'Spectral',
            'turbo', 'nipy_spectral', 'copper', 'bone', 'pink', 'spring', 'summer',
            'autumn', 'winter'
        ]
        self.idx_col, self.it_lin, self.en_exp = 0, 4, False
        self.hd_res = 6000
        self.mostrar_orbita = False
        self.puntos_orbita = None
        self.controles = []
        self.geom_cache = {}
        self.grid_cache = {}
        self.exponente_custom = 2.0
        self.info_math = {
            0: ("Mandelbrot", r"$Z_{n+1} = Z_n^2 + C$", "D ≈ 2.0"),
            1: ("Julia", r"$Z_{n+1} = Z_n^2 + C_fix$", "D ≈ 2.0"),
            2: ("Burning Ship", r"$Z_{n+1} = (|Re| + i|Im|)^2 + C$", "D ≈ 2.0"),
            -1: ("Personalizado", "Fórmula Libre", "D Variable"), 
            3: ("Alfombra Sierpinski", "Remoción central 1/9", "D ≈ 1.89"),
            4: ("Copo de Koch", "L_{n+1} = 4/3 L_n", "D ≈ 1.26"),
            5: ("Curva del Dragón", "Plegado de 90°", "D = 2.0"),
            6: ("Triángulo Sierpinski", "Sucesión de 3 áreas", "D ≈ 1.58"),
            7: ("Árbol de Pitágoras", "Crecimiento binario", "D ≈ 2.0"),
            8: ("Pentágono Áureo", "Simetría n-flake", "D ≈ 1.67"),
            10: ("Copo de Koch Invertido", "Geometría Anti-Copo", "D ≈ 1.26"),
            11: ("Curva de Hilbert", "Curva que llena el espacio", "D = 2.0"),
            12: ("Fractal de Vicsek", "Estructura auto-similar en Cruz", "D ≈ 1.46"),
            13: ("Copo Hexagonal", "Sierpinski Hexflake simétrico", "D ≈ 1.77")
        }
        self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.fig.canvas.mpl_connect('key_press_event', self.tecla)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse)
        self.destinos = {
            0: [ # Mandelbrot
            {"nombre": "Valle de Elefantes", "re": -0.74364, "im": 0.13182, "z": 80},
            {"nombre": "Caballitos de Mar", "re": -0.745, "im": 0.1, "z": 150},
            {"nombre": "Triple Espiral", "re": -0.088, "im": 0.655, "z": 250}
        ],
        1: [ # Julia
            {"nombre": "Conejo Douady", "re": -0.123, "im": 0.745, "z": 10},
            {"nombre": "Galaxia Spiral", "re": -0.7, "im": 0.27, "z": 5},
            {"nombre": "Dendritas", "re": 0, "im": 0.8, "z": 10}
        ]
        },
        self.angulo = 0.5
        
        self.it_pix = 123      # Iteraciones iniciales
        self.val_color = 1.0   # Rango de colores (1.0 = rango completo, 0.5 = mitad)
        self.val_hue = 0.0     # Rotacion de Hue (0.0 = sin cambio, 0.5 = media vuelta)
        self.modo_smooth = False # Smooth desactivado

        self.formula_usuario = "Z**2 + C" # Fórmula inicial por defecto
        self.es_modo_julia = False        # False = Mandelbrot, True = Julia
        self.custom_c = complex(-0.7, 0.27) # Constante inicial si usa modo Julia



        self.animando = False
        self.timer_demo = None
        self.dir_zoom = 1 


        self.categoria_actual = "selector" 



        manager = plt.get_current_fig_manager()
        

        manager.window.state('zoomed')
        
        
        manager.set_window_title("Explorador Interactivo de Fractales - Feria de Ciencias 2027")


        try:
            manager.window.iconphoto(False, plt.PhotoImage(file="Los Fractales xD Icon.png"))
        except:
            pass 


        self.menu_principal()
        plt.show()






    def limpiar(self):
        for c in self.controles:

            if hasattr(c, 'disconnect_events'):
                c.disconnect_events()
            if hasattr(c, 'ax'):
                c.ax.remove()
        self.controles = []



# ----

    def menu_principal(self):
        self.en_exp = False
        self.it_lin = 4
        self.limpiar()
        self.fig.clf()
        
        plt.get_current_fig_manager().set_window_title("Explorador Interactivo de Fractales - Menú Principal")

        self.fig.text(0.5, 0.94, "EXPLORADOR INTERACTIVO DE FRACTALES", color='gold', fontsize=18, fontweight='bold', fontfamily='monospace', ha='center')
        self.fig.text(0.5, 0.90, "Feria de Ciencias - Matemática Fractal", color='#666666', fontsize=10, fontfamily='monospace', ha='center')


        self.conf_menu = [
            (-2.1, 0.6, -1.35, 1.35, 0, "Conjunto de Mandelbrot"),
            (-1.5, 1.5, -1.5, 1.5, 1, "Conjunto Julia"),
            (-1.8, 0.8, -1.8, 0.8, 2, "Conjunto de Burning Ship"),
            (-1.5, 1.5, -1.5, 1.5, -1, "Fractal Complejo Personalizado"),
            (0, 1, 0, 1, 3, "Alfombra de Sierpinski"),
            (-0.2, 1.2, -0.4, 1.0, 4, "Copo de Koch Invertido"),
            (-0.4, 1.4, -0.7, 1.1, 5, "Dragón"),
            (-0.1, 1.1, -0.1, 1.1, 6, "Triángulo de Sierpinski"),
            (-0.2, 1.2, -0.3, 1.1, 7, "Árbol"),
            (-0.1, 1.1, -0.1, 1.1, 8, "Pentágono de Sierpinski"),
            (-0.2, 1.2, -0.4, 1.0, 10, "Copo de Koch"),
            (0.0, 1.0, 0.0, 1.0, 11, "Curva Hilbert"),
            (0.0, 1.0, 0.0, 1.0, 12, "Fractal Vicsek"),
            (0.5, 0.5, 0.0, 1.0, 13, "Copo Hexagonal")
        ]


        if self.categoria_actual == "selector":
            self.mostrar_pantalla_selector()
        elif self.categoria_actual == "complejos":
            self.mostrar_sub_menu(filtrar_complejos=True)
        elif self.categoria_actual == "geometricos":
            self.mostrar_sub_menu(filtrar_complejos=False)

    def mostrar_pantalla_selector(self):

        self.fig.text(0.5, 0.65, "Selecciona una categoría para comenzar la exploración:", color='#888888', fontsize=12, fontfamily='monospace', ha='center')


        ax_bc1 = self.fig.add_axes([0.22, 0.40, 0.24, 0.12])
        btn_c1 = Button(ax_bc1, 'FRACTALES COMPLEJOS\n\n(Mandelbrot, Julia, ...)', color='#111111', hovercolor='#1d2d2d')
        ax_bc1.spines[['top','bottom','left','right']].set_visible(False)
        btn_c1.label.set_color('cyan')
        btn_c1.label.set_fontfamily('monospace')
        btn_c1.label.set_fontweight('bold')
        btn_c1.label.set_fontsize(11)
        btn_c1.on_clicked(lambda e: self.cambiar_vista_menu("complejos"))
        self.controles.append(btn_c1)


        ax_bc2 = self.fig.add_axes([0.54, 0.40, 0.24, 0.12])
        btn_c2 = Button(ax_bc2, 'FRACTALES GEOMÉTRICOS\n\n(Sierpinski, Koch, ...)', color='#111111', hovercolor='#2d2613')
        ax_bc2.spines[['top','bottom','left','right']].set_visible(False)
        btn_c2.label.set_color('gold')
        btn_c2.label.set_fontfamily('monospace')
        btn_c2.label.set_fontweight('bold')
        btn_c2.label.set_fontsize(11)
        btn_c2.on_clicked(lambda e: self.cambiar_vista_menu("geometricos"))
        self.controles.append(btn_c2)
        
        self.fig.canvas.draw()

    def cambiar_vista_menu(self, destino):
        self.categoria_actual = destino
        self.menu_principal()

    def mostrar_sub_menu(self, filtrar_complejos):
        if filtrar_complejos:
            items = [c for c in self.conf_menu if c[4] <= 2 or c[4] == -1]
            columnas_max = 4
            y_pos_inicial = 0.38
            titulo_seccion = "Categoría: Fractales Matemáticos de Variable Compleja"
            color_seccion = "cyan"
        else:
            items = [c for c in self.conf_menu if c[4] >= 3]
            columnas_max = 5
            y_pos_inicial = 0.54 
            titulo_seccion = "Categoría: Fractales Geométricos Estructurales"
            color_seccion = "gold"

        self.fig.text(0.5, 0.84, titulo_seccion, color=color_seccion, fontsize=12, fontfamily='monospace', fontweight='bold', ha='center')


        for idx, conf in enumerate(items):
            col = idx % columnas_max
            fila = idx // columnas_max
            

            ancho_eje = 0.13 
            
            if columnas_max == 5:

                x_pos = 0.04 + col * 0.188  
                y_pos = y_pos_inicial if fila == 0 else y_pos_inicial - 0.36 
            elif columnas_max == 4:
                x_pos = 0.06 + col * 0.23
                y_pos = y_pos_inicial if fila == 0 else y_pos_inicial - 0.36
            else:
                x_pos = 0.16 + col * 0.25
                y_pos = y_pos_inicial

            c_lin = self.colores_lin[self.idx_col % 6]
            tipo = conf[4]

            ax = self.fig.add_axes([x_pos, y_pos, ancho_eje, 0.22], facecolor='black')

            
            if tipo == 4:

                lx, ly = [], []
                p1, p2, p3 = np.array([0.15, 0.25]), np.array([0.5, 0.856]), np.array([0.85, 0.25])
                

                gen_koch(p1, p2, 2, lx, ly)
                lx.append(p2[0]); ly.append(p2[1])
                gen_koch(p2, p3, 2, lx, ly)
                lx.append(p3[0]); ly.append(p3[1])
                gen_koch(p3, p1, 2, lx, ly)
                lx.append(p1[0]); ly.append(p1[1])
                
                ax.fill(lx, ly, color=c_lin, alpha=0.15)
                ax.plot(lx, ly, color=c_lin, lw=1)
                ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect('equal')


# ------



            elif tipo == 12:
                pts = []
                gen_vicsek(0, 0, 1.0, 3, pts)
                for tx, ty in pts: ax.plot(tx, ty, color=c_lin, lw=0.8)
                ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal')
            elif tipo == 13:
                pts = []
                gen_hexflake(0.5, 0.5, 0.45, 2, pts)
                for tx, ty in pts: ax.fill(tx, ty, color=c_lin, alpha=0.3); ax.plot(tx, ty, color=c_lin, lw=0.8)
                ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect('equal')


            elif tipo == 10:
                # --- MINIATURA COPO INVERTIDO (CONECTADO HACIA ADENTRO) ---
                lx, ly = [], []
                p1, p2, p3 = np.array([0.15, 0.25]), np.array([0.5, 0.856]), np.array([0.85, 0.25])
                
                # Sentido de giro inverso (p2 -> p1 -> p3 -> p2) para que apunte hacia adentro
                gen_koch(p2, p1, 2, lx, ly)
                lx.append(p1[0]); ly.append(p1[1])
                gen_koch(p1, p3, 2, lx, ly)
                lx.append(p3[0]); ly.append(p3[1])
                gen_koch(p3, p2, 2, lx, ly)
                lx.append(p2[0]); ly.append(p2[1])
                
                ax.fill(lx, ly, color=c_lin, alpha=0.15)
                ax.plot(lx, ly, color=c_lin, lw=1)
                ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect('equal')


            elif tipo == 5:
                lx, ly = [], []
                gen_dragon(np.array([0,0]), np.array([1,0]), 8, 1, lx, ly)
                ax.plot(lx, ly, color=c_lin, lw=1)
                ax.set_aspect('equal')
            elif tipo == 6:
                pts = []
                gen_sierpinski(np.array([0.05, 0.05]), np.array([0.95, 0.05]), np.array([0.5, 0.83]), 3, pts)
                for tx, ty in pts: 
                    ax.fill(tx, ty, color=c_lin, alpha=0.3)
                    ax.plot(tx, ty, color=c_lin, lw=1)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            elif tipo == 7:
                pts = []
                gen_arbol(np.array([0.45,0]), np.array([0.55,0]), 4, pts)
                for tx, ty in pts: ax.plot(tx, ty, color=c_lin, lw=1)
                ax.set_aspect('equal')
            elif tipo == 8:
                pts = []
                gen_pentagonos(np.array([0.5,0.5]), 0.4, np.pi/2, 3, pts)
                for tx, ty in pts: ax.fill(tx, ty, color=c_lin, alpha=0.4); ax.plot(tx, ty, color=c_lin, lw=1)
                ax.set_aspect('equal')
            elif tipo == -1:
                img = calcular_pixeles(80, 80, 41, conf[0], conf[1], conf[2], conf[3], 1, 3, 0.0, 0.8, False)
                ax.imshow(img, cmap=self.paletas[self.idx_col % len(self.paletas)], origin='lower', aspect='equal')
            elif tipo == 11:
                lx, ly = [], []
                gen_hilbert(0, 0, 1, 0, 0, 1, 3, lx, ly) # Nivel 3 para miniatura
                ax.plot(lx, ly, color=c_lin, lw=1.2)
                ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal')
            else:
                img = calcular_pixeles(80, 80, 41, conf[0], conf[1], conf[2], conf[3], tipo, 3, -0.7, 0.27, False)
                ax.imshow(img, cmap=self.paletas[self.idx_col % len(self.paletas)], origin='lower', aspect='equal')
            
            ax.axis('off')
            ax.set_title(conf[5], color='#dddddd', fontsize=9, fontfamily='monospace', fontweight='bold', pad=4)
            
            ax_b = self.fig.add_axes([x_pos + 0.02, y_pos - 0.04, 0.1, 0.03])
            btn = Button(ax_b, 'Explorar \u2192', color='#111111', hovercolor='#252525')
            ax_b.spines[['top','bottom','left','right']].set_visible(False)
            btn.label.set_color('cyan')
            btn.label.set_fontfamily('monospace')
            btn.label.set_fontsize(8)
            btn.on_clicked(lambda e, t=tipo, c=conf: self.abrir(t, c))
            self.controles.append(btn)

        # Botón para retroceder de categoría
        ax_ret = self.fig.add_axes([0.43, 0.02, 0.14, 0.035])
        btn_ret = Button(ax_ret, '← Volver al Inicio', color='#111111', hovercolor='#222222')
        ax_ret.spines[['top','bottom','left','right']].set_visible(False)
        btn_ret.label.set_color('#aaaaaa')
        btn_ret.label.set_fontfamily('monospace')
        btn_ret.label.set_fontsize(9)
        btn_ret.on_clicked(lambda e: self.cambiar_vista_menu("selector"))
        self.controles.append(btn_ret)

        self.fig.canvas.draw()










# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-




    def abrir(self, tipo, c):
        self.limpiar()
        self.en_exp = True
        self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f = c
        self.x_orig, self.y_orig = self.x_min, self.y_min 
        self.x_max_orig, self.y_max_orig = self.x_max, self.y_max
        

        # --- TÍTULO DINÁMICO SEGÚN EL FRACTAL SELECCIONADO ---
        plt.get_current_fig_manager().set_window_title(f"Explorando: {self.nom_f}")
        
        self.x_orig, self.y_orig = self.x_min, self.y_min 

        self.fig.clf()
        self.ax = self.fig.add_subplot(111, facecolor='black')
        self.fig.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9)

        self.image = self.ax.imshow(np.zeros((2, 2)), cmap=self.paletas[self.idx_col % len(self.paletas)],
                                    origin='lower', aspect='equal', visible=False,
                                    extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                                    interpolation='nearest', alpha=1.0)
        try:
            self.image.set_clim(0.0, 1.0)
        except Exception:
            pass
        self.orbit_line, = self.ax.plot([], [], color='black', marker='o', markersize=2,
                                        lw=0.5, alpha=0.7, visible=False)
        self._geometric_artists = []

        # TEXTOS FIJOS (Subimos info_math a 0.45 para dar más espacio)
        self.txt_help = self.fig.text(0.02, 0.5, "", color='#888888', fontsize=9, family='monospace', va='center')
        self.txt_palette = self.fig.text(0.02, 0.92, "", color='white', fontsize=10, family='monospace')
        self.txt_info = self.fig.text(0.98, 0.40, "", color='white', fontsize=10, 
                                      family='monospace', ha='right', va='bottom',
                                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Botón Volver (Estilo minimalista)
        ax_v = self.fig.add_axes([0.02, 0.02, 0.08, 0.04])
        btn = Button(ax_v, '← Menú', color='#1a1a1a', hovercolor='#333333') 
        ax_v.spines[['top','bottom','left','right']].set_visible(False) # Quita el borde gris del botón
        btn.label.set_color('#aaaaaa')
        btn.label.set_fontfamily('monospace')
        btn.label.set_fontsize(9)
        btn.on_clicked(lambda e: self.menu_principal()); self.controles.append(btn)

        
        # SLIDER: Solo para fractales geométricos (tipo 3 en adelante)
        if tipo >= 3:
            max_i = 3 if tipo == 9 else (12 if tipo == 5 else 8)
            ax_s = self.fig.add_axes([0.3, 0.05, 0.3, 0.03], facecolor='#222222')
            self.slid = Slider(ax_s, 'Detalle ', 0, max_i, valinit=self.it_lin, valfmt='%d', valstep=1)
            self.slid.label.set_color('white'); self.slid.on_changed(self.cambiar_it)
            self.controles.append(self.slid)

        # CASILLA JULIA: Solo para Julia (tipo 1). La bajamos a 0.20
        if tipo == 1:
            ax_box = self.fig.add_axes([0.86, 0.18, 0.12, 0.04])
            v_init = str(self.julia_c).replace('(','').replace(')','').replace('j','i')
            self.txt_julia = TextBox(ax_box, 'C = ', initial=v_init, color='#111111')
            self.txt_julia.label.set_color('white')
            self.txt_julia.text_disp.set_color('cyan')
            self.txt_julia.on_submit(self.cambiar_c_julia)
            self.controles.append(self.txt_julia)

        # --- CASILLAS DE NAVEGACIÓN RÁPIDA ---
        if tipo <= 2:
            ax_x = self.fig.add_axes([0.20, 0.02, 0.12, 0.03])
            self.box_x = TextBox(ax_x, 'Re ', initial=f"{(self.x_min+self.x_max)/2:.2f}")
            self.box_x.label.set_color('#ffffff')
        
            ax_y = self.fig.add_axes([0.35, 0.02, 0.12, 0.03])
            self.box_y = TextBox(ax_y, 'Im ', initial=f"{(self.y_min+self.y_max)/2:.2f}")
            self.box_y.label.set_color('#ffffff')
        
            ax_z = self.fig.add_axes([0.50, 0.02, 0.12, 0.03])
            self.box_z = TextBox(ax_z, 'Zoom ', initial="1")
            self.box_z.label.set_color('#ffffff')

            # Botón Capturar (Color Oro)
            ax_cap = self.fig.add_axes([0.65, 0.02, 0.10, 0.03])
            self.btn_capturar = Button(ax_cap, 'Capturar', color='#2b2510', hovercolor='#473d1a')
            self.btn_capturar.label.set_color('#ffd700') 
            self.btn_capturar.label.set_fontfamily('monospace')
            self.btn_capturar.label.set_fontweight('bold')
            self.btn_capturar.label.set_fontsize(8)
            self.btn_capturar.on_clicked(self.capturar_coordenadas_actuales)
            self.controles.append(self.btn_capturar)

            # Ocultamos los bordes estrictamente dentro del condicional donde existen las variables
            ax_x.spines[['top','bottom','left','right']].set_visible(False)
            ax_y.spines[['top','bottom','left','right']].set_visible(False)
            ax_z.spines[['top','bottom','left','right']].set_visible(False)
            ax_cap.spines[['top','bottom','left','right']].set_visible(False)

            self.box_x.on_submit(self.ir_a_coords)
            self.box_y.on_submit(self.ir_a_coords)
            self.box_z.on_submit(self.ir_a_coords)
            self.controles.extend([self.box_x, self.box_y, self.box_z])


            # --- AGREGAR ESTO JUSTO DESPUÉS DE: self.controles.extend([self.box_x, self.box_y, self.box_z]) ---
            self.caja_activa = None
            
            # Rastreador de clics para saber cuál caja está activa
            def al_hacer_clic_x(_): self.caja_activa = self.box_x
            def al_hacer_clic_y(_): self.caja_activa = self.box_y
            def al_hacer_clic_z(_): self.caja_activa = self.box_z
            
            # Conectamos el evento de clic de Matplotlib a los ejes de cada caja
            ax_x.figure.canvas.mpl_connect('button_press_event', lambda e: al_hacer_clic_x(None) if e.inaxes == ax_x else None)
            ax_y.figure.canvas.mpl_connect('button_press_event', lambda e: al_hacer_clic_y(None) if e.inaxes == ax_y else None)
            ax_z.figure.canvas.mpl_connect('button_press_event', lambda e: al_hacer_clic_z(None) if e.inaxes == ax_z else None)






# ---



    
        if self.tipo in self.destinos:
            ax_hub = self.fig.add_axes([0.86, 0.10, 0.12, 0.04])
            btn_hub = Button(ax_hub, 'Lugares', color='#222222', hovercolor='#444444')
            btn_hub.label.set_color('gold')
            btn_hub.on_clicked(self.mostrar_hub) # <--- Ahora llama al HUB
            self.controles.append(btn_hub)
   
        # Dentro de def abrir, busca donde creas el slider de detalle:
        if tipo == 7:
            ax_ang = self.fig.add_axes([0.3, 0.01, 0.3, 0.03], facecolor='#222222')
            self.slid_ang = Slider(ax_ang, 'Ángulo ', 0.2, 0.8, valinit=0.5)
            self.slid_ang.label.set_color('white')
            self.slid_ang.label.set_fontfamily('monospace') # <-- Estilo Monospace
            self.slid_ang.valtext.set_fontfamily('monospace')
            self.slid_ang.valtext.set_color('cyan')
            self.slid_ang.on_changed(self.cambiar_ang)
            self.controles.append(self.slid_ang)

        # --- NUEVOS CONTROLES DE RENDERIZADO (Solo para fractales de píxeles) ---
        if tipo < 3 or tipo == -1:
            # Slider de Iteraciones
            ax_it = self.fig.add_axes([0.78, 0.75, 0.17, 0.02], facecolor='#222222')
            self.slid_it = Slider(ax_it, 'Iter ', 0, 800, valinit=self.it_pix, valfmt='%d')
            self.slid_it.label.set_color('white')
            self.slid_it.label.set_fontfamily('monospace') # <-- Estilo Monospace
            self.slid_it.valtext.set_fontfamily('monospace')
            self.slid_it.valtext.set_color('cyan')
            self.slid_it.on_changed(self.cambiar_it_pix_fast)
            self.controles.append(self.slid_it)

            # Slider de Rango de Color
            ax_c = self.fig.add_axes([0.78, 0.70, 0.17, 0.02], facecolor='#222222')
            self.slid_col = Slider(ax_c, 'Rango ', 0.0, 1.0, valinit=self.val_color, valstep=0.01)
            self.slid_col.label.set_color('white')
            self.slid_col.label.set_fontfamily('monospace') # <-- Estilo Monospace
            self.slid_col.valtext.set_fontfamily('monospace')
            self.slid_col.valtext.set_color('cyan')
            self.slid_col.on_changed(self.cambiar_color_fast)
            self.controles.append(self.slid_col)

            # Slider de Hue (Rotacion de colores)
            ax_h = self.fig.add_axes([0.78, 0.65, 0.17, 0.02], facecolor='#222222')
            self.slid_hue = Slider(ax_h, 'Hue ', 0.0, 1.0, valinit=self.val_hue, valstep=0.01)
            self.slid_hue.label.set_color('white')
            self.slid_hue.label.set_fontfamily('monospace') # <-- Estilo Monospace
            self.slid_hue.valtext.set_fontfamily('monospace')
            self.slid_hue.valtext.set_color('cyan')
            self.slid_hue.on_changed(self.cambiar_hue_fast)
            self.controles.append(self.slid_hue)

            # Botón para activar/desactivar Smooth (Estilo Premium sin bordes)
            ax_sm = self.fig.add_axes([0.78, 0.60, 0.17, 0.03])
            txt_sm = 'Smooth: ON' if self.modo_smooth else 'Smooth: OFF'
            col_sm = '#0a2f1d' if self.modo_smooth else '#1a1a1a' # Verde tecnológico vs Gris oscuro
            col_txt = '#00ff66' if self.modo_smooth else '#888888'
            self.btn_sm = Button(ax_sm, txt_sm, color=col_sm, hovercolor='#252525')
            ax_sm.spines[['top','bottom','left','right']].set_visible(False) # Oculta borde gris tridimensional
            self.btn_sm.label.set_color(col_txt)
            self.btn_sm.label.set_fontfamily('monospace')
            self.btn_sm.label.set_fontsize(9)
            self.btn_sm.on_clicked(self.toggle_smooth)
            self.controles.append(self.btn_sm)

            # Slider adicional para ajustar la resolución HD de guardado (solo fractales de píxeles)
            ax_hd = self.fig.add_axes([0.02, 0.10, 0.18, 0.03], facecolor='#222222')
            self.slid_hd = Slider(ax_hd, 'HD Res', 1000, 8000, valinit=self.hd_res, valfmt='%d', valstep=500)
            self.slid_hd.label.set_color('white')
            self.slid_hd.label.set_fontfamily('monospace')
            self.slid_hd.valtext.set_color('cyan')
            self.slid_hd.valtext.set_fontfamily('monospace')
            self.slid_hd.on_changed(self.cambiar_hd_res)
            self.controles.append(self.slid_hd)


        # --- PANEL IZQUIERDO: CONTROLES PARA FRACTAL PERSONALIZADO (ID -1) ---
        if tipo == -1:
            # A. LaTeX Visual
            f_latex = r"$Z_{n+1} = " + self.formula_usuario.replace("**", "^") + "$"
            self.txt_latex_visual = self.fig.text(0.07, 0.85, f_latex, color='gold', fontsize=12, va='center')
            self.txt_latex_visual.set_fontfamily('monospace')
            
            # B. Cuadro de Entrada: Fórmula en Python Puro
            ax_form = self.fig.add_axes([0.07, 0.75, 0.15, 0.04])
            self.box_formula = TextBox(ax_form, 'Fórmula: ', initial=self.formula_usuario, color='#111111')
            ax_form.spines[['top','bottom','left','right']].set_visible(False) # Quita marco gris
            self.box_formula.label.set_color('white')
            self.box_formula.label.set_fontfamily('monospace')
            self.box_formula.text_disp.set_color('#00ff00') 
            self.box_formula.text_disp.set_fontfamily('monospace')
            self.box_formula.on_submit(self.cambiar_formula_custom)
            self.controles.append(self.box_formula)
            
            # C. Botón de Modo: Mandelbrot vs Julia
            ax_modo = self.fig.add_axes([0.07, 0.68, 0.15, 0.04])
            txt_modo = "Tipo: Julia" if self.es_modo_julia else "Tipo: Mandelbrot"
            col_modo = "#0a2a2a" if self.es_modo_julia else "#2a2a0a" # Tonos oscuros mate elegantes
            col_txt_m = "cyan" if self.es_modo_julia else "gold"
            self.btn_modo = Button(ax_modo, txt_modo, color=col_modo, hovercolor='#252525')
            ax_modo.spines[['top','bottom','left','right']].set_visible(False) # Quita marco gris
            self.btn_modo.label.set_color(col_txt_m)
            self.btn_modo.label.set_fontfamily('monospace')
            self.btn_modo.label.set_fontsize(9)
            self.btn_modo.on_clicked(self.toggle_modo_custom)
            self.controles.append(self.btn_modo)
            
            # D. Cuadro de Entrada: Número Fijo C
            ax_c_custom = self.fig.add_axes([0.07, 0.58, 0.15, 0.04])
            c_init_str = str(self.custom_c).replace('(','').replace(')','').replace('j','i')
            self.box_c_custom = TextBox(ax_c_custom, 'C Fijo: ', initial=c_init_str, color='#111111')
            ax_c_custom.spines[['top','bottom','left','right']].set_visible(False) # Quita marco gris
            self.box_c_custom.label.set_color('white')
            self.box_c_custom.label.set_fontfamily('monospace')
            self.box_c_custom.text_disp.set_color('cyan')
            self.box_c_custom.text_disp.set_fontfamily('monospace')
            self.box_c_custom.on_submit(self.cambiar_c_custom)
            self.controles.append(self.box_c_custom)

        self.actualizar()





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-



    def cambiar_formula_custom(self, texto):
        self.formula_usuario = texto.strip()
        # Actualizamos el texto LaTeX visual arriba a la izquierda
        f_latex = r"$Z_{n+1} = " + self.formula_usuario.replace("**", "^") + "$"
        self.txt_latex_visual.set_text(f_latex)
        # Parse exponent once and cache it to avoid repeated string work
        exponente = 2.0
        formula_clean = self.formula_usuario.lower().replace(" ", "")
        if "**" in formula_clean:
            try:
                parte_numerica = ""
                for caracter in formula_clean.split("**")[1]:
                    if caracter.isdigit() or caracter == ".":
                        parte_numerica += caracter
                    else:
                        break
                if parte_numerica:
                    exponente = float(parte_numerica)
            except Exception:
                exponente = 2.0
        self.exponente_custom = exponente
        self.actualizar()

    def toggle_modo_custom(self, _):
        self.es_modo_julia = not self.es_modo_julia
        txt = "Tipo: Julia" if self.es_modo_julia else "Tipo: Mandelbrot"
        col = "#004444" if self.es_modo_julia else "#444400"
        self.btn_modo.color = col
        self.btn_modo.label.set_text(txt)
        self.fig.canvas.draw_idle()
        self.actualizar()

    def cambiar_c_custom(self, texto):
        try:
            self.custom_c = complex(texto.replace('i', 'j').replace(' ', ''))
            self.actualizar()
        except:
            c_init_str = str(self.custom_c).replace('(','').replace(')','').replace('j','i')
            self.box_c_custom.set_val(c_init_str)

    def _save_image_async(self, filename=None):
        # Grab the current image array and save it in a background thread using PIL if available
        try:
            arr = None
            if hasattr(self, 'image') and self.image is not None:
                arr = self.image.get_array()
            if arr is None:
                print('No hay imagen para guardar.')
                return
            arrf = np.array(arr, dtype=np.float32)[::-1, :]
            cmap = plt.get_cmap(self.paletas[self.idx_col % len(self.paletas)])
            rgba = (cmap(arrf) * 255).astype(np.uint8)
            if filename is None:
                fname = f"fractal_{time.strftime('%Y%m%d_%H%M%S')}.png"
            else:
                fname = filename
            def _worker(data, name):
                try:
                    if Image is not None:
                        Image.fromarray(data).save(name, optimize=True)
                    else:
                        plt.imsave(name, data[:,:, :3], cmap=None)
                    print('Guardado en', name)
                except Exception as ex:
                    print('Error guardando imagen:', ex)
            t = Thread(target=_worker, args=(rgba, fname), daemon=True)
            t.start()
            print('Guardado en background:', fname)
        except Exception:
            pass

    def _apply_palette_and_hue(self, img):
        # img: 2D float array in [0,1]
        # Applies current palette, range (val_color) and hue rotation (val_hue)
        try:
            img_scaled = np.clip(np.array(img, dtype=np.float32) * float(self.val_color), 0.0, 1.0)
            cmap = plt.get_cmap(self.paletas[self.idx_col % len(self.paletas)])
            rgba = cmap(img_scaled)
            # rotate hue in RGB (keep alpha)
            rgb = rgba[..., :3]
            hsv = mcolors.rgb_to_hsv(rgb)
            hsv[..., 0] = (hsv[..., 0] + float(self.val_hue)) % 1.0
            rgb2 = mcolors.hsv_to_rgb(hsv)
            rgba[..., :3] = rgb2
            return rgba
        except Exception:
            try:
                cmap = plt.get_cmap(self.paletas[self.idx_col % len(self.paletas)])
                return cmap(img)
            except Exception:
                return np.dstack([img, img, img, np.ones_like(img)])


    def toggle_smooth(self, _):
        self.modo_smooth = not self.modo_smooth
        if hasattr(self, 'btn_sm'):
            txt_sm = 'Smooth: ON' if self.modo_smooth else 'Smooth: OFF'
            col_sm = '#0a2f1d' if self.modo_smooth else '#1a1a1a'
            col_txt = '#00ff66' if self.modo_smooth else '#888888'
            self.btn_sm.label.set_text(txt_sm)
            self.btn_sm.color = col_sm
            self.btn_sm.label.set_color(col_txt)
        self.actualizar()

    def cambiar_it_pix(self, v):
        self.it_pix = int(v)
        self.actualizar()

    def cambiar_color(self, v):
        self.val_color = v
        self.actualizar()

    def cambiar_hue(self, v):
        self.val_hue = v
        self.actualizar()

    def cambiar_ang(self, v):
        self.angulo = v
        self.actualizar()

    def cambiar_it(self, v): 
        self.it_lin = int(v) # Guarda el nuevo nivel de detalle
        self.actualizar()     # Redibuja el fractal con ese nuevo nivel


    def cambiar_c_julia(self, texto):
        try:
            # Python usa 'j', el usuario usa 'i'
            self.julia_c = complex(texto.replace('i', 'j').replace(' ', ''))
            self.actualizar()
        except:
            # Si hay error, reseteamos al valor que estaba
            v = str(self.julia_c).replace('(','').replace(')','').replace('j','i')
            self.txt_julia.set_val(v)

    def _get_cached_geometry(self, clave, generador):
        if clave not in self.geom_cache:
            self.geom_cache[clave] = generador()
        return self.geom_cache[clave]

    def _get_grid(self, w, h):
        # Cachea la malla de coordenadas por tamaño y límites actuales
        key = (int(w), int(h), float(self.x_min), float(self.x_max), float(self.y_min), float(self.y_max))
        if key in self.grid_cache:
            return self.grid_cache[key]
        x_range = np.linspace(self.x_min, self.x_max, int(w), dtype=np.float64)
        y_range = np.linspace(self.y_min, self.y_max, int(h), dtype=np.float64)
        cx_grid, cy_grid = np.meshgrid(x_range, y_range)
        self.grid_cache[key] = (cx_grid, cy_grid)
        # keep cache small: if too many entries, remove oldest
        try:
            if len(self.grid_cache) > 8:
                k = next(iter(self.grid_cache))
                del self.grid_cache[k]
        except Exception:
            pass
        return cx_grid, cy_grid

# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-

    def on_mouse(self, e):
        if not self.en_exp or e.inaxes != self.ax: return

        if self.tipo not in self.info_math: return
    

        nom, form, dim = self.info_math[self.tipo]
        coord = f"C = {e.xdata:.4f} + {e.ydata:.4f}i" if self.tipo < 3 else f"X: {e.xdata:.3f} Y: {e.ydata:.3f}"
        self.txt_info.set_text(f"{nom}\n{form}\n{dim}\n{coord}")
        if self.mostrar_orbita and self.tipo < 3:
            ox, oy = [], []
            zx, zy = (0, 0) if self.tipo != 1 else (e.xdata, e.ydata)
            cx, cy = (e.xdata, e.ydata) if self.tipo != 1 else (-0.7, 0.27)
            for _ in range(60):
                ox.append(zx); oy.append(zy)
                if self.tipo == 2: nx, ny = zx*zx - zy*zy + cx, abs(2*zx*zy) + cy
                else: nx, ny = zx*zx - zy*zy + cx, 2*zx*zy + cy
                zx, zy = nx, ny
                if zx*zx + zy*zy > 10: break
            self.orbit_line.set_data(ox, oy)
            self.orbit_line.set_visible(True)
        else:
            self.orbit_line.set_visible(False)
        self.fig.canvas.draw_idle()

    def actualizar(self):
        if not self.en_exp: return
        c_lin = self.colores_lin[self.idx_col % 6]
        self.ax.set_facecolor('black')
        self.image.set_cmap(self.paletas[self.idx_col % len(self.paletas)])
        self.image.set_visible(False)
        self.orbit_line.set_visible(False)
        if hasattr(self, '_geometric_artists'):
            for art in self._geometric_artists:
                try:
                    art.remove()
                except:
                    pass
            self._geometric_artists = []


        # ACTUALIZAR AYUDA CON INSTRUCCIONES DE PORTAPAPELES
        h_text = "CONTROLES:\n\nESC: Menú\nC / c: Cambiar paleta\nR: Reset\np: Guardar imagen\nP: Guardar imagen en HD"
        if self.tipo < 3: 
            h_text += "\nO: Órbita"
            h_text += "\nA: Auto-Demo (Animar)"
            h_text += "\n\nCOPIAR COORD:\n1. Clic en casilla\n2. Ctrl + C"
            h_text += "\n\nPEGAR COORD:\n1. Clic en casilla\n2. Ctrl + V"
            
        self.txt_help.set_text(h_text)
        self.txt_palette.set_text(f"PALETA: {self.paletas[self.idx_col % len(self.paletas)]}")

        # El texto de info matemática lo actualiza on_mouse, 
        # así que aquí no hace falta crear nada nuevo.

        # INFO MATEMÁTICA (Banda Derecha)
        '''
        self.txt_info = self.fig.text(0.98, 0.25, "", color='white', fontsize=10, 
                                      family='monospace', ha='right', va='bottom',
                                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        '''

        #if self.tipo <= 2:
        #    self.box_x.set_val(f"{(self.x_min + self.x_max)/2:.4f}")
        #    self.box_y.set_val(f"{(self.y_min + self.y_max)/2:.4f}")
        #    self.box_z.set_val(f"{2.0 / (self.x_max - self.x_min):.1f}")



        if self.tipo == 3:
            # Aseguramos que it_sierp sea al menos 1 para que se vea el cuadrado inicial
            nivel = max(1, int(self.it_lin)) 
            cx, cy = self._get_grid(600, 600)
            img = _calcular_pixeles_njit(cx, cy, self.it_pix, 3, nivel, 0.0, 0.0, False, 2.0)
            
            img_modificado = (img * self.val_color + self.val_hue) % 1.0
            try:
                print(f"[DEBUG] actualizar tipo=3 img.shape={img.shape} min={img.min():.6f} max={img.max():.6f} x_min={self.x_min} x_max={self.x_max} y_min={self.y_min} y_max={self.y_max}")
            except Exception:
                pass
            self.image.set_data(img_modificado)
            try:
                self.image.set_clim(0.0, 1.0)
            except Exception:
                pass
            try:
                self.image.set_interpolation('nearest')
            except Exception:
                pass
            try:
                self.image.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
            except Exception:
                pass
            try:
                self.image.set_visible(True)
            except Exception:
                pass





        elif self.tipo == 4 or self.tipo == 10:
            # --- VISOR GENERAL DE AMBOS COPOS DE KOCH ---
            key = ('koch', self.tipo, self.it_lin)
            if key in self.geom_cache:
                lx, ly = self.geom_cache[key]
            else:
                lx, ly = [], []
                p1 = np.array([0.15, 0.25])
                p2 = np.array([0.5, 0.856])
                p3 = np.array([0.85, 0.25])
                if self.tipo == 4:
                    gen_koch(p1, p2, self.it_lin, lx, ly)
                    lx.append(p2[0]); ly.append(p2[1])
                    gen_koch(p2, p3, self.it_lin, lx, ly)
                    lx.append(p3[0]); ly.append(p3[1])
                    gen_koch(p3, p1, self.it_lin, lx, ly)
                    lx.append(p1[0]); ly.append(p1[1])
                else:
                    gen_koch(p2, p1, self.it_lin, lx, ly)
                    lx.append(p1[0]); ly.append(p1[1])
                    gen_koch(p1, p3, self.it_lin, lx, ly)
                    lx.append(p3[0]); ly.append(p3[1])
                    gen_koch(p3, p2, self.it_lin, lx, ly)
                    lx.append(p2[0]); ly.append(p2[1])
                self.geom_cache[key] = (lx, ly)

            artists = self.ax.fill(lx, ly, color=c_lin, alpha=0.3)
            self._geometric_artists.extend(artists)
            line_artist, = self.ax.plot(lx, ly, color=c_lin, lw=1.5)
            self._geometric_artists.append(line_artist)

# ---

        elif self.tipo == 5:
            key = ('dragon', self.it_lin)
            if key in self.geom_cache:
                lx, ly = self.geom_cache[key]
            else:
                lx, ly = [], []
                gen_dragon(np.array([0,0]), np.array([1,0]), self.it_lin, 1, lx, ly)
                self.geom_cache[key] = (lx, ly)
            line_artist, = self.ax.plot(lx, ly, color=c_lin, lw=1.5)
            self._geometric_artists.append(line_artist)
        elif self.tipo == 6:
            key = ('sierpinski', self.it_lin)
            if key in self.geom_cache:
                pts = self.geom_cache[key]
            else:
                pts = []
                gen_sierpinski(np.array([0,0]), np.array([1,0]), np.array([0.5,0.866]), self.it_lin, pts)
                self.geom_cache[key] = pts
            for tx, ty in pts:
                artists = self.ax.fill(tx, ty, color=c_lin, alpha=0.3)
                self._geometric_artists.extend(artists)
                line_artist, = self.ax.plot(tx, ty, color=c_lin, lw=1.5)
                self._geometric_artists.append(line_artist)
        elif self.tipo == 7:
            key = ('arbol', self.it_lin, self.angulo)
            if key in self.geom_cache:
                pts = self.geom_cache[key]
            else:
                pts = []
                gen_arbol(np.array([0.4, 0]), np.array([0.6, 0]), self.it_lin, pts, self.angulo)
                self.geom_cache[key] = pts
            for tx, ty in pts:
                artists = self.ax.fill(tx, ty, color=c_lin, alpha=0.3)
                self._geometric_artists.extend(artists)
                line_artist, = self.ax.plot(tx, ty, color=c_lin, lw=1.2)
                self._geometric_artists.append(line_artist)
        elif self.tipo == 8:
            key = ('pentagonos', self.it_lin)
            if key in self.geom_cache:
                pts = self.geom_cache[key]
            else:
                pts = []
                gen_pentagonos(np.array([0.5,0.5]), 0.4, np.pi/2, self.it_lin, pts)
                self.geom_cache[key] = pts
            for tx, ty in pts:
                artists = self.ax.fill(tx, ty, color=c_lin, alpha=0.4)
                self._geometric_artists.extend(artists)
                line_artist, = self.ax.plot(tx, ty, color=c_lin, lw=1.5)
                self._geometric_artists.append(line_artist)
        elif self.tipo == 9:
            caras = []; it_m = min(self.it_lin, 3)
            gen_cubo_iso(-0.5, -0.5, -0.5, 1.0, it_m, caras)
            for pts, cara in caras:
                if cara == "top": fcol = c_lin
                elif cara == "left": fcol = plt.cm.colors.to_hex(np.array(plt.cm.colors.to_rgb(c_lin))*0.8)
                else: fcol = plt.cm.colors.to_hex(np.array(plt.cm.colors.to_rgb(c_lin))*0.6)
                poly = np.array(pts)
                artists = self.ax.fill(poly[:,0], poly[:,1], facecolor=fcol, edgecolor='black', lw=0.3)
                self._geometric_artists.extend(artists)
        elif self.tipo == 11:
            # Ajustamos dinámicamente el detalle máximo a 6 para evitar que se congele
            nivel_h = min(6, max(1, int(self.it_lin)))
            key = ('hilbert', nivel_h)
            if key in self.geom_cache:
                lx, ly = self.geom_cache[key]
            else:
                lx, ly = [], []
                gen_hilbert(0, 0, 1, 0, 0, 1, nivel_h, lx, ly)
                self.geom_cache[key] = (lx, ly)
            line_artist, = self.ax.plot(lx, ly, color=c_lin, lw=1.5)
            self._geometric_artists.append(line_artist)
            self.ax.set_xlim(-0.05, 1.05); self.ax.set_ylim(-0.05, 1.05)
        elif self.tipo == 12:
            nivel_v = min(5, max(1, int(self.it_lin)))
            key = ('vicsek', nivel_v)
            if key in self.geom_cache:
                pts = self.geom_cache[key]
            else:
                pts = []
                gen_vicsek(0, 0, 1.0, nivel_v, pts)
                self.geom_cache[key] = pts
            for tx, ty in pts:
                line_artist, = self.ax.plot(tx, ty, color=c_lin, lw=1.2)
                self._geometric_artists.append(line_artist)
            self.ax.set_xlim(-0.05, 1.05); self.ax.set_ylim(-0.05, 1.05)
            
        elif self.tipo == 13:
            # Forzamos un límite seguro de detalle para que no se cuelgue la PC
            nivel_h = min(4, max(1, int(self.it_lin))) 
            key = ('hexflake', nivel_h)
            if key in self.geom_cache:
                pts = self.geom_cache[key]
            else:
                pts = []
                gen_hexflake(0.5, 0.5, 0.45, nivel_h, pts)
                self.geom_cache[key] = pts
            for tx, ty in pts: 
                artists = self.ax.fill(tx, ty, color=c_lin, alpha=0.3)
                self._geometric_artists.extend(artists)
                line_artist, = self.ax.plot(tx, ty, color=c_lin, lw=1.2)
                self._geometric_artists.append(line_artist)
            self.ax.set_xlim(0.0, 1.0)
            self.ax.set_ylim(0.0, 1.0)



        else:
            # --- MOTOR DINÁMICO PARA FRACTAL PERSONALIZADO ---
            jx, jy = self.julia_c.real, self.julia_c.imag
            tipo_calculo = self.tipo
            exponente_custom = 2.0 # Por defecto arranca al cuadrado
            
            if self.tipo == -1:
                # use cached exponent parsed on formula change
                exponente_custom = self.exponente_custom
                if self.es_modo_julia:
                    jx, jy = self.custom_c.real, self.custom_c.imag
                    tipo_calculo = 1
                else:
                    tipo_calculo = 0
            
            # Pasamos las iteraciones normales de tu slider en pantalla
            max_it_actual = self.it_pix
            
            # --- ¡OJO AQUÍ! Agregamos exponente_custom al final de la llamada ---
            # Use cached grid + specialized kernels when possible
            cx, cy = self._get_grid(600, 600)
            is_julia = (tipo_calculo == 1)
            if tipo_calculo == 2:
                img = _burning_ship_njit(cx, cy, max_it_actual, jx, jy, self.modo_smooth)
            elif exponente_custom is not None and exponente_custom != 2.0:
                img = _exp_fractal_njit(cx, cy, max_it_actual, is_julia, jx, jy, self.modo_smooth, exponente_custom)
            else:
                img = _mandelbrot_julia_njit(cx, cy, max_it_actual, is_julia, jx, jy, self.modo_smooth)
            # Fallback: si el kernel devolvió una imagen completamente uniforme, usa la versión original
            try:
                if np.isfinite(img).all():
                    if img.max() - img.min() < 1e-12:
                        img = calcular_pixeles(600, 600, max_it_actual, self.x_min, self.x_max,
                                               self.y_min, self.y_max, tipo_calculo, self.it_lin,
                                               jx, jy, self.modo_smooth, exponente_custom)
            except Exception:
                pass
            
            img_rgba = self._apply_palette_and_hue(img)
            try:
                print(f"[DEBUG] actualizar tipo!=3 img.shape={img.shape} min={img.min():.6f} max={img.max():.6f} x_min={self.x_min} x_max={self.x_max} y_min={self.y_min} y_max={self.y_max}")
            except Exception:
                pass
            self.image.set_data(img_rgba)
            try:
                self.image.set_clim(0.0, 1.0)
            except Exception:
                pass
            try:
                self.image.set_interpolation('nearest')
            except Exception:
                pass
            try:
                self.image.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
            except Exception:
                pass
            try:
                self.image.set_visible(True)
            except Exception:
                pass



        # --- CÁLCULO DE PROPORCIÓN PARA EVITAR EL APLASTADO ---
        ancho_mat = self.x_max - self.x_min
        alto_mat = self.y_max - self.y_min
        
        # Buscamos el lado más grande para que el fractal no se estire
        lado = max(ancho_mat, alto_mat)
        centro_x = (self.x_min + self.x_max) / 2
        centro_y = (self.y_min + self.y_max) / 2
        
        # Ajustamos los límites para que formen un CUADRADO real
        self.x_min, self.x_max = centro_x - lado/2, centro_x + lado/2
        self.y_min, self.y_max = centro_y - lado/2, centro_y + lado/2

        # Establecemos los límites en el eje
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        # 'auto' + nuestro cálculo manual = FRACTAL PERFECTO SIN ERRORES
        #self.ax.set_aspect('auto') 
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        # Aseguramos que la imagen use los límites finales
        try:
            if hasattr(self, 'image') and self.image is not None:
                self.image.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
                self.image.set_visible(True)
        except Exception:
            pass
        self.fig.canvas.draw_idle()




# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-


    def zoom(self, e):
        if not self.en_exp or e.inaxes != self.ax or e.xdata is None or e.ydata is None: 
            return
        
        f = 0.8 if e.button == 'up' else 1.25
        mx, my = e.xdata, e.ydata # Punto de anclaje
        
        # Zoom quirúrgico: las distancias al mouse se encogen/estiran por igual
        self.x_min = mx - (mx - self.x_min) * f
        self.x_max = mx + (self.x_max - mx) * f
        self.y_min = my - (my - self.y_min) * f
        self.y_max = my + (self.y_max - my) * f
        
        self.actualizar()


    # --- Slider fast-preview helpers ---
    def _cancel_slider_timer(self):
        try:
            if hasattr(self, '_slider_timer') and self._slider_timer is not None:
                try:
                    self._slider_timer.stop()
                except:
                    pass
                self._slider_timer = None
        except:
            self._slider_timer = None

    def _schedule_preview(self):
        # Stop previous timer and start a new one to finalize the full render
        self._cancel_slider_timer()
        # Render an immediate low-res preview
        try:
            self._render_preview()
        except Exception:
            pass
        # Schedule finalize after short delay (ms)
        try:
            t = self.fig.canvas.new_timer(interval=300)
            t.add_callback(self._finalize_slider_change)
            t.start()
            self._slider_timer = t
        except Exception:
            self._slider_timer = None

    def _render_preview(self):
        # Fast, low-resolution preview used while dragging sliders
        if not self.en_exp: return
        if self.tipo >= 3 and self.tipo != -1:
            return
        # preview target size (lower than full-res 600)
        pre_w = max(80, int(600 * 0.25))
        pre_h = pre_w

        # determine calculation mode and parameters (mirrors actualizar logic)
        jx, jy = self.julia_c.real, self.julia_c.imag
        tipo_calculo = self.tipo
        # determine calculation mode and parameters (mirrors actualizar logic)
        tipo_calculo = self.tipo
        # use cached exponent parsed when user changed formula
        exponente_custom = self.exponente_custom if self.tipo == -1 else 2.0
        if self.tipo == -1:
            if self.es_modo_julia:
                jx, jy = self.custom_c.real, self.custom_c.imag
                tipo_calculo = 1
            else:
                tipo_calculo = 0

        # compute preview image
        try:
            # compute preview image using cached grid and fast kernels
            cx, cy = self._get_grid(pre_w, pre_h)
            is_julia = (tipo_calculo == 1)
            if tipo_calculo == 2:
                img = _burning_ship_njit(cx, cy, self.it_pix, jx, jy, self.modo_smooth)
            elif exponente_custom is not None and exponente_custom != 2.0:
                img = _exp_fractal_njit(cx, cy, self.it_pix, is_julia, jx, jy, self.modo_smooth, exponente_custom)
            else:
                img = _mandelbrot_julia_njit(cx, cy, self.it_pix, is_julia, jx, jy, self.modo_smooth)
            try:
                if np.isfinite(img).all() and (img.max() - img.min() < 1e-12):
                    img = calcular_pixeles(pre_w, pre_h, self.it_pix, self.x_min, self.x_max,
                                           self.y_min, self.y_max, tipo_calculo, self.it_lin,
                                           jx, jy, self.modo_smooth, exponente_custom)
            except Exception:
                pass
            img_rgba = self._apply_palette_and_hue(img)
            try:
                print(f"[DEBUG] preview img shape={img.shape} min={img.min():.6f} max={img.max():.6f}")
            except Exception:
                pass
            self.image.set_data(img_rgba)
            try:
                self.image.set_clim(0.0, 1.0)
            except Exception:
                pass
            try:
                self.image.set_interpolation('nearest')
            except Exception:
                pass
            try:
                self.image.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
            except Exception:
                pass
            try:
                self.image.set_visible(True)
            except Exception:
                pass
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _finalize_slider_change(self):
        # Called after user stops moving the slider: perform full update
        self._cancel_slider_timer()
        try:
            self.actualizar()
        except Exception:
            pass

    def cambiar_it_pix_fast(self, v):
        self.it_pix = int(v)
        self._schedule_preview()

    def cambiar_hd_res(self, v):
        self.hd_res = int(v)
        # No actualiza automáticamente; solo ajusta el valor para guardar con P

    def cambiar_color_fast(self, v):
        self.val_color = v
        self._schedule_preview()

    def cambiar_hue_fast(self, v):
        self.val_hue = v
        self._schedule_preview()






    def ir_a_coords(self, texto):
        try:
            nx = float(self.box_x.text)
            ny = float(self.box_y.text)
            nz = float(self.box_z.text)
            
            ancho = 2.0 / (nz if nz > 0 else 1)
            # Forzamos que sea un cuadrado perfecto para evitar deformaciones
            self.x_min, self.x_max = nx - ancho/2, nx + ancho/2
            self.y_min, self.y_max = ny - ancho/2, ny + ancho/2
            
            self.actualizar()
        except:
            pass


    def capturar_coordenadas_actuales(self, _):
        if self.tipo <= 2:
            # 1. Calculamos los valores reales actuales del plano complejo
            centro_x = (self.x_min + self.x_max) / 2
            centro_y = (self.y_min + self.y_max) / 2
            nivel_zoom = 2.0 / (self.x_max - self.x_min)

            # 2. Apagamos las señales internas para evitar que llamen a ir_a_coords
            self.box_x.eventson = False
            self.box_y.eventson = False
            self.box_z.eventson = False

            # 3. Forzamos la inserción de los textos usando el método nativo oficial
            self.box_x.set_val(f"{centro_x:.14f}")
            self.box_y.set_val(f"{centro_y:.14f}")
            self.box_z.set_val(f"{nivel_zoom:.2f}")

            # 4. Volvemos a encender las señales para cuando quieras presionar ENTER manualmente
            self.box_x.eventson = True
            self.box_y.eventson = True
            self.box_z.eventson = True

            # 5. Refrescamos la UI de forma pasiva
            self.fig.canvas.draw_idle()







    def tecla(self, e):
        if e.key is not None and e.key.lower() == 'c':
            self.idx_col += 1
            if self.en_exp:
                self.actualizar()
            else:
                self.menu_principal()
        elif e.key == 'r' and self.en_exp:
            self.x_min, self.x_max, self.y_min, self.y_max = self.x_orig, self.x_max_orig, self.y_orig, self.y_max_orig
            self.actualizar()
        elif e.key == 'p' and self.en_exp:
            # Fast save: grab the current displayed image and save it in background
            try:
                self._save_image_async()
                print("[Captura] Guardando imagen rápida en background...")
            except Exception:
                pass
        elif e.key == 'P' and self.en_exp:
            ancho_px = alto_px = int(self.hd_res)
            max_it_render = int(self.it_pix)
            print(f"Iniciando renderizado Ultra-HD ({ancho_px}x{alto_px}) con {max_it_render} iteraciones...")
            print("Esto puede tardar unos segundos, usaremos todos los núcleos de tu PC.")
            # Determine calculation mode and exponent
            jx, jy = self.julia_c.real, self.julia_c.imag
            tipo_calculo = self.tipo
            exponente_custom = self.exponente_custom if self.tipo == -1 else 2.0
            if self.tipo == -1:
                if self.es_modo_julia:
                    jx, jy = self.custom_c.real, self.custom_c.imag
                    tipo_calculo = 1
                else:
                    tipo_calculo = 0
            # generate grid and compute high-res image using specialized kernels
            cx, cy = self._get_grid(ancho_px, alto_px)
            is_julia = (tipo_calculo == 1)
            if tipo_calculo == 2:
                img_hq = _burning_ship_njit(cx, cy, max_it_render, jx, jy, self.modo_smooth)
            elif exponente_custom is not None and exponente_custom != 2.0:
                img_hq = _exp_fractal_njit(cx, cy, max_it_render, is_julia, jx, jy, self.modo_smooth, exponente_custom)
            else:
                img_hq = _mandelbrot_julia_njit(cx, cy, max_it_render, is_julia, jx, jy, self.modo_smooth)
            # color adjustments (apply palette + hue rotation)
            img_rgba_hq = self._apply_palette_and_hue(img_hq)
            # render to a saving figure
            fig_save = plt.figure(figsize=(ancho_px/100, alto_px/100), dpi=100, facecolor='black')
            ax_save = fig_save.add_axes([0, 0, 1, 1])
            ax_save.imshow(img_rgba_hq, origin='lower', aspect='auto')
            ax_save.axis('off')
            
            # --- CÁLCULO DE DATOS EXACTOS PARA EL NOMBRE ÚNICO ---
            centro_x = (self.x_min + self.x_max) / 2
            centro_y = (self.y_min + self.y_max) / 2
            nivel_zoom = 2.0 / (self.x_max - self.x_min)
            
            # Formateamos el nombre reemplazando espacios por guiones bajos
            clean_name = self.nom_f.replace(" ", "_")
            
            # 4. Guardado final con la nomenclatura exacta solicitada
            nombre_archivo = f"Fractal_Image_HD_{clean_name}_[{centro_x:.14f}_{centro_y:.14f}_{nivel_zoom:.2f}]_it-{max_it_render}_{ancho_px}x{alto_px}.png"
            fig_save.savefig(nombre_archivo, facecolor='black', edgecolor='none', pad_inches=0)
            plt.close(fig_save) # Liberamos memoria
            
            print(f"¡TERMINADO! Tu fractal HD está listo: {nombre_archivo}")

        elif e.key == 'escape': 
            self.menu_principal()

        elif e.key == 'o':
            self.toggle_orbita()
            
        # (Dentro de tu def tecla, añade estos bloques correspondientes)
        elif e.key == 'a' and self.en_exp:
            self.toggle_modo_demo()
            
        elif e.key == 'escape': 
            # Si volvemos al menú, apagamos el temporizador para liberar memoria
            if self.animando:
                self.toggle_modo_demo()
            self.menu_principal()



        # --- SOLUCIÓN DE COPIAR Y PEGAR ACTUALIZADA ---
        elif e.key in ['ctrl+c', 'ctrl+v'] and self.en_exp and self.tipo <= 2:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw() # Oculta la ventana auxiliar de Tkinter
            
            # Si el usuario hace Ctrl+C/V pero aún no hizo clic en ninguna caja, apuntamos a la de X por defecto
            if not hasattr(self, 'caja_activa') or self.caja_activa is None:
                self.caja_activa = self.box_x
                
            if self.caja_activa is not None:
                if e.key == 'ctrl+c':
                    # COPIAR AUTOMÁTICO: Como Matplotlib no deja seleccionar texto con mouse,
                    # al presionar Ctrl+C copiamos TODO el contenido de la casilla activa directamente.
                    root.clipboard_clear()
                    root.clipboard_append(self.caja_activa.text)
                    print(f"[Portapapeles] Copiado de la casilla: {self.caja_activa.text}")
                elif e.key == 'ctrl+v':
                    # PEGAR AUTOMÁTICO: Reemplaza el texto por lo que tengas en el portapapeles
                    try:
                        clipboard_text = root.clipboard_get().strip()
                        self.caja_activa.eventson = False
                        self.caja_activa.set_val(clipboard_text)
                        self.caja_activa.eventson = True
                        print(f"[Portapapeles] Pegado en la casilla: {clipboard_text}")
                    except:
                        pass
            root.destroy()

# ---

    def toggle_modo_demo(self):
        self.animando = not self.animando
        if self.animando:
            # Creamos un temporizador nativo de Matplotlib (ejecuta la función cada 30ms)
            self.timer_demo = self.fig.canvas.new_timer(interval=30)
            self.timer_demo.add_callback(self.animar_cuadro_demo)
            self.timer_demo.start()
            print("[Modo Demo] Activado - Animación en ejecución.")
        else:
            if self.timer_demo is not None:
                self.timer_demo.stop()
                self.timer_demo = None
            print("[Modo Demo] Desactivado - Estado estático.")

    def animar_cuadro_demo(self):
        if not self.en_exp:
            return
            
        # 1. Variamos sutilmente el Hue de color cíclicamente entre 0.0 y 1.0
        self.val_hue = (self.val_hue + 0.001) % 1.0
        
        # Si el slider de Hue existe en la interfaz, actualizamos su barra visual en vivo
        if hasattr(self, 'slid_hue') and self.slid_hue in self.controles:
            self.slid_hue.eventson = False
            self.slid_hue.set_val(self.val_hue)
            self.slid_hue.eventson = True

        # 2. Solo aplicamos auto-zoom dinámico en los fractales matemáticos de píxeles (Mandelbrot, Julia, Ship, Custom)
        if self.tipo <= 2 or self.tipo == -1:
            centro_x = (self.x_min + self.x_max) / 2
            centro_y = (self.y_min + self.y_max) / 2
            ancho = self.x_max - self.x_min
            
            # Control dinámico de límites para no romper la precisión Float64
            zoom_actual = 2.0 / ancho
            if zoom_actual > 50000000: # Si se acerca demasiado, invertimos la marcha para alejarse
                self.dir_zoom = -1
            elif zoom_actual < 1.5:    # Si se aleja al tamaño original, vuelve a entrar
                self.dir_zoom = 1
                
            # Factor de escala suave por cuadro (0.985 acerca, 1.015 aleja)
            factor = 0.990 if self.dir_zoom == 1 else 1.010
            nuevo_ancho = (ancho * factor) / 2
            
            self.x_min, self.x_max = centro_x - nuevo_ancho, centro_x + nuevo_ancho
            self.y_min, self.y_max = centro_y - nuevo_ancho, centro_y + nuevo_ancho

        # 3. Solicitamos un redibujado síncrono suave de los datos
        self.actualizar()



    def toggle_orbita(self):
        # Solo tiene sentido activar órbitas si estamos explorando Mandelbrot, Julia o Burning Ship
        if self.en_exp and self.tipo < 3:
            self.mostrar_orbita = not self.mostrar_orbita
            
            # Si se apaga, borramos inmediatamente la línea actual de la pantalla
            if not self.mostrar_orbita and self.puntos_orbita is not None:
                self.puntos_orbita.remove()
                self.puntos_orbita = None
                self.fig.canvas.draw_idle()


    def viajar_a_destino(self, lug):
        # Calculamos el rango basado en el zoom del destino
        ancho = 2.0 / lug['z']
        # Mantenemos la proporción actual de la ventana
        ratio = (self.x_max - self.x_min) / (self.y_max - self.y_min)
        alto = ancho / ratio
        
        self.x_min, self.x_max = lug['re'] - ancho/2, lug['re'] + ancho/2
        self.y_min, self.y_max = lug['im'] - alto/2, lug['im'] + alto/2
        
        # Volvemos a abrir el explorador con estas nuevas coordenadas
        conf_destino = (self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f)
        self.abrir(self.tipo, conf_destino)

    def mostrar_hub(self, _):
        if self.tipo not in self.destinos: return
        self.limpiar()
        self.fig.clf()
        self.fig.patch.set_facecolor('black')
        
        lugares = self.destinos[self.tipo]
        plt.suptitle(f"Explorador de Destinos: {self.nom_f}", color='gold', fontsize=15, y=0.95)

        for i, lug in enumerate(lugares):
            # Miniatura (ajustada para que quepan 3 en una fila)
            ax_min = self.fig.add_axes([0.05 + i*0.32, 0.4, 0.26, 0.4], facecolor='black')
            
            # Render ultra-rápido para el HUB
            ancho_m = 2.0 / lug['z']
            img = calcular_pixeles(120, 120, 40, lug['re']-ancho_m/2, lug['re']+ancho_m/2, 
                                   lug['im']-ancho_m/2, lug['im']+ancho_m/2, 
                                   self.tipo, self.it_lin, self.julia_c.real, self.julia_c.imag)
            
            ax_min.imshow(img, cmap=self.paletas[self.idx_col % len(self.paletas)], origin='lower', aspect='auto')
            ax_min.set_title(lug['nombre'], color='white', fontsize=10)
            ax_min.axis('off')

            # Botón "Viajar" justo debajo de cada miniatura
            ax_b = self.fig.add_axes([0.08 + i*0.32, 0.32, 0.2, 0.05])
            btn = Button(ax_b, 'VIAJAR', color='#111111', hovercolor='#004400')
            btn.label.set_color('#00ff00')
            # El lambda l=lug asegura que cada botón guarde su destino correcto
            btn.on_clicked(lambda e, l=lug: self.viajar_a_destino(l))
            self.controles.append(btn)

        # Botón para cancelar y volver al fractal donde estábamos
        ax_v = self.fig.add_axes([0.42, 0.1, 0.16, 0.06])
        btn_v = Button(ax_v, 'VOLVER', color='#222222')
        btn_v.label.set_color('white')
        btn_v.on_clicked(lambda e: self.abrir(self.tipo, (self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f)))
        self.controles.append(btn_v)
        
        self.fig.canvas.draw()

        '''
        Lugares interesantes que encontré en Mandelbrot:
            (No se que nombre ponerle): -0.90768376224943 0.26786740030236 125000 (se ve mejor con 500 iteraciones)
            Isla de Julia 1: -1.76897875927975 0.00231501640996 950000 (se ve mejor con 200 iteraciones)
            Isla de Julia 2: -1.74505268893232 0.00567039476653 20000 (se ve mejor con 200 iteraciones)
            Isla de Julia 3: -1.74505268893232 0.00567039476653 2349295580266.30 (se ve mejor con 800 iteraciones)
            Valle de los Elefantes: 0.2508 0 18000 (se ve mejor con 800 iteraciones)
            Ciudad: -0.71676612825470 0.29301036148871 2302026 (se ve mejor con 800 iteraciones)
            El Sol: -0.74916057574342 0.10051009064492 11640 (se ve mejor con 800 iteraciones)
                '''



    def actualizar_desde_hub(self):
        conf_actual = (self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f)
        self.abrir(self.tipo, conf_actual)



if __name__ == "__main__":
    AppFractales()


'''
Lista de cosas que quiero añadir a mi codigo:
    - Optimizacion
    - Fractal de Newton
    - MAS OPTIMIZACION
    - Otras cosas
    '''