import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from numba import jit, prange

plt.rcParams['toolbar'] = 'None'

@jit(nopython=True, parallel=True) # Agregamos parallel=True para máxima velocidad
def calcular_pixeles(ancho, alto, max_iter, x_min, x_max, y_min, y_max, tipo, it_sierp, jx, jy, modo_smooth):
    x_range = np.linspace(x_min, x_max, ancho)
    y_range = np.linspace(y_min, y_max, alto)
    fractal = np.zeros((alto, ancho), dtype=np.float64)
    
    # Usamos prange para que Numba use todos los núcleos de tu PC
    for i in prange(alto):
        for j in range(ancho):
            cx, cy = x_range[j], y_range[i]
            

            if tipo == 3: 
                # LÓGICA DE ALFOMBRA CORREGIDA
                tx, ty = cx, cy
                it = 0
                dentro = False
                
                # Verificamos si el punto está dentro del área de la alfombra
                if tx >= 0.0 and tx <= 1.0 and ty >= 0.0 and ty <= 1.0:
                    while it < it_sierp:
                        # Usamos valores fijos para evitar errores de precisión de Numba
                        if (tx > 0.333333 and tx < 0.666666) and (ty > 0.333333 and ty < 0.666666):
                            break
                        tx = (tx * 3.0) % 1.0
                        ty = (ty * 3.0) % 1.0
                        it += 1
                    
                    if it == it_sierp:
                        dentro = True
                
                # Asignamos 1.0 para que se vea con color, 0.0 para el fondo
                fractal[i, j] = 0.2 if dentro else 1.0



            

            else: # --- LÓGICA DE MANDELBROT / JULIA / BURNING SHIP ---
                if tipo == 1: zx, zy, sx, sy = cx, cy, jx, jy
                else: zx, zy, sx, sy = 0.0, 0.0, cx, cy
                
                it = 0
                while zx*zx + zy*zy <= 3000.0 and it < max_iter:
                    if tipo == 2: nx, ny = zx*zx - zy*zy + sx, abs(2*zx*zy) + sy
                    else: nx, ny = zx*zx - zy*zy + sx, 2*zx*zy + sy
                    zx, zy = nx, ny
                    it += 1
                
                if it < max_iter and modo_smooth:
                    log_zn = np.log(zx*zx + zy*zy) / 2
                    nu = np.log(log_zn / np.log(2)) / np.log(2)
                    fractal[i, j] = (it + 1 - nu) / max_iter
                else:
                    fractal[i, j] = it / max_iter
    return fractal





# --- FUNCIONES GEOMÉTRICAS ---
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





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-





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
    
    # Esta es la línea mágica que inclina el árbol
    # Usamos alpha para decidir dónde se corta la hipotenusa
    pm = p4 + alpha * (p3 - p4) + np.sqrt(alpha * (1 - alpha)) * np.array([-(p3[1]-p4[1]), (p3[0]-p4[0])])
    
    lista.append(([p1[0], p2[0], p3[0], p4[0], p1[0]], [p1[1], p2[1], p3[1], p4[1], p1[1]]))
    if it > 1:
        gen_arbol(p4, pm, it-1, lista, alpha)
        gen_arbol(pm, p3, it-1, lista, alpha)





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-





def gen_cubo_iso(x, y, z, tam, it, lista):
    # Proyección isométrica: x_2d = (x-y)*cos(30), y_2d = (x+y)*sin(30)-z
    def p(px, py, pz):
        return [(px - py) * 0.866, (px + py) * 0.5 - pz]

    if it == 0:
        # Definimos las 3 caras visibles del cubito
        # Cara Superior
        lista.append(([p(x,y,z+tam), p(x+tam,y,z+tam), p(x+tam,y+tam,z+tam), p(x,y+tam,z+tam)], "top"))
        # Cara Frontal Izquierda
        lista.append(([p(x,y,z), p(x+tam,y,z), p(x+tam,y,z+tam), p(x,y,z+tam)], "left"))
        # Cara Frontal Derecha
        lista.append(([p(x+tam,y,z), p(x+tam,y+tam,z), p(x+tam,y+tam,z+tam), p(x+tam,y,z+tam)], "right"))
        return

    nt = tam / 3
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                # Lógica de Menger: quitar centros de caras y centro total
                agujeros = (1 if dx==1 else 0) + (1 if dy==1 else 0) + (1 if dz==1 else 0)
                if agujeros < 2:
                    gen_cubo_iso(x+dx*nt, y+dy*nt, z+dz*nt, nt, it-1, lista)

    if it == 0:
        # Proyección Isométrica simple:
        # x' = (x - y) * cos(30°)
        # y' = (x + y) * sin(30°) - z
        f = lambda px, py, pz: np.array([(px - py) * 0.866, (px + py) * 0.5 - pz])
        
        # Definimos las caras visibles del cubito
        p = [f(x,y,z), f(x+tam,y,z), f(x+tam,y+tam,z), f(x,y+tam,z), 
             f(x,y,z+tam), f(x+tam,y,z+tam), f(x+tam,y+tam,z+tam), f(x,y+tam,z+tam)]
        
        # Cara Superior, Derecha y Frontal
        lista.append(([p[4][0], p[5][0], p[6][0], p[7][0], p[4][0]], [p[4][1], p[5][1], p[6][1], p[7][1], p[4][1]])) # Top
        lista.append(([p[1][0], p[5][0], p[6][0], p[2][0], p[1][0]], [p[1][1], p[5][1], p[6][1], p[2][1], p[1][1]])) # Right
        lista.append(([p[0][0], p[1][0], p[5][0], p[4][0], p[0][0]], [p[0][1], p[1][1], p[5][1], p[4][1], p[0][1]])) # Front
        return

    nuevo_tam = tam / 3
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                # Lógica de la Esponja: eliminar el centro de cada cara y el corazón
                agujeros = 0
                if dx == 1: agujeros += 1
                if dy == 1: agujeros += 1
                if dz == 1: agujeros += 1
                if agujeros < 2:
                    gen_menger_isometrico(x + dx*nuevo_tam, y + dy*nuevo_tam, z + dz*nuevo_tam, nuevo_tam, it-1, lista)

    n = 3**it
    cubo = np.ones((n, n, n), dtype=np.bool_)
    for i in range(it):
        p = 3**i
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    # Lógica de Sierpinski 3D optimizada
                    m = 0
                    if (x // p) % 3 == 1: m += 1
                    if (y // p) % 3 == 1: m += 1
                    if (z // p) % 3 == 1: m += 1
                    if m >= 2:
                        cubo[x, y, z] = False
    return cubo

    # Creamos una matriz 3D de ceros y unos
    n = 3**it
    cubo = np.ones((n, n, n), dtype=bool)
    
    for i in range(it):
        paso = 3**i
        # Generamos la máscara de agujeros
        # Si la posición x, y o z en base 3 tiene dos "1"s, se elimina
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    # Lógica de Sierpinski 3D: 
                    # si al menos dos coordenadas tienen el dígito 1 en la posición i
                    if ( (x // paso) % 3 == 1 and (y // paso) % 3 == 1 ) or \
                       ( (y // paso) % 3 == 1 and (z // paso) % 3 == 1 ) or \
                       ( (x // paso) % 3 == 1 and (z // paso) % 3 == 1 ):
                        cubo[x, y, z] = False
    return cubo





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-





class AppFractales:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8), facecolor='black')
        self.julia_c = complex(-0.7, 0.27) # Valor inicial
        self.colores_lin = ['#00ffff', '#ff00ff', '#ffffff', '#00ff00', '#ffaa00', '#ff4444']
        self.paletas = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'gnuplot2', 'ocean', 'gist_earth', 'terrain', 'cubehelix', 'hot']
        self.idx_col, self.it_lin, self.en_exp = 0, 4, False
        self.mostrar_orbita = False
        self.puntos_orbita = None
        self.controles = []
        self.info_math = {
            0: ("Mandelbrot", r"$Z_{n+1} = Z_n^2 + C$", "D ≈ 2.0"),
            1: ("Julia", r"$Z_{n+1} = Z_n^2 + C_fix$", "D ≈ 2.0"),
            2: ("Burning Ship", r"$Z_{n+1} = (|Re| + i|Im|)^2 + C$", "D ≈ 2.0"),
            3: ("Alfombra Sierpinski", "Remoción central 1/9", "D ≈ 1.89"),
            4: ("Copo de Koch", "L_{n+1} = 4/3 L_n", "D ≈ 1.26"),
            5: ("Curva del Dragón", "Plegado de 90°", "D = 2.0"),
            6: ("Triángulo Sierpinski", "Sucesión de 3 áreas", "D ≈ 1.58"),
            7: ("Árbol de Pitágoras", "Crecimiento binario", "D ≈ 2.0"),
            8: ("Pentágono Áureo", "Simetría n-flake", "D ≈ 1.67")
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
         # Añade esto al final de tu __init__
        self.it_pix = 123      # Iteraciones iniciales
        self.val_color = 0.9   # Amplitud de color inicial
        self.val_hue = 0.0     # Desplazamiento inicial
        self.modo_smooth = True # Smooth desactivado





        self.menu_principal()
        plt.show()

    def limpiar(self):
        for c in self.controles:
            # Desconectamos los eventos antes de borrar el eje
            if hasattr(c, 'disconnect_events'):
                c.disconnect_events()
            if hasattr(c, 'ax'):
                c.ax.remove()
        self.controles = []

    def menu_principal(self):
        self.en_exp = False
        self.it_lin = 4
        self.limpiar()
        self.fig.clf()
        
        # 1. LISTA DE 10 FRACTALES CON EL HUECO INCLUIDO
        self.conf_menu = [
            # Mandelbrot: Centrado en el cardioide
            (-2.1, 0.6, -1.35, 1.35, 0, "Mandelbrot"),
    
            # Julia: Cuadrado de 3.0 x 3.0
            (-1.5, 1.5, -1.5, 1.5, 1, "Julia"),
    
            # Burning Ship: Cuadrado de 2.6 x 2.6
            (-1.8, 0.8, -1.8, 0.8, 2, "Burning Ship"),

            # --- AQUÍ AGREGO EL HUECO (Posición 4 en la grilla, índice 3) ---
            (0.0, 0.0, 0.0, 0.0, -1, "Personalizado"),
    
            # Alfombra Sierpinski: Ahora pasa a la fila de abajo/siguiente lugar automáticamente
            (0, 1, 0, 1, 3, "Alfombra"),
    
            # Copo de Koch
            (-0.2, 1.2, -0.4, 1.0, 4, "Koch"),
    
            # Curva del Dragón
            (-0.4, 1.4, -0.7, 1.1, 5, "Dragón"),
    
            # Triángulo Sierpinski
            (-0.1, 1.1, -0.1, 1.1, 6, "Triángulo"),
    
            # Árbol de Pitágoras
            (-0.2, 1.2, -0.3, 1.1, 7, "Árbol"),
    
            # Pentágono Áureo
            (-0.1, 1.1, -0.1, 1.1, 8, "Pentágono")
        ]

        # 2. Dibujamos en dos filas (5 arriba, 5 abajo) para que no se amontonen
        for i, conf in enumerate(self.conf_menu):
            fila = i // 5  
            columna = i % 5
            x_pos = 0.05 + columna * 0.185
            y_pos = 0.62 if fila == 0 else 0.22 
            
            c_lin = self.colores_lin[self.idx_col % 6]
            tipo = conf[4] 

            # --- TRUCO: Solo si NO es el hueco, dibujamos ---
            if tipo != -1:
                # 1. Crear el eje de la miniatura
                ax = self.fig.add_axes([x_pos, y_pos, 0.15, 0.22], facecolor='black')
                
                if tipo == 4:
                    lx, ly = [], []
                    gen_koch(np.array([0,0]), np.array([1,0]), 2, lx, ly)
                    ax.plot(lx, ly, color=c_lin, lw=1)
                elif tipo == 5:
                    lx, ly = [], []
                    gen_dragon(np.array([0,0]), np.array([1,0]), 6, 1, lx, ly)
                    ax.plot(lx, ly, color=c_lin, lw=1)
                elif tipo == 6:
                    pts = []
                    gen_sierpinski(np.array([0,0]), np.array([1,0]), np.array([0.5,0.86]), 2, pts)
                    for tx, ty in pts: ax.fill(tx, ty, color=c_lin, alpha=0.3); ax.plot(tx, ty, color=c_lin, lw=1)
                elif tipo == 7:
                    pts = []
                    gen_arbol(np.array([0.45,0]), np.array([0.55,0]), 4, pts)
                    for tx, ty in pts: ax.plot(tx, ty, color=c_lin, lw=1)
                elif tipo == 8:
                    pts = []
                    gen_pentagonos(np.array([0.5,0.5]), 0.4, np.pi/2, 2, pts)
                    for tx, ty in pts: ax.fill(tx, ty, color=c_lin, alpha=0.4); ax.plot(tx, ty, color=c_lin, lw=1)
                else:
                    img = calcular_pixeles(80, 80, 20, conf[0], conf[1], conf[2], conf[3], tipo, 3, -0.7, 0.27, False)
                    ax.imshow(img, cmap=self.paletas[self.idx_col % 12], origin='lower', aspect='auto')
                
                ax.axis('off')
                ax.set_title(conf[5], color='white', fontsize=8, pad=2)
                
                # 2. Botón "Ver" (También adentro del IF para que no salga en el hueco)
                ax_b = self.fig.add_axes([x_pos + 0.025, y_pos - 0.05, 0.1, 0.035])
                btn = Button(ax_b, 'Ver', color='#222222', hovercolor='#444444')
                btn.label.set_color('white')
                btn.label.set_fontsize(7)
                btn.on_clicked(lambda e, t=tipo, c=conf: self.abrir(t, c))
                self.controles.append(btn)

        self.fig.canvas.draw()





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-




    def abrir(self, tipo, c):
        self.limpiar()
        self.en_exp = True
        self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f = c
        self.x_orig, self.y_orig = self.x_min, self.y_min 
        self.x_max_orig, self.y_max_orig = self.x_max, self.y_max
        
        self.fig.clf()
        self.ax = self.fig.add_subplot(111, facecolor='black')
        self.fig.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.9)

        
        # TEXTOS FIJOS (Subimos info_math a 0.45 para dar más espacio)
        self.txt_help = self.fig.text(0.02, 0.5, "", color='#888888', fontsize=9, family='monospace', va='center')
        self.txt_info = self.fig.text(0.98, 0.40, "", color='white', fontsize=10, 
                                      family='monospace', ha='right', va='bottom',
                                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Botón Volver
        ax_v = self.fig.add_axes([0.02, 0.02, 0.08, 0.04])
        btn = Button(ax_v, '← Menú', color='#333333'); btn.label.set_color('white')
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

        # --- CASILLAS DE NAVEGACIÓN RÁPIDA (Abajo a la izquierda) ---
        if tipo <= 2:
            ax_x = self.fig.add_axes([0.15, 0.02, 0.20, 0.03])
            self.box_x = TextBox(ax_x, 'Re ', initial=f"{(self.x_min+self.x_max)/2:.2f}")
            self.box_x.label.set_color('#ffffff')
        
        if tipo <= 2:
            ax_y = self.fig.add_axes([0.40, 0.02, 0.20, 0.03])
            self.box_y = TextBox(ax_y, 'Im ', initial=f"{(self.y_min+self.y_max)/2:.2f}")
            self.box_y.label.set_color('#ffffff')
        
        if tipo <= 2:
            ax_z = self.fig.add_axes([0.65, 0.02, 0.20, 0.03])
            self.box_z = TextBox(ax_z, 'Zoom ', initial="1")
            self.box_z.label.set_color('#ffffff')

        if tipo <= 2:
        # Conectamos las tres a la misma función
            self.box_x.on_submit(self.ir_a_coords)
            self.box_y.on_submit(self.ir_a_coords)
            self.box_z.on_submit(self.ir_a_coords)
        
            self.controles.extend([self.box_x, self.box_y, self.box_z])

    
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
            self.slid_ang.on_changed(self.cambiar_ang)
            self.controles.append(self.slid_ang)

        # --- NUEVOS CONTROLES DE RENDERIZADO (Solo para tipos 0, 1, 2) ---
        if tipo < 3:
            # Slider de Iteraciones
            ax_it = self.fig.add_axes([0.86, 0.65, 0.12, 0.02], facecolor='#222222')
            self.slid_it = Slider(ax_it, 'Iter ', 0, 777, valinit=self.it_pix, valfmt='%d')
            self.slid_it.label.set_color('white')
            self.slid_it.on_changed(self.cambiar_it_pix)
            self.controles.append(self.slid_it)

            # Slider de Amplitud de Color (Color)
            ax_c = self.fig.add_axes([0.86, 0.60, 0.12, 0.02], facecolor='#222222')
            self.slid_col = Slider(ax_c, 'Rango ', 0.9, 1.0, valinit=self.val_color)
            self.slid_col.label.set_color('white')
            self.slid_col.on_changed(self.cambiar_color)
            self.controles.append(self.slid_col)

            # Slider de Hue (Desplazamiento)
            ax_h = self.fig.add_axes([0.86, 0.55, 0.12, 0.02], facecolor='#222222')
            self.slid_hue = Slider(ax_h, 'Hue ', 0.0, 1.0, valinit=self.val_hue)
            self.slid_hue.label.set_color('white')
            self.slid_hue.on_changed(self.cambiar_hue)
            self.controles.append(self.slid_hue)

            # Botón para activar/desactivar Smooth
            ax_sm = self.fig.add_axes([0.86, 0.50, 0.12, 0.03])
            txt_sm = 'Smooth: ON' if self.modo_smooth else 'Smooth: OFF'
            col_sm = '#004400' if self.modo_smooth else '#440000'
            self.btn_sm = Button(ax_sm, txt_sm, color=col_sm, hovercolor='#555555')
            self.btn_sm.label.set_color('white')
            self.btn_sm.on_clicked(self.toggle_smooth)
            self.controles.append(self.btn_sm)



        self.actualizar()





# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-





    def toggle_smooth(self, _):
        self.modo_smooth = not self.modo_smooth
        self.actualizar()
        # Re-abrimos para que el botón cambie de color y texto (o usa set_text)
        self.abrir(self.tipo, (self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f))

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

# -E-N-V-I-A-R---P-O-R---P-A-R-T-E-S-

    def on_mouse(self, e):
        if not self.en_exp or e.inaxes != self.ax: return
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
            if self.puntos_orbita: self.puntos_orbita.remove()
            self.puntos_orbita, = self.ax.plot(ox, oy, color='black', marker='o', markersize=2, lw=0.5, alpha=0.7)
        self.fig.canvas.draw_idle()

    def actualizar(self):
        if not self.en_exp: return
        self.ax.clear(); self.puntos_orbita = None
        c_lin = self.colores_lin[self.idx_col % 6]
        
        # ACTUALIZAR AYUDA (En lugar de crear texto nuevo)
        h_text = "CONTROLES:\n\nESC: Menú\nC: Color\nR: Reset\nP: Guardar"
        if self.tipo < 3: h_text += "\nO: Órbita"
        self.txt_help.set_text(h_text) # <--- CAMBIO AQUÍ

        # El texto de info matemática lo actualiza on_mouse, 
        # así que aquí no hace falta crear nada nuevo.

        # INFO MATEMÁTICA (Banda Derecha)
        '''
        self.txt_info = self.fig.text(0.98, 0.25, "", color='white', fontsize=10, 
                                      family='monospace', ha='right', va='bottom',
                                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        '''

        if self.tipo == 3:
            # Aseguramos que it_sierp sea al menos 1 para que se vea el cuadrado inicial
            nivel = max(1, int(self.it_lin)) 
            img = calcular_pixeles(600, 600, self.it_pix, self.x_min, self.x_max, 
                                   self.y_min, self.y_max, 3, nivel, 
                                   0.0, 0.0, False)
            
            img_modificado = (img * self.val_color + self.val_hue) % 1.0
            self.ax.imshow(img_modificado, extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                           cmap=self.paletas[self.idx_col % 12], origin='lower', aspect='equal') # <-- Usá 'equal'





        elif self.tipo == 4:
            lx, ly = [], []
            p1, p2, p3 = np.array([0,0]), np.array([1,0]), np.array([0.5, 0.866])
            gen_koch(p1, p2, self.it_lin, lx, ly); gen_koch(p2, p3, self.it_lin, lx, ly); gen_koch(p3, p1, self.it_lin, lx, ly)
            lx.append(0.0); ly.append(0.0)
            self.ax.fill(lx, ly, color=c_lin, alpha=0.3); self.ax.plot(lx, ly, color=c_lin, lw=1.5)
        elif self.tipo == 5:
            lx, ly = [], []; gen_dragon(np.array([0,0]), np.array([1,0]), self.it_lin, 1, lx, ly); self.ax.plot(lx, ly, color=c_lin, lw=1.5)
        elif self.tipo == 6:
            pts = []; gen_sierpinski(np.array([0,0]), np.array([1,0]), np.array([0.5,0.866]), self.it_lin, pts)
            for tx, ty in pts: self.ax.fill(tx, ty, color=c_lin, alpha=0.3); self.ax.plot(tx, ty, color=c_lin, lw=1.5)
        elif self.tipo == 7:
            pts = []; gen_arbol(np.array([0.4, 0]), np.array([0.6, 0]), self.it_lin, pts, self.angulo)
            for tx, ty in pts: self.ax.fill(tx, ty, color=c_lin, alpha=0.3); self.ax.plot(tx, ty, color=c_lin, lw=1.2)
        elif self.tipo == 8:
            pts = []; gen_pentagonos(np.array([0.5,0.5]), 0.4, np.pi/2, self.it_lin, pts)
            for tx, ty in pts: self.ax.fill(tx, ty, color=c_lin, alpha=0.4); self.ax.plot(tx, ty, color=c_lin, lw=1.5)
        elif self.tipo == 9:
            caras = []; it_m = min(self.it_lin, 3)
            gen_cubo_iso(-0.5, -0.5, -0.5, 1.0, it_m, caras)
            for pts, cara in caras:
                if cara == "top": fcol = c_lin
                elif cara == "left": fcol = plt.cm.colors.to_hex(np.array(plt.cm.colors.to_rgb(c_lin))*0.8)
                else: fcol = plt.cm.colors.to_hex(np.array(plt.cm.colors.to_rgb(c_lin))*0.6)
                poly = np.array(pts)
                self.ax.fill(poly[:,0], poly[:,1], facecolor=fcol, edgecolor='black', lw=0.3)
                

        else:
            jx, jy = self.julia_c.real, self.julia_c.imag
            img = calcular_pixeles(600, 600, self.it_pix, self.x_min, self.x_max, 
                                   self.y_min, self.y_max, self.tipo, self.it_lin, 
                                   jx, jy, self.modo_smooth)
            
            img_modificado = (img * self.val_color + self.val_hue) % 1.0
            
            # Dibujamos usando los límites que calculó el zoom
            self.ax.imshow(img_modificado, extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                           cmap=self.paletas[self.idx_col % 12], origin='lower', aspect='auto')

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
        self.ax.set_aspect('auto') 
        self.ax.axis('off')
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


    def tecla(self, e):
        if e.key == 'c':
            self.idx_col += 1
            if self.en_exp: self.actualizar()
            else: self.menu_principal()
        elif e.key == 'r' and self.en_exp:
            self.x_min, self.x_max, self.y_min, self.y_max = self.x_orig, self.x_max_orig, self.y_orig, self.y_max_orig
            self.actualizar()
        elif e.key == 'p' and self.en_exp:
            ancho_px, alto_px = 7777, 7777
            max_it_render = 777          # Mucho más detalle en los bordes
            
            print(f"Iniciando renderizado Ultra-HD ({ancho_px}x{alto_px})...")
            print("Esto puede tardar unos segundos, usaremos todos los núcleos de tu PC.")
            
            # 1. Calculamos la matriz de píxeles a máxima calidad
            jx, jy = self.julia_c.real, self.julia_c.imag
            img_hq = calcular_pixeles(ancho_px, alto_px, max_it_render, 
                                      self.x_min, self.x_max, self.y_min, self.y_max, 
                                      self.tipo, self.it_lin, jx, jy, self.modo_smooth)
            
            # 2. Aplicamos tus ajustes de color
            img_hq = (img_hq * self.val_color + self.val_hue) % 1.0
            
            # 3. Creamos una figura especial solo para el guardado (sin ejes ni botones)
            fig_save = plt.figure(figsize=(ancho_px/100, alto_px/100), dpi=100, facecolor='black')
            ax_save = fig_save.add_axes([0, 0, 1, 1]) # Ocupa el 100% de la imagen
            ax_save.imshow(img_hq, cmap=self.paletas[self.idx_col % 12], 
                           origin='lower', aspect='auto')
            ax_save.axis('off')
            
            # 4. Guardado final
            nombre_archivo = f"Render_UltraHD_Fractal_{self.tipo}.png"
            fig_save.savefig(nombre_archivo, facecolor='black', edgecolor='none', pad_inches=0)
            plt.close(fig_save) # Liberamos memoria
            
            print(f"¡TERMINADO! Tu fractal HD está listo: {nombre_archivo}")

        elif e.key == 'escape': 
            self.menu_principal()



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
            
            ax_min.imshow(img, cmap=self.paletas[self.idx_col % 12], origin='lower', aspect='auto')
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

    def actualizar_desde_hub(self):
        conf_actual = (self.x_min, self.x_max, self.y_min, self.y_max, self.tipo, self.nom_f)
        self.abrir(self.tipo, conf_actual)



if __name__ == "__main__":
    AppFractales()
