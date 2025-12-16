import numpy as np 
import math
import plotly.graph_objects as go
from simpleai.search import SearchProblem, astar,depth_first,breadth_first,uniform_cost,greedy
import time

# Parámetros globales
SCALE_FACTOR = 10.0177  # Metros por píxel
MAX_HEIGHT_DIFF = 0.75  # Diferencia máxima de altura permitida

# Funciones de conversión de coordenadas
def to_pixel_coords(x, y, num_rows):
    r = num_rows - round(y / SCALE_FACTOR)
    c = round(x / SCALE_FACTOR)
    return r, c

def to_real_coords(r, c, num_rows):
    x = c * SCALE_FACTOR
    y = (num_rows - r) * SCALE_FACTOR
    return x, y

class MarsPathfinder(SearchProblem):
    def __init__(self, scale, max_height, mars_map, start, goal):
        super().__init__(start)
        self.scale = scale
        self.max_height = max_height
        self.mars_map = mars_map
        self.goal = goal
        self.num_rows, self.num_cols = mars_map.shape

    def actions(self, state):
        act = []
        if state is None:
            return act
        r, c = state
        possible_moves = [
            (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)
        ]
        for nr, nc in possible_moves:
            if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols:
                if self.mars_map[nr][nc] != -1:
                    if abs(self.mars_map[r][c] - self.mars_map[nr][nc]) < self.max_height:
                        act.append((nr, nc))
        return act
    
    def result(self, state, action):
        return action if action else state
    
    def is_goal(self, state):
        return state == self.goal
    
    def heuristic(self, state):
        if state is None:
            return float('inf')
        r, c = state
        gr, gc = self.goal
        return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)
    
    def visualize_path(self, result):
        if result is None or result.path() is None:
            print("No se encontró una ruta.")
            return
        
        path = [state for state, action in result.path() if state is not None and isinstance(state, tuple)]
        
        if not path:
            print("Error: el camino calculado no contiene datos válidos.")
            return
        
        path_x, path_y, path_z = [], [], []
        for r, c in path:
            real_x, real_y = to_real_coords(r, c, self.num_rows)
            path_x.append(real_x)
            path_y.append(real_y)
            path_z.append(self.mars_map[r, c])
        
        x = SCALE_FACTOR * np.arange(self.mars_map.shape[1])
        y = SCALE_FACTOR * np.arange(self.mars_map.shape[0])
        X, Y = np.meshgrid(x, y)
        
        fig = go.Figure(data=[
            go.Surface(x=X, y=Y, z=np.flipud(self.mars_map), colorscale='hot', cmin=0,
                       lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
                       lightposition=dict(x=0, y=self.mars_map.shape[0] / 2, z=2 * self.mars_map.max())),
            go.Scatter3d(x=path_x, y=path_y, z=path_z, name='Ruta A*', mode='markers+lines',
                         marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Viridis", size=4))
        ])
        
        fig.update_layout(
            title="Ruta Óptima en Superficie de Marte",
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=self.mars_map.shape[0] / self.mars_map.shape[1],
                                 z=max(self.mars_map.max() / x.max(), 0.2)),
                zaxis=dict(range=[0, self.mars_map.max()])
            )
        )
        fig.show()

if __name__ == "__main__":

    # Cargar el mapa de Marte
    mars_map = np.load("map2.npy")
    num_rows, num_cols = mars_map.shape
    
    # Coordenadas de inicio y destino en el mundo real
    opciones = {
        "1": {"start": (5000, 7600), "goal": (3600, 8600)},
        "2": {"start": (1400, 3600), "goal": (1900, 5800)},
        "3": {"start": (6050, 1750), "goal": (1750, 2500)},
        "4": {"start": (2500, 6000), "goal": (2200, 9000)},
        "5": {"start": (4300, 4700), "goal": (730, 9200)},
        "6": {"start": (2300, 1370), "goal": (4000, 11500)},
        "7": {"start": (3500, 3800), "goal": (3700, 6800)},
        "8": {"start": (2800, 750), "goal": (1900, 5500)},
        "9": {"start": (2400, 2800), "goal": (3500, 1400)},
        "10": {"start": (2200, 7300), "goal": (1500, 7300)},
        "11": {"start": (1050, 1000), "goal": (3420, 2000)}
    }

    ruta = "8"  

    coords = opciones.get(ruta)
    start_real, goal_real = coords["start"], coords["goal"]
    
    '''    start_real = (5000, 7600)
    goal_real = (3600, 8600)'''
    
    # Convertir coordenadas a índices de matriz
    start_state = to_pixel_coords(*start_real, num_rows)
    goal_state = to_pixel_coords(*goal_real, num_rows)
    
    print("Inicio:", start_state)
    print("Destino:", goal_state)
    
    # Planificador de ruta
    pathfinder = MarsPathfinder(SCALE_FACTOR, MAX_HEIGHT_DIFF, mars_map, start_state, goal_state)
    
    start_time=time.time()
    
    # Algoritmos
    alg = 1 
    if alg == 1:
        result = astar(pathfinder, graph_search=True)
    if alg == 2:
        result = depth_first(pathfinder, graph_search=True)
    if alg == 3:
        result = breadth_first(pathfinder, graph_search=True)
    if alg == 4:
        result = uniform_cost(pathfinder, graph_search=True)
    if alg == 5:
        result = greedy(pathfinder, graph_search=True)
 
    end_time=time.time()
 
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")

# Mostrar resultados
if result is not None and result.path():
    print("Distancia recorrida:", SCALE_FACTOR * result.cost)
    pathfinder.visualize_path(result)
else:
    print("No se encontró un camino.")
    
    # Mostrar resultados
    if result is not None and result.path():
        print("Distancia recorrida:", SCALE_FACTOR * result.cost)
        pathfinder.visualize_path(result)
    else:
        print("No se encontró un camino.")