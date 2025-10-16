import random
import math

from city import City
from ant import Ant

class ACOEngine:
    def __init__(self, num_cities, num_ants, alpha, beta, rho, q):
        self.num_cities = num_cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        self.cities = []
        self.ant = []
        self.distance_matrix = []
        self.pheromone_matrix = []
        
        self.best_tour = []
        self.best_tour_length = float('inf')
        
        self.best_tour = []
        self.best_tour_length = float('inf')

        # --- THEM DONG MOI ---
        # Co hieu de bao cho main.py biet khi co lo trinh moi tot hon
        self.new_best_tour_found = False 

        self.current_ant_index = 0
        self.generation = 0
        self.last_ant_tour = []
        
        self.current_ant_index = 0
        self.generation = 0
        self.last_ant_tour = []
        
    def initialize(self, screen_width, screen_height):
        for _ in range(self.num_cities):
            x = random.randint(20,screen_width - 20)
            y = random.randint(20,screen_height - 20)
            self.cities.append(City(x,y))
    
        self.distance_matrix =  [[0.0 for _ in range(self.num_cities)] for _ in range(self.num_cities)]
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                dist = math.sqrt((self.cities[i].x - self.cities[j].x)**2 + (self.cities[i].y - self.cities[j].y)**2) 
                self.distance_matrix[i][j] = dist

        self.pheromone_matrix = [[1.0 for _ in range(self.num_cities)] for _  in range(self.num_cities)]
        
        self.ants = [Ant() for _ in range(self.num_ants)]
        
    def run_interation(self):
        
        self._let_ants_build_tours()
        self._update_pheromones()
        
    def _let_ants_build_tours(self):
        for ant in self.ants:
            ant.clear()
            visited = [False] * self.num_cities
            start_city = random.randint(0, self.num_cities - 1)
            ant.tour.append(start_city)
            visited[start_city] = True
            
            while len(ant.tour) <self.num_cities:
                current_city = ant.tour[-1]
                probabilities = []
                possible_next_cities = []
                prob_sum = 0.0
                
                for next_city in range(self.num_cities):
                    if not visited[next_city]:
                        pheromone = self.pheromone_matrix[current_city][next_city]**self.alpha
                        heuristic = (1.0/(self.distance_matrix[current_city][next_city]+1e-10))**self.beta 
                        prob_sum += pheromone * heuristic
                        
                for next_city in range(self.num_cities):
                    if not visited[next_city]:
                        pheromone = self.pheromone_matrix[current_city][next_city]**self.alpha
                        heuristic = (1.0/(self.distance_matrix[current_city][next_city]+1e-10))**self.beta
                        probability = (pheromone*heuristic)/prob_sum
                        probabilities.append(probability)
                        possible_next_cities.append(next_city)
                        
                if possible_next_cities:
                    chosen_city = random.choices(possible_next_cities,weights=probabilities, k =1 )[0]
                    ant.tour.append(chosen_city)
                    visited[chosen_city] = True
                    
            ant.tour_length = 0
            for i in range(len(ant.tour)):
                start_node = ant.tour[i]
                end_node = ant.tour[(i+1)%self.num_cities]
                ant.tour_length += self.distance_matrix[start_node][end_node]
                
            if ant.tour_length < self.best_tour_length:
                self.best_tour_length = ant.tour_length
                self.best_tour = ant.tour[:]
                self.new_best_tour_found = True
                
    def update(self):
        
        # Kiem tra xem con kien hien tai da di het dan chua
        if self.current_ant_index < self.num_ants:
            # Neu chua, hay de con kien hien tai xay dung lo trinh cua no
            self._let_one_ant_build_tour(self.current_ant_index)
            # Sau khi no di xong, chuan bi cho con kien tiep theo o khung hinh ke tiep
            self.current_ant_index += 1
        else:
            # Neu tat ca cac con kien da di xong trong the he nay
            # Hay cap nhat pheromone tren toan ban do
            self._update_pheromones()
            # Va reset lai de bat dau mot the he moi
            self.current_ant_index = 0
            self.generation += 1
            # Xoa lo trinh cua con kien cuoi cung de khong bi ve thua
            self.last_ant_tour = []
    
    
    def _let_one_ant_build_tour(self, ant_index):
        # Lay ra con kien can di chuyen tu danh sach
        ant = self.ants[ant_index]
        ant.clear()
        visited = [False] * self.num_cities
        
        start_city = random.randint(0, self.num_cities - 1)
        ant.tour.append(start_city)
        visited[start_city] = True

        while len(ant.tour) < self.num_cities:
            current_city = ant.tour[-1]
            probabilities = []
            possible_next_cities = []
            prob_sum = 0.0

            for next_city in range(self.num_cities):
                if not visited[next_city]:
                    pheromone = self.pheromone_matrix[current_city][next_city] ** self.alpha
                    heuristic = (1.0 / (self.distance_matrix[current_city][next_city] + 1e-10)) ** self.beta
                    prob_sum += pheromone * heuristic

            for next_city in range(self.num_cities):
                if not visited[next_city]:
                    pheromone = self.pheromone_matrix[current_city][next_city] ** self.alpha
                    heuristic = (1.0 / (self.distance_matrix[current_city][next_city] + 1e-10)) ** self.beta
                    if prob_sum > 0:
                        probability = (pheromone * heuristic) / prob_sum
                    else: # Tranh chia cho 0 neu khong co duong di
                        probability = 0
                    probabilities.append(probability)
                    possible_next_cities.append(next_city)
            
            if possible_next_cities:
                chosen_city = random.choices(possible_next_cities, weights=probabilities, k=1)[0]
                ant.tour.append(chosen_city)
                visited[chosen_city] = True
        
        ant.tour_length = 0
        for i in range(len(ant.tour)):
            start_node = ant.tour[i]
            end_node = ant.tour[(i + 1) % self.num_cities] 
            ant.tour_length += self.distance_matrix[start_node][end_node]
        
        # Luu lai lo trinh cua con kien vua di xong
        self.last_ant_tour = ant.tour[:]
        
        if ant.tour_length < self.best_tour_length:
            self.best_tour_length = ant.tour_length
            self.best_tour = ant.tour[:]    

    def _update_pheromones(self):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                self.pheromone_matrix[i][j] *= (1.0 - self.rho)
                
        for ant in self.ants:
            pheromone_to_add = self.q/ant.tour_length
            
            for i in range(len(ant.tour)):
                start_node = ant.tour[i]
                end_node = ant.tour[(i+1)%self.num_cities]
                
                self.pheromone_matrix[start_node][end_node] += pheromone_to_add
                self.pheromone_matrix[end_node][start_node] += pheromone_to_add