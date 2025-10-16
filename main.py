import pygame
from aco_engine import ACOEngine

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

COLOR_BACKGROUND = (20,20,40)
COLOR_CITY = (255,100,100)
COLOR_BEST_TOUR = (255,204,0)
COLOR_PHEROMONE = (200,200,255)
COLOR_TEXT = (255,255,255)
COLOR_ANT_TOUR = (255, 255, 255) # Mau trang de ve duong di cua kien hien tai
COLOR_TEXT = (255, 255, 255)

NUM_CITIES = 20
NUM_ANTS = 30
ALPHA = 1.0
BETA = 3.0
RHO = 0.5
Q = 100.0

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ant Colony")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial",18)
    engine = ACOEngine(NUM_CITIES,NUM_ANTS,ALPHA,BETA,RHO,Q)
    engine.initialize(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    best_tour_alpha = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type  == pygame.QUIT:
                running = False
        engine.update()
        # --- LOGIC MOI: Dieu khien hieu ung fade-in ---
        # Kiem tra co hieu tu "bo nao"
        if engine.new_best_tour_found:
            # Neu co lo trinh moi, reset do mo ve mot gia tri nho (de bat dau hien mo)
            best_tour_alpha = 50 
            # Tat co hieu di de khong reset lien tuc
            engine.new_best_tour_found = False
        
        # Neu do mo chua dat toi da, hay tang dan len theo thoi gian
        if best_tour_alpha < 255:
            # Toc do fade-in, ban co the dieu chinh so nay
            best_tour_alpha += 5 
            # Dam bao khong vuot qua 255
            if best_tour_alpha > 255:
                best_tour_alpha = 255
        screen.fill(COLOR_BACKGROUND)
        
        for i in range(engine.num_cities):
            for j in range(i+1,engine.num_cities):
                p1 = (engine.cities[i].x, engine.cities[i].y)
                p2 = (engine.cities[i].x, engine.cities[i].y)
                
                pheromone_value = engine.pheromone_matrix[i][j]
                alpha_value = min(255, int(pheromone_value *50))
                
                line_surface = screen.convert_alpha()
                line_surface.fill((0,0,0,0)) # Trong suot
                try: # Them try-except de tranh loi khi alpha_value qua lon
                    pygame.draw.line(line_surface, (*COLOR_PHEROMONE, alpha_value), p1, p2, 1)
                    screen.blit(line_surface, (0,0))
                except:
                    pass
        
        if engine.last_ant_tour:
            points = [ (engine.cities[i].x, engine.cities[i].y) for i in engine.last_ant_tour ]
            # Ve duong di cua kien hien tai bang mau trang, net dut
            pygame.draw.lines(screen, COLOR_ANT_TOUR, True, points, 1)
        
        if engine.best_tour:
            points = [ (engine.cities[i].x, engine.cities[i].y) for i in engine.best_tour ]
            
            # Tao mot be mat rieng biet co ho tro do mo
            tour_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            
            # Ve duong thang len be mat do voi do mo hien tai
            pygame.draw.lines(tour_surface, (*COLOR_BEST_TOUR, best_tour_alpha), True, points, 5) # Tang do day len 3
            
            # Dan be mat do len man hinh chinh
            screen.blit(tour_surface, (0,0))

        # Ve cac thanh pho
        for city in engine.cities:
            pygame.draw.circle(screen, COLOR_CITY, (city.x, city.y), 7)
        
        # Ve thong tin
        info_text = f"Thế hệ: {engine.generation} | Kiến số: {engine.current_ant_index}/{engine.num_ants} | Độ dài tốt nhất: {engine.best_tour_length:.2f}"
        text_surface = font.render(info_text, True, COLOR_TEXT)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()