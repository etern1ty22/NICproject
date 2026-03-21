import numpy as np
from typing import List, Tuple, Dict

class GreedyVRPTW:
    """
    Greedy solver for the Vehicle Routing Problem with Time Windows (VRPTW).
    Implementation of the 'Greedy baseline' task using a nearest-neighbor heuristic.
    """

    def __init__(self, 
                 capacity: int, 
                 time_windows: List[Tuple[float, float]], 
                 demands: List[float], 
                 service_times: List[float], 
                 dist_matrix: np.ndarray):
        """
        Initializes the solver with project-specific data.
        :param capacity: Maximum vehicle capacity.
        :param time_windows: List of [start, end] windows for each node (0 is depot).
        :param demands: List of demands for each node.
        :param service_times: Service duration at each node.
        :param dist_matrix: Distance/Travel time matrix.
        """
        self.capacity = capacity
        self.tw = np.array(time_windows)
        self.demands = np.array(demands)
        self.service_times = np.array(service_times)
        self.dist = dist_matrix
        
        # Depot constraints (index 0) [cite: 72]
        self.depot_start = self.tw[0][0]
        self.depot_end = self.tw[0][1]
        self.n_customers = len(demands) - 1

    def _check_feasibility(self, current_node: int, next_node: int, current_time: float, current_load: int) -> Tuple[bool, float]:
        """
        Internal check for capacity and time window constraints[cite: 77, 78].
        Ensures the vehicle can visit the next node and return to the depot.
        """
        # 1. Capacity check
        if current_load + self.demands[next_node] > self.capacity:
            return False, 0.0

        # 2. Time window check for the next customer
        arrival_time = current_time + self.dist[current_node][next_node]
        if arrival_time > self.tw[next_node][1]:
            return False, 0.0

        # 3. Departure time calculation (includes waiting time and service time)
        start_service = max(arrival_time, self.tw[next_node][0])
        departure_time = start_service + self.service_times[next_node]

        # 4. Mandatory check: Can we return to the depot before it closes?
        if departure_time + self.dist[next_node][0] > self.depot_end:
            return False, 0.0

        return True, departure_time

    def solve(self) -> List[List[int]]:
        """
        Main construction loop using nearest feasible neighbor logic.
        Returns a list of routes, where each route starts and ends at the depot (0).
        """
        unvisited = set(range(1, self.n_customers + 1))
        routes = []

        while unvisited:
            route = [0]  # Start at the depot [cite: 72]
            current_load = 0
            current_time = self.depot_start
            found_any = False

            while True:
                last = route[-1]
                best_cust = None
                best_dist = float('inf')
                best_departure = 0.0

                for cust in unvisited:
                    # Feasibility check for the candidate customer
                    is_ok, departure_time = self._check_feasibility(
                        last, cust, current_time, current_load
                    )
                    
                    if is_ok and self.dist[last][cust] < best_dist:
                        best_dist = self.dist[last][cust]
                        best_cust = cust
                        best_departure = departure_time

                if best_cust is not None:
                    # Add customer to the current route
                    unvisited.remove(best_cust)
                    route.append(best_cust)
                    current_load += self.demands[best_cust]
                    current_time = best_departure
                    found_any = True
                else:
                    # Handle unsolvable cases (e.g., tight time windows)
                    if not found_any:
                        raise RuntimeError(
                            f"Cannot serve remaining customers: {unvisited}. "
                            "No feasible neighbors found for a new route."
                        )
                    
                    route.append(0)  # Return to depot
                    routes.append(route)
                    break

        return routes

    def evaluate(self, routes: List[List[int]]) -> Dict:
        """
        Evaluation harness: calculates metrics and validates total feasibility.
        Checks capacity, time windows, and if all customers are served.
        """
        total_dist = 0.0
        is_feasible = True
        visited_customers = []
        
        for route in routes:
            route_load = 0
            current_time = self.depot_start
            
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                dist_leg = self.dist[u][v]
                total_dist += dist_leg
                arrival_time = current_time + dist_leg
                
                if v != 0:  # Customer node
                    visited_customers.append(v)
                    route_load += self.demands[v]
                    
                    # Validate constraints for reporting
                    if route_load > self.capacity or arrival_time > self.tw[v][1]:
                        is_feasible = False
                    
                    current_time = max(arrival_time, self.tw[v][0]) + self.service_times[v]
                else:  # Return to depot
                    if arrival_time > self.depot_end:
                        is_feasible = False
                    
        # Verify all customers are included exactly once
        all_served = (len(set(visited_customers)) == self.n_customers and 
                      len(visited_customers) == self.n_customers)

        return {
            "total_distance": round(total_dist, 2),
            "num_vehicles": len(routes),
            "is_feasible": is_feasible and all_served,
            "all_served": all_served
        }
