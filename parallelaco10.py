import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re

def convert_to_meters(distance_str):
    if 'km' in distance_str:
        distance_km = float(distance_str.replace(' km', ''))  # Remove ' km' suffix and convert to float
        return distance_km * 1000  # Convert kilometers to meters
    elif 'm' in distance_str:
        return float(distance_str.replace(' m', ''))  # Remove ' m' suffix and convert to float
    else:
        raise ValueError("Invalid distance unit. Only 'km' and 'm' units are supported.")

def create_graph_from_files(folder_path):
    # Initialize an undirected graph
    G = nx.Graph()

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)
            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                origin = row['origin_cluster']
                destination = row['destination_cluster']
                distance = row['distance']
                # Skip adding edge if origin and destination are the same
                if origin == destination:
                    continue
                # Convert distance to meters if in kilometers
                distance_in_meters = convert_to_meters(distance)
                # Add nodes to the graph if they don't exist
                if not G.has_node(origin):
                    G.add_node(origin)
                if not G.has_node(destination):
                    G.add_node(destination)
                # Add edge between origin and destination with distance as weight
                G.add_edge(origin, destination, weight=distance_in_meters)
    return G

# Specify the folder path containing the CSV files
folder_path = 'Cluster_edge_data_20'

# Create the graph from files in the folder
G_20 = create_graph_from_files(folder_path)

# Print basic information about the graph
print("Number of nodes:", G_20.number_of_nodes())
print("Number of edges:", G_20.number_of_edges())

# Specify the folder path containing the CSV files
folder_path = 'Cluster_data_20'

def add_severity_to_cluster(folder_path):
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            matches = re.findall(r'\d+', filename)

            # Convert the matches to integers
            number = [int(match) for match in matches][0]
            
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filepath)
            column_name = 'severity'

            # Sum the values in the specified column
            total_sum = df[column_name].sum()
            print("number",number,"severity",total_sum)
            
            G_20.nodes[number]['severity'] = total_sum
            
add_severity_to_cluster(folder_path)

import networkx as nx
import random
import csv
import pandas as pd

class Agent:
    def __init__(self, agent_id, start_node, graph):
        self.agent_id = agent_id
        self.current_node = start_node # node id of current node
        self.graph = graph
        self.travel_time = 0 # Distance left to travel in meters
        self.path=[]
        # Define other attributes as needed (e.g., travel time, pheromone memory)

    def select_next_node(self,travel_speed,graph,nodes,edges,alpha,beta,visited_nodes,police_station_nodes):
        """
        Parameters:
            - travel_speed: float:
                The travelling speed of agents in meters/min
            - graph: NetworkX graph
                The graph representing the environment where agents will navigate.
            - nodes: dictionary of node objects
                dictionary of node objects {node_id : node_id}
            - edges: dictionary of edge objects
                dictionary of edge objects {edge_id : edge_id}
        """
        if self.travel_time > 0:
            # Simulate travelling to next node if still travelling
            self.travel(travel_speed=travel_speed)
            
        # Implement ACO algorithm to select the next node based on pheromone levels and heuristic information
        # Return the selected node
        else:
            # Indicate that current_node is visited as the agent has arrived
            if len(visited_nodes) == 20:
                nodes[self.current_node].visit(track=True)
            else:
                nodes[self.current_node].visit()
            # Select Next Node
            probabilities = []
            for edge in edges:#graph.edges(self.current_node):
                if self.current_node in edge:
                    
                    intersection = set(edge) & set(police_station_nodes) 
                    if self.current_node not in police_station_nodes:
                        if intersection:
                            break
                    for node_id in edge:
                        if node_id != self.current_node:
                            potential_next_node = node_id

                    pheromone_level = edges[edge].pheromone
                    heuristic_info = nodes[potential_next_node].idle_time
                    probability = pheromone_level ** alpha * heuristic_info ** beta
                    probabilities.append((potential_next_node,probability))
                
            total_probability = sum(prob for _, prob in probabilities)
            selection = random.uniform(0, total_probability)

            # Extract neighbors and probabilities into separate lists
            neighbors, probs = zip(*probabilities)

            # Choose a neighbor based on probabilities
            selected_neighbor = random.choices(neighbors, weights=probs, k=1)[0]

            # Update current_node & travel_time
            for edge in edges:
                if set(edge) == set((self.current_node,selected_neighbor)):
                    self.travel_time = edges[edge].distance
                    self.path.append(edge)

            self.current_node = selected_neighbor
            
            
            return self.current_node
        
    def travel(self,travel_speed):
        # Simulate travelling to next node
        new_travel_time = self.travel_time-travel_speed
        if new_travel_time < 0:
            self.travel_time = 0
        else:
            self.travel_time = new_travel_time

    # Define other methods for agent behaviors (e.g., move_to, update_pheromone_memory)

    def reset(self,police_station_nodes):
        self.current_node = random.choice(police_station_nodes)
        self.travel_time = 0 
        self.path=[]
        
    
class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.idle_time = 1  # Current idle time in minutes
        self.num_visits = 0  # Number of times node has been visited
        self.all_idle_time = []  # All idle time across all visits

    def update_idle_time(self):
        self.idle_time += 1
        
    def visit(self,track=False):
        # Updates all the attributes that needs to be updated in a visit from an agent.
        if track:
            self.update_all_idle_time()
            self.update_num_visits()
        self.reset_idle_time()
        
    def update_num_visits(self):
        self.num_visits += 1
        
    def update_all_idle_time(self):
        self.all_idle_time.append(self.idle_time)
        
    def reset_idle_time(self):
        self.idle_time = 1

    def get_average_idle_time(self):
        if self.num_visits == 0:
            return 0
        return sum(self.all_idle_time) / self.num_visits

    def reset(self):
        self.idle_time = 1
        self.num_visits = 0
        self.all_idle_time = []

        
class Edge:
    def __init__(self, edge_id, tau_max, distance):
        self.edge_id = edge_id
        self.pheromone = tau_max
        self.distance = distance
        
    def update_pheromone(self, pheromone, tau_max):
        new_pheromone = pheromone + self.pheromone
        if new_pheromone > tau_max:
            self.pheromone = tau_max
        else:
            self.pheromone = new_pheromone
            
    def decay(self, decay_rate, tau_min):
        new_pheromone = self.pheromone * (1 - decay_rate)
        if new_pheromone < tau_min:
            self.pheromone = tau_min
        else:
            self.pheromone = new_pheromone
        
    
class Environment:
    def __init__(self, graph, num_agents, num_ants, max_iterations, evaporation_rate, alpha, beta, Q, agent_max_iterations, agent_min_iteration, tau_min, tau_max, travel_speed, police_station_nodes, convergence_threshold):
        """
        Parameters:
            - graph: NetworkX graph
                The graph representing the environment where agents will navigate.
            - num_agents: int
                The number of agents in the environment.
            - num_ants: int
                The number of ants in the environment
            - max_iterations: int
                The maximum number of iterations the simulation will run.
            - evaporation_rate: float
                The rate at which pheromones evaporate in the environment.
            - alpha: float
                The weight parameter for the pheromone level in the probability calculation.
            - beta: float
                The weight parameter for the heuristic information in the probability calculation.
            - Q: float
                How much pheromone to deposit in each iteration.
            - agent_max_iterations: int
                The max number of iterations used in the MMAS algorithm for each agent's decision-making.
            - agent_min_iterations
                The min number of iterations used in the MMAS algorithm for each agent's decision-making after visiting each node.
            - tau_min: float
                The minimum pheromone level allowed in the environment.
            - tau_max: float
                The maximum pheromone level allowed in the environment.
            - travel_speed: float
                The speed at which agents travel through the environment in km/h.
        """
        self.graph = graph
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.nodes = {node_id: Node(node_id=node_id) for node_id in self.graph.nodes()}
        self.edges = {(u, v): Edge(edge_id=(u,v),tau_max=self.tau_max, distance=d['weight']) for u, v, d in self.graph.edges(data=True)}
        self.agents = [Agent(agent_id= agent_id,start_node=random.choice(police_station_nodes), graph=graph) for agent_id in range(num_agents)]
        self.num_ants = num_ants
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.agent_max_iterations = agent_max_iterations
        self.travel_speed = travel_speed # Get Travel Speed in meter/min
        self.police_station_nodes = police_station_nodes
        self.convergence_threshold = convergence_threshold
        self.agent_min_iteration = agent_min_iteration
        self.visited_nodes = set()
        self.convergence_counter = 0
        self.agent_convergence_counter = 0
        self.agent_minimum_iteration_counter = 0
        self.save_to_csv = self.setup_save_to_csv()
        
    def setup_save_to_csv(self):
        base = [['Global_Average_Idle_Time']]
        for agent in range(self.num_agents):
            base[0].append(agent)
        return base
        
    def get_node_by_id(self, node_id):
        return self.nodes[node_id]
    
    def get_global_average_idle_time(self):
        """
        Calculate the average idle time across all hotspots
        """
        total_idle_time=0
        total_hotspots=len(self.nodes)-len(self.police_station_nodes) # number of hotspots excluding police station nodes
        for node_id in self.nodes:
            if node_id not in self.police_station_nodes:
                total_idle_time += self.nodes[node_id].get_average_idle_time()
        if total_hotspots == 0:
            return 0
        else:
            return total_idle_time / total_hotspots

    def has_agent_converged(self, current_avg_idle_time, previous_avg_idle_time):
        # Check if the current average idle time is less than the previous
        if current_avg_idle_time < previous_avg_idle_time:
            self.agent_convergence_counter = 0  # Reset the convergence counter
        else:
            # Increment the convergence counter
            self.agent_convergence_counter += 1

        # Check if the convergence counter has reached the threshold
        if self.agent_convergence_counter >= self.convergence_threshold:
            return True
        else:
            return False
        
    def has_converged(self, current_best_result, previous_best_result):
        # Check if the current average idle time is less than the previous
        if current_best_result < previous_best_result:
            self.convergence_counter = 0  # Reset the convergence counter
        else:
            # Increment the convergence counter
            self.convergence_counter += 1

        # Check if the convergence counter has reached the threshold
        if self.convergence_counter >= self.convergence_threshold:
            return True
        else:
            return False

    def run_agent_simulation(self):
        count=0
        for iteration in range(self.agent_max_iterations):
            current_global_avg_idle_time = self.get_global_average_idle_time()
            count+=1
            # Check for convergence
            if len(self.visited_nodes) == 20:
                self.agent_minimum_iteration_counter += 1
                if self.agent_minimum_iteration_counter >= self.agent_min_iteration:
                    if self.has_agent_converged(current_global_avg_idle_time,previous_global_avg_idle_time):
#                         print("Global average idle time has plateaued.")
#                         print("current_global_avg_idle_time: ",current_global_avg_idle_time)
#                         print(count)
                        return (current_global_avg_idle_time,self.get_all_agent_paths())
            # Update previous average idle time for next iteration
            previous_global_avg_idle_time = current_global_avg_idle_time
        
            for agent in self.agents:
                # Select next node & add it to visited nodes
                next_node = (agent.select_next_node(self.travel_speed,self.graph,self.nodes,self.edges,self.alpha,self.beta,self.visited_nodes,self.police_station_nodes))
                if next_node != None:
                    self.visited_nodes.add(next_node)
#                 print("agent ",agent.agent_id,"next: ",next_node)
            
            # Evaporate pheromones, check termination criteria, etc.
            for node in self.graph.nodes():
                self.nodes[node].update_idle_time()
            previous_global_avg_idle_time = current_global_avg_idle_time
        
        return (self.get_global_average_idle_time(),self.get_all_agent_paths())
    
    def get_all_agent_paths(self):
        return {agent.agent_id: agent.path for agent in self.agents}
    
    def run_ant_simulation(self):
        # Placeholder for best_attempt when initialising as no best attempt yet
        best_attempt = (float('inf'),[])
        for ant in range(self.num_ants):
            #print("Ant: ", ant)
            attempt = self.run_agent_simulation()
            self.spread_pheromones(attempt)
            self.decay_pheromones()
            if attempt[0] < best_attempt[0]:
                #print("swapping best attempt",best_attempt[0], "for", attempt[0])
                best_attempt = attempt
            self.reset_agents_nodes_environment()
        return best_attempt
    
    def run_simulation(self):
        best_attempt = (float('inf'),[])
        for iteration in range(self.max_iterations):
            #print("Iteration: ",iteration)
            attempt = self.run_ant_simulation()
            if self.has_converged(attempt[0],best_attempt[0]):
                best_attempt = attempt
                self.save_to_csv.append(self.convert_path_format(best_attempt))
                #self.save_data_to_csv()
                return best_attempt
            if attempt[0] < best_attempt[0]:
                #print("swapping best ant attempt",best_attempt[0], "for", attempt[0])
                best_attempt = attempt
                self.save_to_csv.append(self.convert_path_format(best_attempt))#[best_attempt[0],best_attempt[1]])
        #print("The best attempt is: ",best_attempt[0])
        #self.save_data_to_csv()
        return best_attempt
    
    def convert_path_format(self,best_attempt):
        """
        Convert path format from {agent_id:[path]} to [agentx path, agentx path]
        This is done because 1 csv file cell cannot hold the full path
        """
        base = [best_attempt[0]]
        for agent_id in best_attempt[1]:
            base.append(best_attempt[1][agent_id])
        return base
    
    def spread_pheromones(self,best_attempt):
        for agent in best_attempt[1]:
            for edge in best_attempt[1][agent]:
                self.edges[edge].update_pheromone(self.Q,self.tau_max)
        
    def decay_pheromones(self):
        for edge in self.edges:
            self.edges[edge].decay(self.evaporation_rate,self.tau_min)
    
    def reset_agents_nodes_environment(self):
        for agent in self.agents:
            agent.reset(self.police_station_nodes)
        for node in self.nodes:
            self.nodes[node].reset()
        self.visited_nodes = set()
        self.agent_convergence_counter = 0
        self.agent_minimum_iteration_counter = 0

        
    def save_data_to_csv(self):
        csv_file_path = os.path.join("aco", f'aco_record_agent_{self.num_agents}.csv')
        
        # Create a DataFrame from the data
        df = pd.DataFrame(self.save_to_csv[1:], columns=self.save_to_csv[0])

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)

        print("Data has been written to", csv_file_path)



from itertools import product
import time
import multiprocessing

# Define the function to perform ACO algorithm
def run_ACO(params):
    print("Running ACO with parameters:", params)
    num_ants, evaporation_rate, alpha, beta, Q, tau_min, tau_max, num_runs = params
    avg_scores = []
    for _ in range(num_runs):
        # Run the ACO algorithm with the current combination of parameters
        result = Environment(graph=G_20, num_agents=10, num_ants=num_ants, max_iterations=50,
                         evaporation_rate=evaporation_rate, alpha=alpha, beta=beta, Q=Q,
                         agent_max_iterations=1000, agent_min_iteration=100, tau_min=tau_min,
                         tau_max=tau_max, travel_speed=800, police_station_nodes=['P0', 'P1'],
                         convergence_threshold=5).run_simulation()
        avg_scores.append(result[0])  # Append the average score to the list of scores

    avg_score = sum(avg_scores) / num_runs  # Calculate the average of average scores

    return avg_score, params

if __name__ == "__main__":
    # Define the parameter grid and number of runs
    param_grid = {
        'num_ants': [10, 30, 50, 70, 90],
        'evaporation_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
        'alpha': [1.0, 2.0],
        'beta': [1.0, 2.0],
        'Q': [0.1, 0.5, 1.0, 1.5, 2.0],
        'tau_min': [0.1, 0.5, 1.0],
        'tau_max': [1.0, 5.0, 10.0],
    }
    num_runs = 5  # Number of times to run each combination

    # Initialize the best score and parameters
    best_avg_score = float('inf')
    best_params = None

    # Record start time
    start_time = time.time()

    # Generate parameter combinations
    param_combinations = list(product(*param_grid.values()))

    # Create a multiprocessing Pool
    with multiprocessing.Pool() as pool:
        params_with_runs = [(param + (num_runs,)) for param in param_combinations]
        results = pool.map(run_ACO, params_with_runs)

    # Find the best parameters and score
    for avg_score, params in results:
        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_params = params
    
    keys = ['num_ants', 'evaporation_rate', 'alpha', 'beta', 'Q', 'tau_min', 'tau_max']
    # Output the best_params to a text file
    output_file = os.path.join("aco", f"aco_10agent_best_params.txt")
    with open(output_file, "w") as f:
        f.write("Best Parameters:\n")
        for key, value in zip(keys, best_params):
            f.write(f"{key}: {value}\n")

    print(f"Best Parameters: {best_params}")
    print(f"Best parameters saved to {output_file}")

    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")