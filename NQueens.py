import random
import copy
import numpy as np
from os import system, name 

class Hill_Climbing_Search:
    
    def __init__(self, state = None, max_side = 0, n_queen = 0):
        self.startingState = state
        
        if(state == None and n_queen):
            self.n_queen = n_queen
        elif(state == None and n_queen == 0):
            print("Invalid input value. Setting n_queen equal to 8 by Default")
            self.n_queen = 8
        else:
            self.n_queen = len(state)

        self.total_step_count = 0
        self.max_side = max_side
        self.rem_side = max_side
    
    
    # get_diagonal_right - Returns all the cells which are present diagonally right to the current cell.
    
    def get_diagonal_right(self, rows, columns):
        i = columns+1
        cells = []
        while i < self.n_queen:
            if rows-(i-columns) >= 0: 
                cells.append((rows-(i-columns), i)) 
            if rows+(i-columns) <= self.n_queen-1: 
                cells.append((rows+(i-columns), i))
            i+=1
        return cells
    
    
    # get_horizontal_right - Returns the cells which are present horizontally right to the current cell.

    def get_horizontal_right(self, rows, columns):
        i = columns+1
        cells = []
        while i < self.n_queen:
            cells.append((rows, i))
            i+=1
        return cells

    
    #cells_to_right - Combines the above two methods to get all the cells to the right.
    
    def cells_to_right(self, rows, columns):
        return self.get_horizontal_right(rows,columns) + self.get_diagonal_right(rows,columns)

    
    # get_queens_state - Get cell positions of the queens of the given state

    def get_queens_state(self, state):
        c = []
        for cols, rows in enumerate(state):
            c.append((rows,cols))
        return c


    # calculate_heuristic - Calculate the heuristic value for a given state.
    
    def calculate_heuristic(self, state_cell):
        h = 0
        for row,col in state_cell:
            x = set(state_cell)
            y = set(self.cells_to_right(row,col))
            inter = x.intersection(y)
            h += len(inter)
        return h


    # print_state - Display the N-Queens Problem for the current state as a Matrix.
    
    def print_state(self, state_cell):
        
        print(state_cell)
        for i in range(self.n_queen):
            s = '|'
            for j in range(self.n_queen):
                if((i,j) in state_cell):
                    s += 'Q|'
                else:
                    s += '*|'
            print(s)

    '''
    heuristic_matrix - 
        Calculate heuristic values for all the cells to take the next step.
        Returns [
            'Matrix of Heuristics',
            'Least of Heuristics',
            '2 arrays containing rows and columns of cells containing Least of Heuristics'
        ]
    '''
    def heuristic_matrix(self, state_cell):
        heuristic_m = np.zeros((self.n_queen,self.n_queen), int) + -1
        heuristic_least = sum(range(self.n_queen)) + 1
        heuristic_least_state = None

        for (x,y) in state_cell:
            for i in range(self.n_queen):
                if(x == i):
                    pass
                else:
                    new_state = copy.deepcopy(state_cell)
                    new_state[y] = temp = (i, y)
                    heuristic_m[i,y] = self.calculate_heuristic(new_state)

                    heuristic_least = min(heuristic_least, heuristic_m[i,y])
                    heuristic_least_state = new_state

        return heuristic_m, heuristic_least, np.where(heuristic_m == heuristic_least)

    '''
    steepest_ascent_algorithm - 
        Implementation of the Steepest Ascent Hill Climbing Search. This methods calculates the least heuristic value and moves forward with execution.
        Result:
            1 -> Flat, shoulder or Flat local maxima
            2 -> Local Maxima
            3 -> Success
    '''
    def steepest_ascent_algorithm(self, state = None, h = None, step_size = 0):
        
        state_cell = None
        
        if(step_size == 0):
            state = self.startingState
            state_cell = self.get_queens_state(state)
            h = self.calculate_heuristic(state_cell)
        else:
            state_cell = self.get_queens_state(state)
        
        step_size+=1
        self.total_step_count+=1

        if(h == 0):
            print("Success: ")
            self.print_state(state_cell)
            return 3, step_size
        
        if(step_size == 1):
            print("Initial: ")
            self.print_state(state_cell)
        else:
            print('Step: ', step_size)
            self.print_state(state_cell)
            
        heuristic_m = self.heuristic_matrix(state_cell)
        heuristic_least = heuristic_m[1]
        
        rand = random.randint(0, len(heuristic_m[2][0])-1)
        row = heuristic_m[2][0][rand]
        col = heuristic_m[2][1][rand]

        new_state = copy.deepcopy(state)
        new_state[col] = row

        if(heuristic_least < h):
            return self.steepest_ascent_algorithm(new_state, heuristic_least, step_size)
        elif (heuristic_least > h): 
            print("Search Failed")
            return 2, step_size 
        elif (heuristic_least == h): 
            if(self.rem_side): 
                self.rem_side-=1
                return self.steepest_ascent_algorithm(new_state, heuristic_least, step_size)
            else:
                print("Search Failed")
                return 1, step_size

    # random_state - Generates a new random state each time it is called.
    
    def random_state(self):
        s = []
        for i in range(self.n_queen):
            s.append(random.randint(0,self.n_queen-1))
        return s
        
    
    # random_restart_hill_climbing - This method is used to implement Random Restart Hill Climbing Search by using Steepest Ascent as Base.
        
    def random_restart_hill_climbing(self):
        r = 0
        while True:
            r+=1
            self.startingState = self.random_state()
            output = self.steepest_ascent_algorithm()
            if(output[0] == 3):
                return r, output[1], self.total_step_count
                break
    
class Hill_Climbing_Analysis:
    
    def __init__(self, n_value, max_iterations, max_side = 0):
        self.n_value = n_value
        self.max_iterations = max_iterations
        self.max_side = max_side
        self.steepest_ascent_stats = [[0,[]],[0,[]],[0,[]],[0,[]]]
        self.steepest_ascent_with_side_stats = [[0,[]],[0,[]],[0,[]],[0,[]]]
        self.random_restart_stats = [0, [], [], []]
        self.random_restart_with_side_stats = [0, [], [], []]
    
    
    # do_analysis - Performs Steepest Ascent and Random Restart Hill Climbing max_iterations (Say max_iterations = 100) times.
    
    def do_analysis(self):
        
        if(self.n_value in range(4)):
            print('Please Enter a Value Greater Than 3.')
            return
        
        if(self.max_iterations < 1):
            print('Please Enter a Value Greater Than 1.')
            return

        for n in range(self.max_iterations):
            self.steepest_ascent_stats[0][0]+=1
            self.steepest_ascent_with_side_stats[0][0]+=1
            self.random_restart_stats[0]+=1
            self.random_restart_with_side_stats[0]+=1
            s = []
            
            for i in range(self.n_value):
                s.append(random.randint(0,self.n_value-1))

            print("Hill Climbing Search Analysis")
            hillClimbing = Hill_Climbing_Search(s)
            result = hillClimbing.steepest_ascent_algorithm()
            self.steepest_ascent_stats[result[0]][0]+=1 
            self.steepest_ascent_stats[result[0]][1].append(result[1]) 
            
            print("Hill climbing Search with Sideways Analysis")
            hillClimbing = Hill_Climbing_Search(s, self.max_side)
            result = hillClimbing.steepest_ascent_algorithm()
            self.steepest_ascent_with_side_stats[result[0]][0]+=1 
            self.steepest_ascent_with_side_stats[result[0]][1].append(result[1]) 
            
            print("Random Restart Hill Climbing Search")
            hillClimbing = Hill_Climbing_Search(None, 0, self.n_value)
            result = hillClimbing.random_restart_hill_climbing()
            self.random_restart_stats[1].append(result[0]) 
            self.random_restart_stats[2].append(result[1]) 
            self.random_restart_stats[3].append(result[2]) 

            print("Random Restart Hill Climbing Search with Sideways Analysis")
            hillClimbing = Hill_Climbing_Search(None, self.max_side, self.n_value)
            result = hillClimbing.random_restart_hill_climbing()
            self.random_restart_with_side_stats[1].append(result[0]) 
            self.random_restart_with_side_stats[2].append(result[1]) 
            self.random_restart_with_side_stats[3].append(result[2])
        
        self.print_analysis()
        
   
    # print_analysis - Prints the final analysis of all 4 algorithms
    
    def print_analysis(self):
        self.print_steepest_ascent_stats(self.steepest_ascent_stats, "Hill climbing Search Analysis")
        self.print_steepest_ascent_stats(self.steepest_ascent_with_side_stats, "Hill climbing Search with sideways Analysis")
        self.print_random_restart_stats(self.random_restart_stats, "Random Restart Hill Climbing Search")
        self.print_random_restart_stats(self.random_restart_with_side_stats, "Random Restart Hill Climbing Search with Sideways Analysis")

    
    #print_rand_restart_stat(self) - Display Random Restart Hill Climbing Search Analysis Report.

    def print_random_restart_stats(self, output, head):
        
        total_runs = output[0]
        restart_average = sum(output[1]) / total_runs
        last_steps_average = sum(output[2]) / total_runs
        total_steps_average = sum(output[3]) / total_runs
        
        print("\n\n"+head)
        underline = ''
        for i in range(len(head)): underline+="="
        print(underline)
        print()
        print("N value: ", self.n_value, " (i.e ",self.n_value,"x",self.n_value,")")
        print("Total Runs: ", total_runs)
        print()
        print("Average Restarts: ", restart_average)
        print("Average Steps (last restart): ", last_steps_average)
        print("Average steps (all restarts): ", total_steps_average)
    
    
    # print_steep_climb_stats - Displays the report for steepest ascent algorithm with and without sideways move.

    def print_steepest_ascent_stats(self, output, head):
        
        total_runs = output[0][0]
        
        success = output[3][0]
        
        if success:
            success_rate = round((success/total_runs)*100,2)
            success_steps = output[3][1]
            avg_success_steps = round(sum(success_steps)/success, 2)
        else:
            success_rate = success_steps = avg_success_steps = '-'
        
        failure = output[1][0]+output[2][0]
        
        if failure:
            failure_rate = round((failure/total_runs)*100,2)
            failure_steps = output[1][1]+output[2][1]
            failure_avg_steps = round(sum(failure_steps)/failure,2)
        else:
            failure_rate = failure_steps = failure_avg_steps = '-'
        
        flatRuns = output[1][0]
        
        print("\n\n"+head)
        underline = ''
        for i in range(len(head)): underline+="="
        print(underline)
        print("\nN value: ", self.n_value, " (i.e ",self.n_value,"x",self.n_value,")")
        print("Total Runs: ", total_runs)
        print("\nSuccess, Runs: ", success)
        print("Success, Rate: ", success_rate, "%")
        # print("Success, Steps: ", successSteps)
        print("Success, Average Steps: ", avg_success_steps)
        print("\nFailure, Runs: ", failure)
        print("Failure, Rate: ", failure_rate, "%")
        # print("Failure, Steps: ", failureSteps)
        print("Failure, Average Steps: ", failure_avg_steps)
        print("\n\nFlat local maxima / Shoulder: ", flatRuns)
        return

n_input = 0
input_iterations = 0
input_sideways = 0

#Reading N value of N Queens problem
while(True):
    try:
        n_input = (int)(input("Please enter N value: "))
        if(n_input < 4):
            print("Please enter a number that is above 3! ")
        else:
            break
    except ValueError:
        print("Please enter a number!")

#Reading maximum iterations value
while(True):
    try:
        input_iterations = (int)(input("Please enter iterations value: "))
        if(input_iterations < 1):
            print("Please enter a number that is 1 or above! ")
        else:
            break
    except ValueError:
        print("Please enter a number!")
        
#Reading maximum sideways value
while(True):
    try:
        input_sideways = (int)(input("Please enter a value for the maximum sideways move allowed: "))
        if(input_sideways < 1):
            print("Please enter a number that is 1 or above! ")
        else:
            break
    except ValueError:
        print("Please enter a number!")

if __name__ == "__main__":
    hill_climbing_analysis = Hill_Climbing_Analysis(n_input, input_iterations, input_sideways)
    hill_climbing_analysis.do_analysis()