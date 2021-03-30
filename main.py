import numpy as np
import sympy as sy
from sympy import solve
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt


#Settings
INFINITY = 1000
VIEW_THRESHOLD = 20




#Globals
MAX_X1 = INFINITY
MIN_X1 = -1 * INFINITY
MAX_X2 = INFINITY
MIN_X2 = -1 * INFINITY
MAX_DISTANCE = 2 * INFINITY
x1, x2= sy.symbols('x1 x2')


class Constraint:
    def __init__(self, eq):
        self.equation_string = eq
        self.parse_equation()
        self.create_constraint_polygon()
    

    def normalize_equation(self):
        if self.equation_string.find('<=') == -1 and self.equation_string.find('>=') == -1: #if not found any >= or <= sign
            solution = str(solve(self.equation_string.replace('=', '-'), x2)[0])
            self.equation_string = solution
        else:
            solution = str(solve(self.equation_string, x2))
            infinity_index = solution.find('oo')
            if infinity_index == -1:
                infinity_index = 0

            and_index = solution.find('&')
            if and_index != -1:
                if infinity_index < and_index:
                    solution = solution[and_index + 3 :-1]
                else:
                    solution = solution[1 : and_index - 2]
            self.equation_string = solution


    def parse_equation(self):
        self.normalize_equation()
        self.is_const_x1 = False
        sign = '<='
        sign_index = self.equation_string.find('<=')
        if sign_index == -1:
            sign_index = self.equation_string.find('>=')
            sign = '>='
        if sign_index != -1:
            x2_index = self.equation_string.find('x2')
            if x2_index != -1:
                if x2_index > sign_index:
                    if sign == '<=':
                        sign = '+'
                    else:
                        sign = '-'
                    self.function = sy.sympify(self.equation_string[:sign_index - 1])
                    self.sign = sign
                else:
                    if sign == '<=':
                        sign = '-'
                    else:
                        sign = '+'
                    self.function = sy.sympify(self.equation_string[sign_index + 3:])
                    self.sign = sign
            else:
                self.is_const_x1 = True
                x1_index = self.equation_string.find('x1')
                if x1_index > sign_index:
                    if sign == '<=':
                        sign = '+'
                    else:
                        sign = '-'
                    self.function = sy.sympify(self.equation_string[:sign_index - 1])
                    self.sign = sign
                else:
                    if sign == '<=':
                        sign = '-'
                    else:
                        sign = '+'
                    self.function = sy.sympify(self.equation_string[sign_index + 3:])
                    self.sign = sign
        else:
            self.function = sy.sympify(self.equation_string)
            self.sign = '='


    def eval_function(self):
        if self.is_const_x1:
            self.function_values = np.linspace(MIN_X2, MAX_X2)
            self.function_inputs = [self.function.evalf() for v in self.function_values]
        else:
            self.function_inputs = np.linspace(MIN_X1, MAX_X1)
            self.function_values = sy.lambdify(x1, self.function)(self.function_inputs)
            if type(self.function_values) is not np.ndarray:
                val = self.function_values
                self.function_values = [val for i in self.function_inputs]

    @staticmethod
    def is_feasible_region_exists(constraints):
        feasible_exists = True
        intersection_polygon = constraints[0].constraint_polygon
        for constraint in constraints[1:]:
            intersection_polygon = intersection_polygon.intersection(constraint.constraint_polygon)
            feasible_exists = intersection_polygon.intersects(constraint.constraint_polygon)
        return feasible_exists


    def create_constraint_polygon(self):
        self.eval_function()
        p1 = (self.function_inputs[0], self.function_values[0])
        p2 = (self.function_inputs[-1], self.function_values[-1])
        if not self.is_const_x1:
            if self.sign == '-':
                p3 =  (self.function_inputs[0], self.function_values[0] - MAX_DISTANCE)
                p4 =  (self.function_inputs[-1], self.function_values[-1] - MAX_DISTANCE)
            elif self.sign == '+':
                p3 =  (self.function_inputs[0], self.function_values[0] + MAX_DISTANCE)
                p4 =  (self.function_inputs[-1], self.function_values[-1] + MAX_DISTANCE)
            else:
                p3 = None
                p4 = None
        else:
            if self.sign == '-':
                p3 =  (self.function_inputs[0] - MAX_DISTANCE, self.function_values[0])
                p4 =  (self.function_inputs[-1] - MAX_DISTANCE, self.function_values[-1])
            elif self.sign == '+':
                p3 =  (self.function_inputs[0] + MAX_DISTANCE, self.function_values[0])
                p4 =  (self.function_inputs[-1] + MAX_DISTANCE, self.function_values[-1])
            else: 
                p3 = None
                p4 = None
        if p3 == None and p4 == None:
            self.constraint_polygon = LineString([p1, p2])
        else:
            self.constraint_polygon = Polygon([p1, p2, p4, p3])


    @staticmethod
    def get_feasible_region(constraints):
        intersection_shape = constraints[0].constraint_polygon
        for constraint in constraints[1:]:
            intersection_shape = intersection_shape.intersection(constraint.constraint_polygon)
        if type(intersection_shape) == Polygon:
            shape_type = 'polygon'
            feasible_region_bound_Xs, feasible_region_bound_Ys = intersection_shape.exterior.xy
            feasible_region_bound_Xs.pop()
            feasible_region_bound_Ys.pop()
           
        elif type(intersection_shape) == LineString:
            shape_type = 'line'
            feasible_region_bound_Xs, feasible_region_bound_Ys = intersection_shape.xy
        
        else:
            shape_type = 'point'
            feasible_region_bound_Xs, feasible_region_bound_Ys = intersection_shape.xy
        return (shape_type, feasible_region_bound_Xs, feasible_region_bound_Ys)



class Objective_Function:
    def __init__(self, objective_string):
        self.parse_equation(objective_string)

    def get_isoprofit_line(self, Xs, Ys):
        selected_indexes = []

        for p in Xs:
            if p != MAX_X1 and p != MIN_X1:
                index = Xs.index(p)
                if Ys[index] != MAX_X2 and Ys[index] != MIN_X2:
                    selected_indexes.append(index)
        if len(selected_indexes) != 0:
            x = Xs[selected_indexes[0]]
            y = Ys[selected_indexes[0]]
        else:
            x = 0
            y = 0
        val = self.equation.subs([(x1, x), (x2, y)])
        eq = self.equation - val
        self.isoprofit_line = eq
        if not self.is_const_x1:
            eq = solve(eq, x2)[0]
            self.isoprofit_line = eq
            x1_input = np.linspace(MIN_X1, MAX_X1)
            x2_vals = sy.lambdify(x1, eq)(x1_input)
            if type(x2_vals) is not np.ndarray:
                temp_val = x2_vals
                x2_vals = [temp_val for i in x1_input]
        else:
            x2_vals = np.linspace(MIN_X2, MAX_X2)
            x1_input = [x for v in x2_vals]
        return (x1_input, x2_vals, x, y)
            

    def parse_equation(self, objective_string):
        self.sense = objective_string[:3]
        self.equation = sy.sympify(objective_string[3:])
        x2_index = str(self.equation).find('x2')
        if x2_index != -1:
            self.is_const_x1 = False
        else:
            self.is_const_x1 = True

    
    def get_optimal_solution(self, corner_points_x, corner_points_y):
        optimal_value = self.equation.subs([(x1, corner_points_x[0]), (x2, corner_points_y[0])])
        optimal_point = (corner_points_x[0], corner_points_y[0])
        has_secondary_optimal_point = False
        is_optimal_solution_infinite = False
        other_optimal_point = (corner_points_x[0], corner_points_y[0])
        for i in range(1, len(corner_points_x)):
            temp_val = self.equation.subs([(x1, corner_points_x[i]), (x2, corner_points_y[i])])
            if self.sense == 'max':
                if temp_val > optimal_value:
                    optimal_value = temp_val
                    optimal_point = (corner_points_x[i], corner_points_y[i])
                elif temp_val == optimal_value:
                    has_secondary_optimal_point = True
                    other_optimal_point = (corner_points_x[i], corner_points_y[i])
            else:
                if temp_val < optimal_value:
                    optimal_value = temp_val
                    optimal_point = (corner_points_x[i], corner_points_y[i])
                elif temp_val == optimal_value:
                    has_secondary_optimal_point = True
                    other_optimal_point = (corner_points_x[i], corner_points_y[i])


        if str(self.equation).find('x1') != -1:
            if (optimal_point[0] == MAX_X1 and self.sense == 'max') or (optimal_point[0] == MIN_X1 and self.sense == 'min'):
                is_optimal_solution_infinite = True
        if str(self.equation).find('x2') != -1:
            if (optimal_point[1] == MAX_X2 and self.sense == 'max') or (optimal_point[1] == MIN_X2 and self.sense == 'min'):
                is_optimal_solution_infinite = True
        x1_input = None
        x2_vals = None
        if has_secondary_optimal_point:
            if str(self.equation).find('x1') != -1:
                if (optimal_point[0] == MAX_X1 and self.sense == 'max') or (optimal_point[0] == MIN_X1 and self.sense == 'min'):
                    is_optimal_solution_infinite = True
            if str(self.equation).find('x2') != -1:
                if (optimal_point[1] == MAX_X2 and self.sense == 'max') or (optimal_point[1] == MIN_X2 and self.sense == 'min'):
                    is_optimal_solution_infinite = True
            if not self.is_const_x1:
                x1_input = [optimal_point[0], other_optimal_point[0]]
                x2_vals = [optimal_point[1], other_optimal_point[1]]
            else:
                x2_vals = np.linspace(optimal_point[1], other_optimal_point[1])
                x1_input = [optimal_point[0] for v in x2_vals]
        optimal_line = (x1_input, x2_vals)
        return (has_secondary_optimal_point,is_optimal_solution_infinite, optimal_value, optimal_point, optimal_line)


class Plotter:
    def __init__(self):
        plt.title = '2D LP'
        self.ax = plt.gca()
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['bottom'].set_linewidth(0.25)
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['left'].set_linewidth(0.25)
        self.ax.spines['right'].set_color('none')
        self.ax.axis('equal')
        self.ax.tick_params(axis='both', which='major', labelsize=8)

    def draw_constraint_lines(self, constraints):
        for constraint in constraints:
            plt.plot(constraint.function_inputs, constraint.function_values, 'grey')

    def draw_feasible_region(self, shape_type, feasible_region_bound_Xs, feasible_region_bound_Ys):
        if shape_type == 'polygon':
            plt.fill(feasible_region_bound_Xs, feasible_region_bound_Ys, 'yellow')
        elif shape_type == 'line':
            plt.plot(feasible_region_bound_Xs, feasible_region_bound_Ys,linewidth='3', color='yellow')
        else:
            plt.plot(feasible_region_bound_Xs, feasible_region_bound_Ys,marker='o', markersize=10, color="yellow")

    def draw_isoprofit_line(self, isoprofit_Xs, isoprofit_Ys):
        plt.plot(isoprofit_Xs, isoprofit_Ys,linewidth=2, color='red', linestyle='dashed')

    def draw_optimize_solution(self, has_secondary_optimize_point, optimize_value, optimize_point, optimize_line):
        if not has_secondary_optimize_point:
            plt.plot(optimize_point[0], optimize_point[1], marker='o', markersize=6, color="red")
        else:
            plt.plot(optimize_line[0], optimize_line[1],linewidth=3, color="red")
    
    def show(self, x_view, y_view):
        self.ax.set_xlim([x_view - VIEW_THRESHOLD, x_view + VIEW_THRESHOLD])
        self.ax.set_ylim([y_view - VIEW_THRESHOLD, y_view + VIEW_THRESHOLD])
        plt.show()
        


################################################################################
################################################################################
################################# Driver Code ##################################
################################################################################
################################################################################


print('###########################################################################################')
print('###################################### 2D LP Solver #######################################')
print('###########################################################################################\n\n\n')
print('Name of your variables must be \"x1\" and \"x2\".')
print('Enter Objective function. example: min 2*x1 + x2')
objective_function = Objective_Function(input('Objective Function: '))
no_of_constraints = int(input('Enter number of problem constraints(Including sign restrictions): '))
constraints = []
for i in range(0, no_of_constraints):
    constraints.append(Constraint(input('Constraint #' + str(i + 1) + ': ')))

print('\n===========================================================================================\n')
if not Constraint.is_feasible_region_exists(constraints):
    print('Problem is not feasible')
else:
    plotter = Plotter()
    plotter.draw_constraint_lines(constraints)
    shape_type, feasible_region_bound_Xs, feasible_region_bound_Ys = Constraint.get_feasible_region(constraints)
    plotter.draw_feasible_region(shape_type, feasible_region_bound_Xs, feasible_region_bound_Ys)
    has_secondary_optimize_point,is_optimal_solution_infinite, optimal_value, optimal_point, optimal_line = objective_function.get_optimal_solution(feasible_region_bound_Xs, feasible_region_bound_Ys)

    isoprofit_Xs, isoprofit_Ys, x_lim, y_lim = objective_function.get_isoprofit_line(feasible_region_bound_Xs, feasible_region_bound_Ys)
    plotter.draw_isoprofit_line(isoprofit_Xs, isoprofit_Ys)
    if is_optimal_solution_infinite:
        print('Optimal solution is infinite!')
    else:
        print('Optimal Value: ' + str(optimal_value))
        if has_secondary_optimize_point:
            print('Problem does not have a unique answer!')
            print('One of optimal points: x1 = ' + str(optimal_point[0]) + ', x2 = ' + str(optimal_point[1]) )
        else:
            print('Problem has a unique answer!')
            print('Optimal point: x1 = ' + str(optimal_point[0]) + ', x2 = ' + str(optimal_point[1]) )

        plotter.draw_optimize_solution(has_secondary_optimize_point, optimal_value, optimal_point, optimal_line)
    plotter.show(x_lim, y_lim)