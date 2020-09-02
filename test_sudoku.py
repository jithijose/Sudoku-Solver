import extract_sudoku as es
import extract_number as ns
import solve_sudoku as ss
import matplotlib.pyplot as plt

   
img = 'images/Sudoku.jpeg'

image_sudoku = es.extract_sudoku(img)

grid = ns.extract_numbers(image_sudoku)

print(grid)

solution = ss.sudoku_solver(grid)
#plt.imshow(solution)

print(solution)
plt.show()

