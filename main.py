# Note:
# the color blocks of sudoku should be like:
#                red(coord_1)          blue(coord_1)
#                             --------
#                             |sudoku|
#                             --------
#                blue(coord_2)         red(coord_2)
#

# code start from here
# import libs
import cv2
import numpy as np
from matplotlib import pyplot as plt
import keras

pic_path = "C:/Users/liuyu/Desktop/standard_number-10.jpg"
weights_path = 'c:/users/liuyu/desktop/data_set/weight/weights.h5'

class Img:
    
    def __init__(self, img_bgr):
        # path of image(just for testing)
        self.img_set = []
        self.img_name_set = []
        self.img_type = []

        # code for distortion
        # pass

        # Read and resize
        img_bgr = cv2.resize(img_bgr, (600, 800))
        self.img_set.append(img_bgr)
        self.img_name_set.append('BGR image')
        self.img_type.append(None)

        # Find the red blocks coordinate
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self.img_set.append(img_hsv)
        self.img_name_set.append('HSV image')
        self.img_type.append(None)
        # the boundary of red and blue
        lower_red_1 = np.array([0, 80, 80])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([156, 50, 50])
        upper_red_2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([160, 255, 255])
        # get two parts of red and blue
        red_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
        red_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
        blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        self.img_set.append(red_1)
        self.img_name_set.append('red boundary I')
        self.img_type.append('gray')
        self.img_set.append(red_2)
        self.img_name_set.append('red boundary II')
        self.img_type.append('gray')
        self.img_set.append(blue)
        self.img_name_set.append('blue boundary')
        self.img_type.append('gray')
        # combine the two red parts
        red = cv2.bitwise_or(red_1, red_2)
        self.img_set.append(red)
        self.img_name_set.append('red boundary')
        self.img_type.append('gray')
        # find the edge
        contour_red, hierarchy_red = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_blue, hierarchy_blue = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_bound = img_bgr.copy()
        cv2.drawContours(img_bound, contour_red, -1, (0, 255, 0), 1)
        cv2.drawContours(img_bound, contour_blue, -1, (0, 0, 255), 1)
        self.img_set.append(img_bound)
        self.img_name_set.append('color blocks')
        self.img_type.append(None)
        # find the center of the two largest boundaries
        max_red_1 = 0
        max_red_2 = 0
        max_blue_1 = 0
        max_blue_2 = 0
        max_red_index_1 = 0
        max_red_index_2 = 0
        max_blue_index_1 = 0
        max_blue_index_2 = 0
        # find the two largest boundaries
        for i in range(np.size(contour_red)):
            if np.size(contour_red[i]) > max_red_2:
                max_red_2 = max_red_1
                max_red_1 = np.size(contour_red[i])
                max_red_index_2 = max_red_index_1
                max_red_index_1 = i
                if max_red_2 > max_red_1:
                    tmp = max_red_1
                    max_red_1 = max_red_2
                    max_red_2 = tmp
                    tmp = max_red_index_1
                    max_red_index_1 = max_red_index_2
                    max_red_index_2 = tmp
        for i in range(np.size(contour_blue)):
            if np.size(contour_blue[i]) > max_blue_2:
                max_blue_2 = max_blue_1
                max_blue_1 = np.size(contour_blue[i])
                max_blue_index_2 = max_blue_index_1
                max_blue_index_1 = i
                if max_blue_2 > max_blue_1:
                    tmp = max_blue_1
                    max_blue_1 = max_blue_2
                    max_blue_2 = tmp
                    tmp = max_blue_index_1
                    max_blue_index_1 = max_blue_index_2
                    max_blue_index_2 = tmp
        # get the center of the boundaries
        coord_red_x_1 = 0
        coord_red_y_1 = 0
        for i in range(len(contour_red[max_red_index_1])):
            coord_red_x_1 += contour_red[max_red_index_1][i][0][0]
            coord_red_y_1 += contour_red[max_red_index_1][i][0][1]
        coord_red_x_1 = coord_red_x_1 // len(contour_red[max_red_index_1])
        coord_red_y_1 = coord_red_y_1 // len(contour_red[max_red_index_1])
        print('The first red block is at:' + str([coord_red_x_1, coord_red_y_1]))

        coord_red_x_2 = 0
        coord_red_y_2 = 0
        for i in range(len(contour_red[max_red_index_2])):
            coord_red_x_2 += contour_red[max_red_index_2][i][0][0]
            coord_red_y_2 += contour_red[max_red_index_2][i][0][1]
        coord_red_x_2 = coord_red_x_2 // len(contour_red[max_red_index_2])
        coord_red_y_2 = coord_red_y_2 // len(contour_red[max_red_index_2])
        print('The second red block is at:' + str([coord_red_x_2, coord_red_y_2]))

        coord_blue_x_1 = 0
        coord_blue_y_1 = 0
        for i in range(len(contour_blue[max_blue_index_1])):
            coord_blue_x_1 += contour_blue[max_blue_index_1][i][0][0]
            coord_blue_y_1 += contour_blue[max_blue_index_1][i][0][1]
        coord_blue_x_1 = coord_blue_x_1 // len(contour_blue[max_blue_index_1])
        coord_blue_y_1 = coord_blue_y_1 // len(contour_blue[max_blue_index_1])
        print('The first blue block is at:' + str([coord_blue_x_1, coord_blue_y_1]))

        coord_blue_x_2 = 0
        coord_blue_y_2 = 0
        for i in range(len(contour_blue[max_blue_index_2])):
            coord_blue_x_2 += contour_blue[max_blue_index_2][i][0][0]
            coord_blue_y_2 += contour_blue[max_blue_index_2][i][0][1]
        coord_blue_x_2 = coord_blue_x_2 // len(contour_blue[max_blue_index_2])
        coord_blue_y_2 = coord_blue_y_2 // len(contour_blue[max_blue_index_2])
        print('The first blue block is at:' + str([coord_blue_x_2, coord_blue_y_2]))

        # Perspective transformation
        # using cross product to examine the direction
        if coord_blue_x_1 < coord_blue_x_2 and coord_blue_y_1 > coord_blue_y_2:
            tmp = coord_blue_x_1
            coord_blue_x_1 = coord_blue_x_2
            coord_blue_x_2 = tmp
            tmp = coord_blue_y_1
            coord_blue_y_1 = coord_blue_y_2
            coord_blue_y_2 = tmp
        if coord_red_x_1 > coord_red_x_2 and coord_red_y_1 > coord_red_y_2:
            tmp = coord_red_x_1
            coord_red_x_1 = coord_red_x_2
            coord_red_x_2 = tmp
            tmp = coord_red_y_1
            coord_red_y_1 = coord_red_y_2
            coord_red_y_2 = tmp

        # perspective transformation
        coord_from = np.float32([[coord_red_x_1, coord_red_y_1], [coord_blue_x_1, coord_blue_y_1],
                                 [coord_blue_x_2, coord_blue_y_2], [coord_red_x_2, coord_red_y_2]])
        coord_to = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
        perspective = cv2.getPerspectiveTransform(coord_from, coord_to)
        img_perspective = cv2.warpPerspective(img_bgr, perspective, (600, 600))
        coord_from = np.float32([[12, 12], [588, 0], [0, 588], [588, 588]])
        coord_to = np.float32([[0, 0], [576, 0], [0, 576], [576, 576]])
        perspective = cv2.getPerspectiveTransform(coord_from, coord_to)
        img_perspective = cv2.warpPerspective(img_perspective, perspective, (576, 576))
        self.img_set.append(img_perspective)
        self.img_name_set.append('perspective')
        self.img_type.append(None)

        # Otsu threshold
        img_threshold = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        img_threshold = clahe.apply(img_threshold)
        img_threshold = cv2.GaussianBlur(img_threshold, (3, 3), 0)
        ret, img_threshold = cv2.threshold(img_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.img_set.append(img_threshold)
        self.img_name_set.append('adaptive threshold')
        self.img_type.append('gray')

        # Image split
        self.num_img = []
        self.num_pos_set = []
        for i in range(9):
            for j in range(9):
                tmp = []
                for m in range(64):
                    tmp.append(self.img_set[-1][64 * i + m][64 * j: 64 * j + 64])
                self.num_img.append(tmp)
                self.num_pos_set.append('(' + str(i + 1) + ',' + str(j + 1) + ')')

    def show_sudoku_img(self):
        cv2.imshow("Origin", self.img_set[0])
        cv2.imshow("Sudoku", self.img_set[-1])

    def show_processed_img(self):  # for debugging
        # Get the images processed
        for i in range(len(self.img_set)):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.img_set[i], cmap=self.img_type[i])
            plt.title(self.img_name_set[i])
        plt.show()

    def show_num_img(self):
        for i in range(81):
            plt.subplot(9, 9, i + 1)
            plt.imshow(self.num_img[i], cmap='gray')
            plt.title(self.num_pos_set[i])
        plt.show()


class Network:
    
    def __init__(self):
        self.img_width, self.img_height, self.channels = 64, 64, 1
        self.input_shape = (self.img_width, self.img_height, self.channels)
        self.model = keras.models.Sequential()
        # build the neural network
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        # load the trained weights
        self.model.load_weights(weights_path, by_name=True)
        self.xr = []

    def predict(self, img):
        xr = keras.preprocessing.image.img_to_array(img)
        xr = np.expand_dims(xr, axis=0)
        xr /= 225.0
        num = self.model.predict_classes(xr)[0]
        num = str(num)
        return num


class Sudoku:

    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.result = []
        for i in range(9):
            for j in range(9):
                self.sudoku[i][j] = int(self.sudoku[i][j])

    def solve(self):
        # check if every number in line is unique within line
        def if_line_right(now, line, column, n):
            for i in range(9):
                if now[line][i] == n:
                    return False
            return True

        # check if every number in column is unique within column
        def if_column_right(now, line, column, n):
            for i in range(9):
                if now[i][column] == n:
                    return False
            return True

        # check if every number in the 9x9 square is unique within the 9x9 square
        def if_square_right(now, line, column, n):
            left_x = 3 * (column // 3)
            up_y = 3 * (line // 3)
            for i in range(3):
                for j in range(3):
                    if now[up_y + i][left_x + j] == n:
                        return False
            return True

        def if_current_right(now, line, column, n):
            return (if_line_right(now, line, column, n) and if_column_right(now, line, column, n) and \
                    if_square_right(now, line, column, n))

        def if_full(now):
            for i in range(9):
                for j in range(9):
                    if now[i][j] == 0:
                        return 0
            return 1

        def solve_sudoku(now, index):
            line = index // 9  # line starts from 0
            column = index % 9  # column starts from 0
            if if_full(now) or index >= 81:
                return [True, now]
            if now[line][column] != 0:
                return solve_sudoku(now, index + 1)
            for n in range(1, 10):
                if if_current_right(now, line, column, n):
                    now[line][column] = n
                    if solve_sudoku(now, index + 1)[0]:
                        return [True, now]
                    now[line][column] = 0
            return [False, now]

        self.result = (solve_sudoku(self.sudoku, 0)[1]).copy()
        print("complete sulution:")
        print(self.result)
        print("A solution to the sudoku is found!")
        for i in range(9):
            for j in range(9):
                self.result[i][j] = int(self.result[i][j]) - int(self.sudoku[i][j])
        return self.result


# main
if __name__ == '__main__':
    network = Network()
    # for testing start
    picture = cv2.imread(pic_path, cv2.IMREAD_COLOR)
    # for testing end
    # for the robot start
    '''
    cap = cv2.VideoCapture("0.mp4")
    print("print q to quit the video and enter the next step!")
    while True:
        ret, picture = cap.read() 
        cv2.imshow("capture", picture)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    '''
    # for the robot end
    img = Img(picture)
    sudoku_input = []
    for i in range(9):
        sudoku_input.append([])
        for j in range(9):
            sudoku_input[i].append(network.predict(img.num_img[9 * i + j]))
    sudoku_real = Sudoku(sudoku_input)
    sudoku_output = (sudoku_real.solve()).copy()
    print(sudoku_output)