import cv2


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def save_image(path, image):
    cv2.imwrite(path, image)