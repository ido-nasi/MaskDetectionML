import cv2
import torch
import os
from PIL import Image, ImageDraw, ImageFont

from ResNet18 import ResNet18
from main import CustomDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_PATH = "../predictions/predict_10.jpg"
INPUT_FOLDER = "../predictions"
batch_size = 1


def predict():
    test_dataset = CustomDataSet(INPUT_FOLDER)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load("final_model.pkl", map_location=torch.device('cpu')))

    model.eval()

    for _, (images, names) in enumerate(test_loader):
        images = images.to(device)

        outputs = model(images)
        outputs = torch.round(outputs)

        outputs = outputs.cpu().detach().numpy()

        return outputs[0]


def capture_frame():
    # capture video object
    vid = cv2.VideoCapture(0)

    if vid.isOpened():
        while True:
            ret, frame = vid.read()
            cv2.imshow('Mask Predictor', frame)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.imwrite(IMAGE_PATH, frame)
                break
    else:
        print("\033[931[!] Cannot connect to your camera")

    vid.release()
    cv2.destroyAllWindows()


def show_result(p: float):
    img = Image.open(IMAGE_PATH)
    d1 = ImageDraw.Draw(img)
    font_path = 'DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, size=36)

    text_size = d1.textsize("Without Mask" if p == 0.0 else "With Mask", font=font)
    x = y = 10
    padding = 10

    d1.rectangle((x - padding, y - padding,
                  x + text_size[0] + padding,
                  y + text_size[1] + padding),
                 fill=(0, 0, 0))

    d1.text((x, y),
            "Without Mask" if p == 0.0 else "With Mask",
            fill=(255, 255, 255),
            font=font)
    img.show()


if __name__ == '__main__':
    capture_frame()
    prediction = predict()
    show_result(prediction)
    os.remove(IMAGE_PATH)
