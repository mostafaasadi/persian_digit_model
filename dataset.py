import os
import random
from PIL import Image, ImageDraw, ImageFont

# Configuration
DIGITS = "۰۱۲۳۴۵۶۷۸۹"
FONT_PATHS = [
    "fonts/is.ttf",
    "fonts/v.ttf"
]
FONT_SIZE = 30
IMAGE_SIZE = (30, 40)
SAMPLES_PER_CLASS = 10000
OUTPUT_DIR = "captcha_dataset"

# check if the output directory exists, if not create it
os.makedirs(OUTPUT_DIR, exist_ok=True)
for digit in DIGITS:
    os.makedirs(os.path.join(OUTPUT_DIR, digit), exist_ok=True)

# Load available fonts
available_fonts = []
for font_path in FONT_PATHS:
    if os.path.exists(font_path):
        available_fonts.append(font_path)
    else:
        print(f"Failded to load font: {font_path}")

if not available_fonts:
    print("No fonts available. Please check the font paths.")
    exit()


def add_point_noise(image, noise_level=0.05):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    num_noise_points = int(noise_level * width * height)
    for _ in range(num_noise_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        # make random noise
        if random.random() < 0.5:
            draw.point((x, y), fill=0)
        else:
            draw.point((x, y), fill=255)
    return image


for digit in DIGITS:
    for i in range(SAMPLES_PER_CLASS):
        bg = Image.new("RGB", IMAGE_SIZE, color=(255, 255, 255))

        random_font_path = random.choice(available_fonts)
        try:
            font = ImageFont.truetype(random_font_path, FONT_SIZE)
        except IOError:
            font = ImageFont.truetype(available_fonts[0], FONT_SIZE)

        draw = ImageDraw.Draw(bg)
        text_color = (0, 0, 0)

        # Center the text
        try:
            bbox = draw.textbbox((0, 0), digit, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = (IMAGE_SIZE[0] - text_w) // 2 - bbox[0]
            text_y = (IMAGE_SIZE[1] - text_h) // 2 - bbox[1]
        except AttributeError:
            continue

        draw.text((text_x, text_y), digit, font=font, fill=text_color)

        # rotate the image
        angle = random.uniform(-45, 45)
        bg = bg.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255))

        bg = bg.convert("L")
        threshold = 128
        bg = bg.point(lambda p: 255 if p < threshold else 0, "1")

        # add point noise
        bg = add_point_noise(bg, noise_level=0.01)

        # save the image
        filename = os.path.join(OUTPUT_DIR, digit, f"{digit}_{i:04}.png")
        bg.save(filename)

print(
    f"Total samples: {SAMPLES_PER_CLASS * len(DIGITS)} in '{OUTPUT_DIR}' generated."
)
