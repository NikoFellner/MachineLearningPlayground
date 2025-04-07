from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

# create the checker class
checkerboard = Checker(resolution=1000, tile_size=50)
# draw a checkerboard, saved in .checkerboard_output
if checkerboard.resolution % (2 * checkerboard.tile_size) == 0:
    checkerboard.draw()
    #checkerboard.show()

# Create the Circle class
circle = Circle(resolution=1024, radius=50, position=(256, 256))
circle.draw()
#circle.show()


spectrum = Spectrum(resolution=256)
spectrum.draw()
spectrum.show()


# Create the ImageGenerator Class
image_generator = ImageGenerator(file_path="exercise_data/", label_path="Labels.json", batch_size=14, image_size=[32, 32, 3], shuffle=True, rotation=True, mirroring=True)
image_generator2 = ImageGenerator(file_path="exercise_data/", label_path="Labels.json", batch_size=12, image_size=[256, 256, 3])

#image_generator.show()
#image_generator2.show()