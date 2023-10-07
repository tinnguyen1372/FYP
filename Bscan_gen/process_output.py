from PIL import Image
import math

# Define the color map
color_map = {
    (255, 255, 255, 255): (255, 255, 255, 255),  # White (transparent)
    (246, 255, 0, 255): (255, 255, 0, 255),    # Yellow
    (255, 51, 0, 255): (255, 51, 0, 255),     # Red
}

# Define the threshold for color similarity
color_similarity_threshold = 50  # Adjust as needed

def color_similarity(color1, color2):
    # Calculate the Euclidean distance between two colors
    r1, g1, b1, a1 = color1
    r2, g2, b2, a2 = color2
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2 + (a1 - a2) ** 2)

def process_image(input_path, output_path):
    # Open the input image
    input_image = Image.open(input_path).convert("RGBA")

    # Create a new RGBA image with the same size
    output_image = Image.new("RGBA", input_image.size)

    # Load the pixels of both images
    input_pixels = input_image.load()
    output_pixels = output_image.load()

    # Iterate through each pixel in the input image
    for x in range(input_image.width):
        for y in range(input_image.height):
            pixel = input_pixels[x, y]
            # if not pixel == (255,255,255,255):
            #     print(pixel)
            # Check if the pixel color is in the color map
            if pixel in color_map:
                # Get the corresponding color value from the color map
                color_value = color_map[pixel]

                # Set the pixel color in the output image
                output_pixels[x, y] = color_value
            else:
                # If the color is not in the color map, find the closest color
                closest_color = min(color_map.keys(), key=lambda c: color_similarity(pixel, c))
                
                # Check if the closest color is within the threshold
                if color_similarity(pixel, closest_color) <= color_similarity_threshold:
                    output_pixels[x, y] = color_map[closest_color]
                else:
                    # If the color is not close to any color in the color map, set it to yellow with full opacity
                    output_pixels[x, y] = (255, 255, 0, 255),  # Yellow (RGBA format)

    # Save the output image
    output_image.save(output_path)


if __name__ == "__main__":
    input_healthy_directory = "./Data/Healthy/"
    input_defect_directory = "./Data/Defect/"
    output_healthy_directory = "./Dataset/Healthy/"
    output_defect_directory = "./Dataset/Defect/"
    N_trunk = input("Number of trunk: ")
    for i in range(1,int(N_trunk)):
        print("Processing Image Trunk number {}".format(i))
        healthy_format = "healthy{}.png".format(i)
        defect_format = "defect{}.png".format(i)
        process_image(input_healthy_directory+healthy_format, output_healthy_directory+healthy_format)
        process_image(input_defect_directory+defect_format,output_defect_directory+defect_format)
