import openai
import requests
import io
import tkinter as tk
from tkinter import Label, Entry, Button, Canvas, PhotoImage
from PIL import Image, ImageTk
import datetime

OPENAI_API_KEY = "my-api-key"
openai.api_key = OPENAI_API_KEY

# Function to generate an image from text
def generate_image():
    prompt = text_input.get()
    if not prompt:
        status_label.config(text="Please enter a description.")
        return

    status_label.config(text="Generating image...")

    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,  
            size="512x512"  
        )
        image_url = response["data"][0]["url"]

        image_data = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_data))

        tk_image = ImageTk.PhotoImage(image)

        canvas.image = tk_image
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        status_label.config(text="Image generated successfully!")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        image.save(filename)
        status_label.config(text=f"Image saved as '{filename}'")

    except Exception as e:
        status_label.config(text=f"Error: {e}")
def solve_equation():
    equation = equation_input.get()
    if not equation:
        status_label.config(text="Please enter an equation.")
        return

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Solve the following equation and provide only the roots:"},
                {"role": "user", "content": equation}
            ]
        )

        solution = response["choices"][0]["message"]["content"]
        status_label.config(text=f"Roots: {solution}")

    except Exception as e:
        status_label.config(text=f"Error: {e}")

root = tk.Tk()
root.title("AI Image Generator")
root.geometry("512x800")

Label(root, text="Enter text for image generation:").pack(pady=5)
text_input = Entry(root, width=80)
text_input.pack(pady=5)

generate_button = Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=10)

Label(root, text="Or enter equation to solve (e.g., x^2 - 4 = 0):").pack(pady=5)
equation_input = Entry(root, width=80)
equation_input.pack(pady=5)

solve_button = Button(root, text="Solve Equation", command=solve_equation)
solve_button.pack(pady=10)

status_label = Label(root, text="", fg="blue")
status_label.pack(pady=5)

canvas = Canvas(root, width=512, height=512, bg="white")
canvas.pack(pady=10)

root.mainloop()
