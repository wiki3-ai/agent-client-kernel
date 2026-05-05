import ollama
from PIL import Image
import io
import base64

# Ollama server configuration
OLLAMA_HOST = "http://192.168.64.1:11434"
MODEL = "x/flux2-klein"

def generate_image(prompt, output_path="cat_flux.png"):
    """Generate an image using Ollama's Flux model"""
    
    print(f"Connecting to Ollama at {OLLAMA_HOST}...")
    client = ollama.Client(host=OLLAMA_HOST)
    
    print(f"Generating image with {MODEL}...")
    print(f"Prompt: {prompt}")
    
    try:
        # Flux uses the generate endpoint for image generation
        response = client.generate(
            model=MODEL,
            prompt=prompt,
            stream=False
        )
        
        # Check response type and attributes
        print(f"Response type: {type(response)}")
        print(f"Response dir: {[a for a in dir(response) if not a.startswith('_')]}")
        
        # Try to access image attribute directly
        if hasattr(response, 'image'):
            image_data = response.image
            
            # Handle both base64 string and bytes
            if isinstance(image_data, str):
                print("Image is base64 string, decoding...")
                image_bytes = base64.b64decode(image_data)
            else:
                print(f"Image is type: {type(image_data)}")
                image_bytes = image_data
            
            # Open and save the image
            img = Image.open(io.BytesIO(image_bytes))
            img.save(output_path)
            print(f"✓ Image saved to: {output_path}")
            
            # Display image info
            print(f"  Size: {img.size[0]}x{img.size[1]}")
            print(f"  Format: {img.format}")
            
            return True
            
        else:
            print("No 'image' attribute found")
            # Try to convert to dict
            if hasattr(response, '__dict__'):
                print(f"Response __dict__: {response.__dict__}")
            return False
            
    except Exception as e:
        print(f"✗ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    prompt = "An adorable fluffy cat with bright curious eyes, soft studio lighting, clean background, high-quality pet photography style"
    
    success = generate_image(prompt, "cat_flux.png")
    
    if success:
        print("\n🎉 Image generation successful!")
