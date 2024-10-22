import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import gradio as gr
import time
import gc
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# 1. Spatial Attention Module
# -----------------------------

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

# -----------------------------
# 2. Image VAE Definition
# -----------------------------

class ImageVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(ImageVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim * 2)
        )
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(-1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid()
        )

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

# -----------------------------
# 3. Visual Fractal Brain Definition
# -----------------------------

class VisualFractalBrain(nn.Module):
    def __init__(self, latent_dim=256, max_depth=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_depth = max_depth
        
        self.root = VisualFractalNeuron(latent_dim, latent_dim, max_depth=max_depth)
        self.attention = SpatialAttention(latent_dim)
        
        self.feature_extraction = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, x):
        features = self.feature_extraction(x)
        attended_features = self.attention(features)
        output = self.root(attended_features)
        return output

class VisualFractalNeuron(nn.Module):
    def __init__(self, input_dim, output_dim, depth=0, max_depth=3):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        # Visual processing components
        self.visual_synapse = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        
        self.visual_soma = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Initialize ModuleList correctly
        self._children_list = nn.ModuleList([])
        
        if depth < max_depth:
            for _ in range(2):
                child = VisualFractalNeuron(output_dim, output_dim, depth + 1, max_depth)
                self._children_list.append(child)

    def forward(self, x):
        x = self.visual_synapse(x)
        x = self.visual_soma(x)
        
        if len(self._children_list) > 0:
            child_outputs = []
            for child in self._children_list:
                child_outputs.append(child(x))
            x = torch.mean(torch.stack(child_outputs), dim=0)
        
        return x

# -----------------------------
# 4. Image Processor Class
# -----------------------------

class ImageProcessor:
    def __init__(self, latent_dim=256, max_depth=3):
        print("Initializing ImageProcessor")
        
        # Enhanced CUDA setup
        if not torch.cuda.is_available():
            print("CUDA is not available. Please install CUDA and the appropriate PyTorch version.")
            print("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            raise RuntimeError("CUDA is required for this application.")
        
        # Select GPU and print info
        self.device = self.setup_cuda()
        self.print_gpu_info()
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Initialize components
        print("Initializing VAE...")
        self.vae = ImageVAE(latent_dim).to(self.device)
        
        print(f"Initializing FractalBrain with max_depth={max_depth}...")
        self.fractal_brain = VisualFractalBrain(latent_dim, max_depth).to(self.device)
        
        # Load models if they exist
        self.load_models()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        print("ImageProcessor initialization complete")

    def setup_cuda(self):
        """Setup CUDA device with the most available memory"""
        if torch.cuda.device_count() > 1:
            # Get GPU with most free memory
            max_free_mem = 0
            selected_device = 0
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_mem > max_free_mem:
                    max_free_mem = free_mem
                    selected_device = i
            
            torch.cuda.set_device(selected_device)
            return torch.device(f"cuda:{selected_device}")
        
        return torch.device("cuda:0")

    def print_gpu_info(self):
        """Print detailed GPU information"""
        print("\nGPU Information:")
        print(f"Using: {torch.cuda.get_device_name(self.device)}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"Selected GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}\n")

    def clean_gpu_memory(self):
        """Clean GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def train_vae(self, dataloader, epochs=10, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.vae.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                
                # Use mixed precision training
                with autocast():
                    optimizer.zero_grad(set_to_none=True)
                    recon_images, mu, logvar = self.vae(images)
                    loss = self.vae.loss_function(recon_images, images, mu, logvar)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    used_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                          f"Loss: {loss.item():.4f}, GPU Memory Used: {used_memory:.1f}MB")
                    self.clean_gpu_memory()
                    
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
            
            # Save models after each epoch
            self.save_models(checkpoint_path=f'checkpoint_epoch_{epoch+1}.pth')
            self.clean_gpu_memory()

    def train_fractal_brain(self, dataloader, epochs=5, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.fractal_brain.parameters(), lr=learning_rate)
        self.fractal_brain.train()
        self.vae.eval()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                
                with autocast():
                    with torch.no_grad():
                        z, _, _ = self.vae.encode(images)
                    
                    optimizer.zero_grad(set_to_none=True)
                    processed_z = self.fractal_brain(z)
                    loss = F.mse_loss(processed_z, z)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    used_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                          f"Loss: {loss.item():.4f}, GPU Memory Used: {used_memory:.1f}MB")
                    self.clean_gpu_memory()
            
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
            
            # Save models after each epoch
            self.save_models(checkpoint_path=f'fractal_checkpoint_epoch_{epoch+1}.pth')
            self.clean_gpu_memory()

    def process_image(self, image):
        self.vae.eval()
        self.fractal_brain.eval()
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            z, mu, logvar = self.vae.encode(image_tensor)
            processed_z = self.fractal_brain(z)
            processed_image = self.vae.decode(processed_z)
            output_image = processed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            return output_image

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image.to(self.device)

    def save_models(self, checkpoint_path=None):
        """Save models and optionally create a checkpoint"""
        # Save main model files
        torch.save(self.vae.state_dict(), 'vae.pth')
        torch.save(self.fractal_brain.state_dict(), 'fractal_brain.pth')
        print("Models saved to vae.pth and fractal_brain.pth")
        
        # Save checkpoint if path is provided
        if checkpoint_path:
            torch.save({
                'vae_state_dict': self.vae.state_dict(),
                'fractal_brain_state_dict': self.fractal_brain.state_dict(),
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    def load_models(self):
        """Load models if they exist"""
        try:
            if os.path.exists('vae.pth') and os.path.exists('fractal_brain.pth'):
                print("Loading saved models...")
                self.vae.load_state_dict(torch.load('vae.pth', map_location=self.device))
                self.fractal_brain.load_state_dict(torch.load('fractal_brain.pth', map_location=self.device))
                print("Models loaded successfully")
                return True
            else:
                print("No pre-trained models found")
                return False
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def gradio_process_image(self, image):
        if image is None:
            return None, "No image provided"
        try:
            start_time = time.time()
            processed_image = self.process_image(image)
            processing_time = time.time() - start_time
            gpu_memory = f", GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB" if torch.cuda.is_available() else ""
            return processed_image, f"Processing time: {processing_time:.2f} seconds{gpu_memory}"
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

# -----------------------------
# 5. Main Execution
# -----------------------------


def main():
    print("Starting application...")
    try:
        # Set CUDA device options
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("CUDA Optimization settings enabled")
        
        # Initialize processor
        processor = ImageProcessor(latent_dim=256, max_depth=3)
        
        # Check if models need training
        if not os.path.exists('vae.pth') or not os.path.exists('fractal_brain.pth'):
            print("Training new models...")
            
            # Load dataset with GPU optimizations
            batch_size = 128 if torch.cuda.is_available() else 32
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4 if torch.cuda.is_available() else 0,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if torch.cuda.is_available() else False
            )
            
            print("Training VAE...")
            processor.train_vae(dataloader, epochs=10)
            
            print("Training Fractal Brain...")
            processor.train_fractal_brain(dataloader, epochs=5)
        else:
            print("Using pre-trained models")

        # Processing functions for Gradio
        def process_image_with_stats(image):
            if image is None:
                return None, "No image provided"
            try:
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                    start_time = time.time()
                    processed_image = processor.process_image(image)
                    processing_time = time.time() - start_time
                    gpu_memory = f", GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB" if torch.cuda.is_available() else ""
                    return processed_image, f"Processing time: {processing_time:.2f} seconds{gpu_memory}"
            except Exception as e:
                return None, f"Error processing image: {str(e)}"

        def update_gpu_info():
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                return f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB"
            return "Running on CPU"

        # Create Gradio interface
        with gr.Blocks(title="Fractal Vision Processor") as demo:
            gr.Markdown("# Fractal Vision Processor")
            gr.Markdown("Transform images using fractal neural networks")
            
            with gr.Tab("Process Images"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="numpy", label="Input Image")
                        process_btn = gr.Button("Process Image", variant="primary")
                        gpu_info = gr.Textbox(label="GPU Status", value=update_gpu_info())
                        refresh_btn = gr.Button("Refresh GPU Info")
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Processed Output")
                        info_text = gr.Textbox(label="Processing Info")
                
                process_btn.click(
                    process_image_with_stats,
                    inputs=[input_image],
                    outputs=[output_image, info_text]
                )
                refresh_btn.click(
                    update_gpu_info,
                    outputs=[gpu_info]
                )
            
            with gr.Tab("Model Information"):
                gr.Markdown("### Model Architecture")
                architecture_info = f"""
                - VAE Latent Dimension: 256
                - Fractal Brain Depth: 3
                - Processing Resolution: 128x128
                - Device: {"CUDA" if torch.cuda.is_available() else "CPU"}
                - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}
                """
                gr.Markdown(architecture_info)
                
                if torch.cuda.is_available():
                    gpu_details = f"""
                    ### GPU Information
                    - CUDA Version: {torch.version.cuda}
                    - Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB
                    - Device Name: {torch.cuda.get_device_name(0)}
                    """
                    gr.Markdown(gpu_details)
                
                training_info = """
                ### Training Information
                - Dataset: CIFAR-10
                - Training Images: 50,000
                - VAE Training Epochs: 10
                - Fractal Brain Training Epochs: 5
                - Image Resolution: 128x128
                """
                gr.Markdown(training_info)
            
            # Add examples tab if you want to show sample images
            with gr.Tab("Examples"):
                gr.Markdown("""
                ### Usage Examples
                1. Upload an image using the 'Process Images' tab
                2. Click 'Process Image' to transform it
                3. The processed image will appear on the right
                4. GPU memory usage and processing time will be displayed
                """)

        # Launch the interface
        demo.queue()  # Enable queuing for better handling of concurrent users
        demo.launch(
            server_name="0.0.0.0",  # Makes the server accessible from other devices
            share=False,  # Set to True if you want to generate a public URL
            server_port=7860,  # Default Gradio port
            quiet=True  # Reduce console output
        )

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()