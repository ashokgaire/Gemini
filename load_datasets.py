from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, BatchSampler
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

def image_to_tensor(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).to(device)

def resize_image(image, target_size=(224, 224)):
    return transforms.Resize(target_size)(image)

def create_dataset(data_folder, batch_size=8):
    dataset_dict = {'html': [], 'text': [], 'image': []}

    for company_folder in os.listdir(data_folder):
        company_path = os.path.join(data_folder, company_folder)

        if os.path.isdir(company_path):
            html_path = os.path.join(company_path, 'index.html')
            image_path = os.path.join(company_path, 'selenium_full_screenshot.png')

            if os.path.exists(html_path) and os.path.exists(image_path):
                with open(html_path, 'r', encoding='utf-8') as html_file:
                    html_content = html_file.read()

                    text = extract_text_from_html(html_content)

                    # Open, convert to grayscale, resize, and then convert to tensor
                    try:
                        with open(image_path, 'rb') as image_file:
                            image = Image.open(image_file).convert('L')  # 'L' mode stands for grayscale
                            image = resize_image(image)
                            image_tensor = image_to_tensor(image)

                        dataset_dict['html'].append(html_content)
                        dataset_dict['text'].append(text)
                        dataset_dict['image'].append(image_tensor)

                    except Exception as e:
                        print(f"Error loading image for {company_folder}: {e}")

    dataset = Dataset.from_dict(dataset_dict)
    output_dataset_path = "html_text_image_dataset"
    dataset.save_to_disk(output_dataset_path)


    # Create a DataLoader with a BatchSampler
    batch_sampler = BatchSampler(range(len(dataset)), batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    return dataloader

if __name__ == "__main__":
    data_folder = "prepared_data"
    output_dataset_path = "html_text_image_dataset"

    dataloader = create_dataset(data_folder, batch_size=8)
