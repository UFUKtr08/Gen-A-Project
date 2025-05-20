import os
from PIL import Image

# Fotoğrafların bulunduğu klasör
input_folder = "Persons/Ben"
output_folder = "Persons/Ben/resized_photos"

# Eğer çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki tüm fotoğrafları al
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Fotoğraf türlerine göre filtrele
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Fotoğrafı aç
        img = Image.open(image_path)

        # Fotoğrafı boyutlandır
        img = img.resize((512, 512))

        # Fotoğrafı kaydet
        img.save(output_path)

        # Fotoğrafı göstermek isterseniz
        img.show()

print("Tüm fotoğraflar başarıyla boyutlandırıldı.")