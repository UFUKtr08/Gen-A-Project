import os
import pandas as pd

# Fotoğrafların bulunduğu klasörün yolu
input_folder = r"Persons\\Ben\\resized_photos"  # Ham dize kullanarak yolu düzelttik

# Klasördeki tüm .jpg ve .png dosyalarını al
image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

# Etiketleri fotoğraf isimlerinden çıkarma ve virgülle ayırarak başına <fkylmz> ekleme
labels = ['<fkylmz> ' + ', '.join(os.path.splitext(os.path.basename(path))[0].split(',')) for path in image_paths]

# CSV dosyasını oluşturma
data = {
    'image_path': image_paths,
    'label': labels
}

df = pd.DataFrame(data)

# CSV dosyasına kaydetme
df.to_csv('photo_labels.csv', index=False)