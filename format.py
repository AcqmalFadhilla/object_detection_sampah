import os
import shutil


def separate_annotations_and_images(dataset_path, output_folder):
    # Membuat path absolut dari dataset_path dan output_folder
    dataset_path = os.path.abspath(dataset_path)
    output_folder = os.path.abspath(output_folder)

    # Membuat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Mendapatkan list file dalam folder dataset
    files = os.listdir(dataset_path)

    # Memisahkan file-file gambar dan file-file XML
    image_files = [file for file in files if file.endswith('.jpg')]
    xml_files = [file for file in files if file.endswith('.xml')]

    # Memindahkan file-file gambar ke folder images
    for image_file in image_files:
        shutil.move(os.path.join(dataset_path, image_file), os.path.join(output_folder, 'images', image_file))

    # Memindahkan file-file XML ke folder annotations
    for xml_file in xml_files:
        shutil.move(os.path.join(dataset_path, xml_file), os.path.join(output_folder, 'annotations', xml_file))

# Contoh pemanggilan fungsi
dataset_path = "Dataset/image"  # Ubah sesuai dengan path dataset Anda
output_folder = "Data"  # Ubah sesuai dengan nama folder output yang diinginkan
separate_annotations_and_images(dataset_path, output_folder)