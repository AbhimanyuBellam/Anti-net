# import os
# import csv


# def generate_csv_from_folder(input_folder, output_csv):
#     with open(output_csv, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         # csv_writer.writerow(['Image_Path', 'Label'])

#         for label in os.listdir(input_folder):
#             label_folder = os.path.join(input_folder, label)
#             if not os.path.isdir(label_folder):
#                 continue

#             for image_file in os.listdir(label_folder):
#                 if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                     # image_path = os.path.join(label_folder, image_file)
#                     image_path = image_file
#                     csv_writer.writerow([image_path, label])


# if __name__ == "__main__":
#     input_folder = "vehicles_dataset"
#     output_csv = "vehicles.csv"

#     generate_csv_from_folder(input_folder, output_csv)
#     print("CSV file generated successfully!")

import os
import shutil


def create_image_folders(input_folder, output_folder):
    os.mkdir(output_folder)
    for label in os.listdir(input_folder):
        label_folder = os.path.join(input_folder, label)
        if not os.path.isdir(label_folder):
            continue

        # output_label_folder = os.path.join(output_folder, label)
        # os.makedirs(output_label_folder, exist_ok=True)

        for image_file in os.listdir(label_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(label_folder, image_file)
                shutil.copy(image_path, output_folder)


if __name__ == "__main__":
    input_folder = "vehicles_dataset"
    output_folder = "vehicles_all"

    create_image_folders(input_folder, output_folder)
    print("Images copied to single folders successfully!")
