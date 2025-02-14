import os
import sys
import threading
from androguard.core.bytecodes.apk import APK
from PIL import Image
from enum import Enum


def log(func_name: str, to_log: str):
    print(f"Function: {func_name} is Logging: {to_log}")


def get_dex_bytes(apk: APK) -> bytes:
    for f in apk.get_files():
        if f.endswith(".dex"):
            yield apk.get_file(f)


class ImagesGenerator:
    class GenerateFrom(Enum):
        TXT = 1
        APK = 2

    def __init__(self):
        self.state = None
        self.pathToInputFolder = ""
        self.pathToOutputFolder = ""
        print("ImageGenerator() created.")

    def set_state(self, state: str):
        self.state = self.GenerateFrom.TXT if state == "TXT" else self.state == self.GenerateFrom.APK
        print(f"State changed to Generating from {self.state}")

    def set_paths(self):
        if self.state == self.GenerateFrom.TXT:
            self.pathToInputFolder = "hashes"

        elif self.state == self.GenerateFrom.APK:
            self.pathToInputFolder = "apks"

        self.pathToOutputFolder = "images"
        print(f"All paths has been set.")

    def generate_image_from_txt(self, in_filename: str, folder: str):
        print(f"Thread {threading.currentThread().ident} is generating images from TXT...")
        stream = bytes()
        amount_of_images_encoded = 0
        with open(in_filename) as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                for byte in line:
                    my_str_as_bytes = str.encode(byte)
                    stream += my_str_as_bytes
                current_len = len(stream)
                image = Image.frombytes(mode='L', size=(1, current_len), data=stream)
                image.save(f"{folder}/{line}.png")
                amount_of_images_encoded += 1

        print(f"Successfully encoded {amount_of_images_encoded} images.")

    def generate_image_from_apk(self, goodware_in, folder):
        print(f"Thread {threading.currentThread().ident} is generating images from APK...")
        for paths, names, filenames in os.walk(goodware_in):
            for filename in [f for f in filenames if f.endswith(".apk")]:
                tosave = folder + "/" + filename
                if os.path.isfile(f"D:/DexRay/{tosave}.png"):
                    print(f"file {filename} already exists .. skipping")
                    continue
                fullname = goodware_in + "/" + filename
                try:
                    apk = APK(fullname)
                    stream = bytes()
                    for s in get_dex_bytes(apk):
                        stream += s
                    current_len = len(stream)
                    image = Image.frombytes(mode='L', size=(1, current_len), data=stream)
                    image.save(f"D:/DexRay/{tosave}.png")
                except:
                    print(f"Snap! file {filename} can't be saved!")

    def generate_images(self):
        if self.state == self.GenerateFrom.TXT:
            goodware_in = self.pathToInputFolder + "/goodware_hashes.txt"
            malware_in = self.pathToInputFolder + "/malware_hashes.txt"
            goodware_out = self.pathToOutputFolder + "/txt/goodware"
            malware_out = self.pathToOutputFolder + "/txt/malware"

            goodware_thread = threading.Thread(target=self.generate_image_from_txt,
                                               args=(goodware_in, goodware_out))
            malware_thread = threading.Thread(target=self.generate_image_from_txt,
                                              args=(malware_in, malware_out))

            goodware_thread.start()
            malware_thread.start()

            goodware_thread.join()
            malware_thread.join()
            print("Done encoding... exiting.")
            sys.exit(0)


        else:
            goodware_in = "D:/DexRay/apks/goodware/apps/train"
            malware_in = "D:/DexRay/apks/malware"
            goodware_out = self.pathToOutputFolder + "/apk/goodware_out/train"
            malware_out = self.pathToOutputFolder + "/apk/malware_out"

            print(f"Path to goodware apks: {goodware_in}")
            print(f"Path to malware apks: {malware_in}")
            print(f"Path to goodware images: {goodware_out}")
            print(f"Path to malware images: {malware_out}")

            goodware_thread = threading.Thread(target=self.generate_image_from_apk,
                                               args=(goodware_in, goodware_out))
            malware_thread = threading.Thread(target=self.generate_image_from_apk,
                                              args=(malware_in, malware_out))

            goodware_thread.start()
            malware_thread.start()

            goodware_thread.join()
            malware_thread.join()
            print("Done encoding... exiting.")
            sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise Exception("[!] Argument can be APK or TXT only.")

    imageGenerator = ImagesGenerator()
    imageGenerator.set_state(sys.argv[1])
    imageGenerator.set_paths()
    imageGenerator.generate_images()

    """
    
    ```python3 DexRay.py -p "images/txt" -d "results_dir" -f "results_dir/scores.txt"```
    else:
        filename = sys.argv[1]
        destination_folder = sys.argv[2]
    try:
        apk = APK(filename)
        generate_png(apk, filename, destination_folder)

        print(f"Image successfully generated from {filename}")
    except Exception as e:
        print("[!] An exception occurred with: {}".format(filename))
        print("Exception: {}".format(e))
    """
