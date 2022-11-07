import sys
import threading
from androguard.core.bytecodes.apk import APK
from PIL import Image

def log(func_name:str, to_log:str):
    print(f"Function: {func_name} is Logging: {to_log}")

def get_dex_bytes(apk: APK) -> bytes:
    for f in apk.get_files():
        if f.endswith(".dex"):
            yield apk.get_file(f)


def generate_png(apk: APK, filename: str, folder: str):
    stream = bytes()
    for s in get_dex_bytes(apk):
        stream += s
    current_len = len(stream)
    image = Image.frombytes(mode='L', size=(1, current_len), data=stream)
    image.save(f"{folder}/{filename}.png")


def generate_png_of_text(filename: str, folder: str):
    print(f"Thread {threading.get_native_id} is in generate_png_of_text function.")
    stream = bytes()
    amount_of_images_encoded = 0
    with open(filename) as file:
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


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("No arguments provided, therefor will decode goodware and malware hashes texts to png file.")
        goodware_thread = threading.Thread(target=generate_png_of_text, args=("goodware_hashes.txt", "images/goodware"))
        malware_thread = threading.Thread(target=generate_png_of_text, args=("malware_hashes.txt", "images/malware"))

        goodware_thread.start()
        malware_thread.start()

        goodware_thread.join()
        malware_thread.join()
        print("Done encoding... exiting.")
        sys.exit(0)

    elif len(sys.argv) < 3:
        raise Exception("[!] Usage: python3 apktoimage.py APK DESTINATION")

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

