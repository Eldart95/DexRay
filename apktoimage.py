import sys
import math
from androguard.core.bytecodes.apk import APK
from PIL import Image


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
    stream = bytes()
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


if __name__ == "__main__":

    if len(sys.argv) == 1:
        generate_png_of_text("goodware_hashes.txt", "images/goodware")
        generate_png_of_text("malware_hashes.txt", "images/malware")
        print("Think im done")

    """
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
        print("[!] An exception occured with: {}".format(filename))
        print("Exception: {}".format(e))
    """