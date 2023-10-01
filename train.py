from booruClip import BooruCLIP
from PIL import Image
import requests
from tqdm import tqdm, trange
from io import BytesIO
import os

booru = BooruCLIP("./derpibooru/")
    
api_key = "PUT YOUR API KEY HERE"
derpibooru_search_url = "https://derpibooru.org/api/v1/json/search/images?key="+api_key+"&q=*&page="

total = 0
per_page = 50
pages = 0
last_page = 1
# get last page
if not os.path.exists("lastPage"):
    with open("lastPage", "w") as f:
        f.write("1")
        f.close()
with open("lastPage", "r") as f:
    last_page = int(f.read())
    f.close()
def get_page(page):
    global total
    global pages
    r = requests.get(derpibooru_search_url+str(page))
    if r.status_code == 200:
        r = r.json()
        total = r["total"]
        pages = r["total"]//per_page
        # tqdm gets removed after it's done
        for i in tqdm(r["images"], desc="Page "+str(page), leave=True):
            try:
                image = requests.get(i["representations"]["full"])
                if image.status_code == 200:
                    image = Image.open(BytesIO(image.content))
                    booru.new_image(image, i["tags"], i["id"])
            except Exception as e:
                print(e)
                pass
    else:
        print("Error:", r.status_code, r.text)
    with open("lastPage", "w") as f:
        f.write(str(page))
        f.close()

print("Initializing...")
get_page(1)
print("Total:", total)
print("Pages:", pages)
print("Training on Derpibooru...")
for i in trange(last_page+1, pages):
    get_page(i)
    
# This is extremely slow so as to not hammer the API, but it could be sped up a lot by pre-downloading the images, creating the embeds in batches,
# using multiple threads, by editing your clip-server tensor-flow.yaml to use multiple instances of the CLIP server, etc. Proof of concept really.